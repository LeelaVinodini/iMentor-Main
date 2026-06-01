const log = require("../utils/logger");
const { DAG_PLANNER_PROMPT_TEMPLATE } = require("../config/promptTemplates");

// ... (Rest of imports stay same)
const geminiService = require("./geminiService");
const groqService = require("./groqService");
const ollamaService = require("./ollamaService");
const anthropicService = require("./anthropicService");
const mistralService = require("./mistralService");
const TaskGraphManager = require("./taskGraphManager");

/**
 * Service to decompose complex queries into a Directed Acyclic Graph (DAG) of tasks.
 */
async function decomposeQuery(query, requestContext) {
  const { llmProvider, ...llmOptions } = requestContext;

  // Use a fast but capable model for planning
  const plannerOptions = {
    ...llmOptions,
    temperature: 0.2, // Low temperature for consistent JSON structure
  };

  const llmService =
    llmProvider === "ollama" ? ollamaService :
      (llmProvider === "groq" ? groqService :
        (llmProvider === "anthropic" ? anthropicService :
          (llmProvider === "mistral" ? mistralService : geminiService)));

  // If we are in fallback or preferred mode, ensure we have the right model
  if (llmProvider === 'groq') plannerOptions.model = 'llama-3.1-8b-instant';
  if (llmProvider === 'gemini') plannerOptions.geminiModel = 'gemini-1.5-flash';

  const availableToolsJson = JSON.stringify(requestContext.availableTools || [], null, 2);
  const currentModeInstruction = requestContext.currentModeInstruction || "Standard decomposition.";
  const branchCount = Number.isFinite(requestContext.branchCount)
    ? Math.max(2, Math.min(3, Number(requestContext.branchCount)))
    : 3;

  const prompt = DAG_PLANNER_PROMPT_TEMPLATE(branchCount)
    .replace("{userQuery}", query)
    .replace("{available_tools_json}", availableToolsJson)
    .replace("{current_mode_tool_instruction}", currentModeInstruction);

  // log.info('AI', `Decomposing query via ${llmProvider}...`);

  try {
    const responseText = await llmService.generateContentWithHistory(
      [],
      prompt,
      "You are a meticulous AI planning agent specialized in hierarchical task decomposition.",
      plannerOptions
    );

    const jsonMatch = responseText.match(/```(json)?\s*([\s\S]+?)\s*```/);
    const jsonString = jsonMatch ? jsonMatch[2] : responseText;
    const parsedResponse = JSON.parse(jsonString);

    if (!parsedResponse.plans || !Array.isArray(parsedResponse.plans)) {
      throw new Error("Invalid decomposition format: Missing 'plans' array.");
    }

    log.success('AI', `Query decomposed into ${parsedResponse.plans.length} candidate plans`);
    return parsedResponse.plans;
  } catch (error) {
    log.warn('AI', `Decomposition failed, using direct answer fallback`);

    // Backward compatibility: If decomposition fails, create a single-step linear plan
    return [{
      name: "Direct Answer Plan",
      tasks: [
        {
          id: "primary_analysis",
          title: "Direct Answer",
          description: `Address the query: ${query}`,
          type: "reasoning",
          tool_call: null,
          dependsOn: []
        }
      ]
    }];
  }
}

module.exports = {
    decomposeQuery,
    decomposeTask: decomposeQuery  // alias — agentOrchestrator.js imports this name
};
