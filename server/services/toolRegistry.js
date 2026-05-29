// server/services/toolRegistry.js — Unified (Team3 structure + Team1-6 extra tools)
const log = require('../utils/logger');
const { performWebSearch } = require('./webSearchService.js');
const { conductDeepResearch } = require('./deepResearchOrchestrator.js'); // [Team1-6]
const { queryPythonRagService, queryKgService } = require('./toolExecutionService.js');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const geminiService = require('./geminiService');

// [Team1-6] Gamification grading from chat
let solveBountyInternal, Bounty, executeAgentTask;
try { solveBountyInternal = require('./gamificationService').solveBountyInternal; } catch(e) {}
try { Bounty = require('../models/Bounty'); } catch(e) {}
try { executeAgentTask = require('./agentOrchestrator').executeAgentTask; } catch(e) {}

async function queryAcademicService(query) {
    const pythonServiceUrl = process.env.PYTHON_RAG_SERVICE_URL;
    if (!pythonServiceUrl) {
        throw new Error("Academic search service is not configured on the server.");
    }
    const searchUrl = `${pythonServiceUrl}/academic_search`;
    
    try {
        // log.info('SYSTEM', `Academic search: ${query}`);
        const response = await axios.post(searchUrl, { query }, { timeout: 45000 });
        const papers = response.data?.results || [];
        
        const toolOutput = papers.length > 0
            ? "Found the following relevant academic papers:\n\n" + papers.map((p, index) => 
                `[${index + 1}] **${p.title || 'Untitled Paper'}**\n` +
                `   - Source: ${p.source || 'Unknown'}\n` +
                `   - URL: ${p.url || '#'}\n` +
                `   - Summary: ${p.summary ? p.summary.substring(0, 300) + '...' : 'No summary.'}`
              ).join('\n\n')
            : "No relevant academic papers were found for this query.";
            
        const references = papers.map((p, index) => ({
            number: index + 1,
            source: `${p.title || 'Untitled Paper'} (${p.source || 'N/A'})`,
            url: p.url || '#',
        }));

        return { references, toolOutput };

    } catch (error) {
        const errorMsg = error.response?.data?.error || `Academic Service Error: ${error.message}`;
        throw new Error(errorMsg);
    }
}

function assertResearchIntent(context, toolName) {
    if (context?.intent === 'research') return;
    throw new Error(`${toolName} is disabled unless intent is "research".`);
}

/**
 * Available tools registry.
 * Each tool has:
 *  - description: what the tool does (for LLM routing)
 *  - execute: async function(params, context) => { toolOutput, references }
 *  - requiredParams: array of required parameter names
 *  - meta: chaining/monitoring metadata
 */
const availableTools = {

  /**
   * Web Search: Searches the internet for real-time, up-to-date information on current events, public figures, or general knowledge. This tool is essential for answering questions that require information beyond the model's training cutoff or for verifying facts. It should be used when the user's query indicates a need for current or specific information that is likely to be found online.
   * The execute function performs a web search using the provided query and context, returning both the search results and any relevant references. It also includes an intent assertion to ensure that this tool is only used when the user's intent is research-oriented, preventing misuse in contexts where real-time information is not necessary.
   * The tool's metadata indicates that it belongs to the 'search' category, outputs text, can accept input from previous steps in a chain, has an average latency of 3000ms, is retryable on failure, and currently has no complementary tools listed.
   * Quriess the public internet via performWebSearch() to find current, real-time information relevant to the user's query. This is crucial for answering questions about recent events, specific facts, or any information that may not be included in the model's training data. The tool also checks that the user's intent is research-oriented before allowing execution, ensuring appropriate use of this resource-intensive tool.
   * best for: news, recent events, public figures, general knowledge, fact-checking, and any query that requires up-to-date information from the web.
   * Example usage: If the user asks "What are the latest developments in renewable energy?" or "Who won the World Cup in 2022?", the agent would route these queries to the web_search tool to retrieve current information from the internet.
   * Guard: Requires context.intent === 'research' to prevent misuse in non-research contexts, ensuring that this tool is only used when the user's intent is to gather information from the web.
   * Complements: academic_search (for deeper research), rag_search and kg_search (for document-specific queries), providing a broader search capability when needed.
   */
  web_search: {
    description: "Searches the internet for real-time, up-to-date information on current events, public figures, or general knowledge.",
    execute: async (params, context) => {
        assertResearchIntent(context, 'web_search');
        const { toolOutput, references } = await performWebSearch(params.query);
        return { references, toolOutput: toolOutput || "No results found from web search." };
    },
    requiredParams: ['query'],
    meta: {
        category: 'search',
        outputType: 'text',
        acceptsChainInput: true,
        avgLatencyMs: 3000,
        retryable: true,
        complementaryTools: [],
    },
  },

  /**
   * RAG Search: Searches the content of a specific, user-provided document to answer questions based on its text. This tool is designed for situations where the user has uploaded a document and wants to ask questions about its content. It uses a Retrieval-Augmented Generation (RAG) approach to find relevant information within the document and generate an answer based on that information. The execute function routes the query to a Python RAG service, which handles the retrieval and response generation, allowing for more accurate and context-aware answers based on the specific document's content.
   * best for: answering questions about the content of a specific document, such as "What are the key findings in this research paper?" or "Summarize the main points of this report."
   * Context: Uses context.documentContextName to identify which document to search within, and context.userId for any user-specific processing or logging. It also considers context.criticalThinkingEnabled and context.filter for enhanced retrieval and response generation based on the user's preferences and the nature of the query.
   * Complements: kg_search (for structured information from the same document), providing a comprehensive set of tools for document analysis and question-answering.
   * * Note: Unlike web_search/academic_search, this tool does NOT enforce an intent guard — it can be used whenever document-specific answers are needed.
   */
  rag_search: {
    description: "Searches the content of a specific, user-provided document to answer questions based on its text.",
    execute: async (params, context) => {
        return await queryPythonRagService(
            params.query, 
            context.documentContextName, 
            context.userId,
            context.criticalThinkingEnabled,
            context.filter
        );
    },
    requiredParams: ['query'],
    meta: {
        category: 'search',
        outputType: 'text',
        acceptsChainInput: true,
        avgLatencyMs: 5000,
        retryable: true,
        complementaryTools: ['kg_search'],
    },
  },

/**
 * Knowledge Graph Search: Retrieves structured facts and relationships from a document's pre-built knowledge graph. This tool is used to complement the RAG search by providing access to structured data extracted from the document, allowing for more precise answers to questions that require specific facts or relationships. The execute function queries a knowledge graph service with the provided query and context, returning relevant facts and references from the document's knowledge graph.
 * best for: questions that require specific facts, entities, or relationships from a document, such as "What are the main entities mentioned in this report?" or "How is concept X related to concept Y in this document?"
 * Context: Similar to RAG search, it uses context.documentContextName to identify the relevant document and context.userId for user-specific processing. It is designed to work in tandem with RAG search, providing a structured data perspective to complement the unstructured text retrieval of RAG.
+ * Note: Unlike web_search/academic_search, this tool does NOT enforce an intent guard — use it alongside rag_search for comprehensive document analysis. * Complements: rag_search (for unstructured text retrieval from the same document), offering a dual approach to document analysis and question-answering by providing both unstructured and structured information retrieval capabilities.
 * Note: The actual implementation of the knowledge graph service and how it extracts and structures data from documents is abstracted away in this tool definition, allowing for flexibility in how the knowledge graph is built and queried.
 * Output: Returns raw facts and relationships from the document's knowledge graph, which can be used directly in responses or as references for further reasoning steps in the agent's execution plan.
 */
  kg_search: {
    description: "Finds structured facts and relationships within a document's pre-built knowledge graph. Use this to complement RAG search.",
     execute: async (params, context) => {
        const facts = await queryKgService(params.query, context.documentContextName, context.userId);
        return { references: [], toolOutput: facts };
    },
    requiredParams: ['query'],
    meta: {
        category: 'search',
        outputType: 'text',
        acceptsChainInput: true,
        avgLatencyMs: 2000,
        retryable: true,
        complementaryTools: ['rag_search'],
    },
  },

  /**
   * Academic Search: Finds academic papers, research articles, and scholarly publications from scientific databases. This tool is essential for users who are looking for in-depth, credible sources on academic or technical topics. The execute function asserts that the user's intent is research-oriented before routing the query to an academic search service, which retrieves relevant papers based on the query. The results include both a summary of the findings and references to the original sources, allowing users to explore the information further if needed.
   * best for: peer-reviewed papers,academic research, literature reviews, finding scholarly articles, and any query that requires credible, in-depth sources from scientific databases.
   * Context: Uses context to ensure that the tool is only used when the user's intent is research-oriented, preventing misuse in non-research contexts. It also allows for user-specific processing or logging through context.userId if needed.
   * Guard: Requires context.intent === 'research' to ensure that this tool is used appropriately in contexts where users are seeking academic information.
   * Complements: web_search (for broader search capabilities) and deep_research (for comprehensive research tasks), providing a specialized tool for academic queries while allowing users to leverage other search tools as needed for a more holistic research approach.
   * Output: Formatted list of papers with title, source, URL, and truncated summary + structured references for each paper, enabling users to quickly assess the relevance of the results and access the original sources for more information.
   * Note: The actual implementation of the academic search service and how it retrieves and processes data from scientific databases is abstracted away in this tool definition, allowing for flexibility in how the academic search is performed and which databases are used.
   
  */
  academic_search: {
    description: "Finds academic papers, research articles, and scholarly publications from scientific databases.",
    execute: async (params, context) => {
        assertResearchIntent(context, 'academic_search');
        return await queryAcademicService(params.query);
    },
    requiredParams: ['query'],
    meta: {
        category: 'search',
        outputType: 'text',
        acceptsChainInput: true,
        avgLatencyMs: 8000,
        retryable: true,
        complementaryTools: ['web_search'],
    },
  },
  /**
   * Document Generation Tool
   * Creates a new document file (PPTX or DOCX format, saved as Markdown) on a given topic using the LLM's internal knowledge. This tool is designed for users who want to quickly generate structured documents based on a topic of interest, without needing to provide specific content. The execute function generates a detailed outline for a presentation or a comprehensive document based on the specified topic and document type, saves it as a Markdown file, and provides a download link for the generated document. It also includes error handling to ensure that any issues during generation are logged and communicated back to the user.
   * Best for: when users explicitly ask to create, make, build, or generate a file on a specific topic, such as "Generate a presentation on the impacts of climate change" or "Create a document about the history of artificial intelligence."
   * Params: topic (what to write about) and doc_type (pptx or docx, which determines the structure of the generated content).
   * Output: Download link + preview of the generated document content, allowing users to access the file directly and get a glimpse of the generated content before downloading.
   * Note: The actual file is saved as Markdown for simplicity and universal readability, but it can be easily converted to PPTX or DOCX format if needed. The tool focuses on generating well-structured content based on the topic, leveraging the LLM's knowledge to create informative and coherent documents.
   */
  generate_document: {
    description: "Generates a document file (like a PPTX or DOCX) on a given topic using internal knowledge. Use this when the user explicitly asks to 'create', 'make', 'build', or 'generate' a file. You must infer the 'topic' and 'doc_type' from the user's query.",
    execute: async (params, context) => {
        const { topic, doc_type } = params;
        const outputDir = path.join(__dirname, '..', 'assets', 'generated_docs');

        try {
            // Ensure output directory exists
            if (!fs.existsSync(outputDir)) {
                fs.mkdirSync(outputDir, { recursive: true });
            }

            // Generate structured content using LLM
            const docPrompt = doc_type === 'pptx'
                ? `Create a detailed presentation outline on "${topic}" with exactly 8-10 slides. For each slide provide:
SLIDE TITLE: <title>
CONTENT:
- Bullet point 1
- Bullet point 2
- Bullet point 3
SPEAKER NOTES: <brief notes>
---
Make it educational and well-structured.`
                : `Write a comprehensive, well-structured document on "${topic}".
Include:
1. Title and subtitle
2. Table of contents
3. Introduction
4. 3-5 main sections with subsections
5. Key takeaways / Summary
6. References (if applicable)
Format it in clean Markdown with proper headings, bullet points, and emphasis.`;

            const generatedContent = await geminiService.generateText(docPrompt, {
                apiKey: context?.apiKey,
                maxOutputTokens: 4096
            });

            // Save as markdown file (universally readable, can be converted to DOCX/PPTX)
            const timestamp = Date.now();
            const safeTopicName = topic.replace(/[^a-zA-Z0-9\s-]/g, '').replace(/\s+/g, '_').substring(0, 50);
            const fileName = `${safeTopicName}_${timestamp}.md`;
            const filePath = path.join(outputDir, fileName);

            fs.writeFileSync(filePath, generatedContent, 'utf-8');

            const downloadPath = `/api/upload/generated/${fileName}`;

            log.success('AI', `Document generated: ${fileName} (${generatedContent.length} chars)`);

            return {
                toolOutput: `✅ Document "${topic}" generated successfully as ${doc_type.toUpperCase()} format.\n\n📄 **Download**: [${fileName}](${downloadPath})\n\n---\n\n**Preview:**\n${generatedContent.substring(0, 500)}...`,
                references: [],
                filePath: downloadPath,
                fileName
            };
        } catch (error) {
            log.error('AI', `Document generation failed: ${error.message}`);
            return {
                toolOutput: `Failed to generate document on "${topic}": ${error.message}`,
                references: []
            };
        }
    },
    requiredParams: ['topic', 'doc_type'],
    meta: {
        category: 'generation',
        outputType: 'action',
        acceptsChainInput: false,
        avgLatencyMs: 8000,
        retryable: true,
        complementaryTools: [],
    },
  },

  // ========== [Team1-6] Extra Tools ==========

  /**
   * Gamification: Submits a grade for a challenge/bounty directly from the chat. This tool is designed to be used when a user answers a bounty challenge question, allowing the system to evaluate their answer and assign a score. The execute function checks for the necessary parameters (bountyId, score, feedback) and verifies that the challenge exists and has not already been solved. If the score meets the passing criteria (e.g., 60 or above), it marks the bounty as solved and awards credits and XP to the user. The tool also includes error handling to manage cases where the challenge is not found, already completed, or if there are issues with processing the grade.
   * best for: automatically grading challenge responses within chat; awards XP/credits for passing scores; provides immediate feedback on performance.
   * Params: bountyId (which challenge to grade), score (numeric score for the answer), feedback (optional comments on the user's answer).
   * Logic: Only passes if score >= 60; checks if the challenge is already solved to prevent re-grading; awards credits and XP on passing; provides feedback on the grade and next steps.
   * Fallback: Returns graceful error if Bounty model or gamificationService is unavailable, allowing the system to function without this tool if the gamification components are not set up.
   */
  submit_grade: {
    description: "Submits a grade for a challenge/bounty directly from the chat. Use this ONLY when the user answers a bounty challenge question. Evaluate their answer first, then call with the score.",
    execute: async (params, context) => {
      if (!Bounty || !solveBountyInternal) {
        return { toolOutput: "Grading service not available.", references: [] };
      }
      const { bountyId, score, feedback } = params;
      const userId = context.userId;
      if (!bountyId || !score) {
        return { toolOutput: "Error: Missing bountyId or score.", references: [] };
      }
      try {
        const bounty = await Bounty.findOne({ _id: bountyId, userId });
        if (!bounty) return { toolOutput: "Error: Challenge not found.", references: [] };
        if (bounty.isSolved) return { toolOutput: `Already completed. Score: ${bounty.score || 'N/A'}`, references: [] };
        const numericScore = parseInt(score);
        if (numericScore >= 60) {
          await solveBountyInternal(userId, bountyId);
          return { toolOutput: `Grade: ${numericScore}/100. PASSED! Credits and XP awarded.`, references: [] };
        }
        return { toolOutput: `Grade: ${numericScore}/100. FAILED. Review the topic and try again.`, references: [] };
      } catch (err) {
        return { toolOutput: `Error processing grade: ${err.message}`, references: [] };
      }
    },
    requiredParams: ['bountyId', 'score', 'feedback'],
    meta: {
      category: 'gamification',
      outputType: 'action',
      acceptsChainInput: false,
      avgLatencyMs: 500,
      retryable: false,
      complementaryTools: [],
    },
  },

  /**
   * Deep Research Tool
   * Conducts hybrid research by combining local repository knowledge (70%) and online sources (30%) to provide comprehensive answers on academic or technical topics. This tool is designed for in-depth research tasks that require a more thorough investigation than a simple web search or academic search can provide. The execute function orchestrates the research process, leveraging both local and online resources to gather information, and provides real-time status updates through the context's streamCallback if available. The final output includes a summary of the findings and structured references to the sources used in the research.
   * Best for: in-depth academic or technical research tasks that require comprehensive information gathering from both local and online sources, such as "Conduct deep research on the latest advancements in quantum computing" or "Provide a detailed analysis of the impacts of climate change using both scientific literature and recent news articles."
   * Features: Streams progress updates during the research process, allowing users to see the status of the research in real-time. Combines the strengths of local repository knowledge (which may include pre-indexed documents, internal databases, or cached information) with online sources to provide a well-rounded and thorough answer.
   * Output: Summary text + structured references array with title/source and URL for each source, enabling users to understand the basis of the research findings and access the original sources for further reading.
   * Complements: academic_search and web_search, providing a more comprehensive research tool that can leverage both the depth of academic sources and the breadth of web information when needed for complex research queries.
   */
  deep_research: {
    description: "Conducts comprehensive hybrid research combining local repository knowledge (70%) and online sources (30%). Use for in-depth academic or technical topics.",
    execute: async (params, context) => {
      const result = await conductDeepResearch(params.query, context, (status) => {
        if (context.streamCallback) {
          context.streamCallback({
            type: 'thought',
            content: `> 🔍 [Research Status] ${status}\n\n`,
            structured: { step: 'research_status', status }
          });
        }
      });
      return {
        toolOutput: result.summary,
        references: result.sources.map((s, i) => ({ number: i + 1, source: s.title || s.url, url: s.url }))
      };
    },
    requiredParams: ['query'],
    meta: {
      category: 'search',
      outputType: 'text',
      acceptsChainInput: true,
      avgLatencyMs: 15000,
      retryable: true,
      complementaryTools: ['academic_search', 'web_search'],
    },
  },

  /**
   * Autonomous Agent Tool
   * Breaks down a complex user goal into sub-tasks and executes them automatically using a DAG (directed acyclic graph). Use for multi-step problems requiring planning.
   * Delegates complex, multi-step goals to the Agent Orchestrator, which handles task decomposition, planning, and execution. This tool is ideal for situations where the user's request involves multiple steps or requires a strategic approach to achieve the desired outcome. The execute function checks for the availability of the agent orchestration service and routes the goal accordingly, providing a fallback message if the service is not available.
   * Best for: tasks requiring planning, sequencing, or parallel execution of multiple steps to achieve a complex goal, such as "Plan and execute a social media campaign for a new product launch" or "Organize a virtual conference on AI ethics, including speaker invitations, agenda planning, and promotional activities."
   * Param: goal (the complex user goal that needs to be achieved, which may require multiple steps or actions).
   * Output: The output will depend on the specific tasks executed by the agent orchestrator, but it should provide a summary of the actions taken and the results achieved in relation to the original goal. This may include updates on each sub-task, final outcomes, and any relevant references or resources used during execution.
   * Fallback: Returns graceful error if agentOrchestrator service is unavailable, allowing the system to function without this tool if the orchestration components are not set up.
   * Note: High latency(~30s) - use only for genuinely complex, multi-step tasks where the benefits of orchestration outweigh the wait time. Not suitable for simple queries or tasks that can be handled by other tools with lower latency.
   */
  autonomous_agent: {
    description: "Breaks down a complex user goal into sub-tasks and executes them automatically using a DAG (directed acyclic graph). Use for multi-step problems requiring planning.",
    execute: async (params, context) => {
      if (!executeAgentTask) {
        return { toolOutput: "Autonomous agent service not available.", references: [] };
      }
      return await executeAgentTask(params.goal, context);
    },
    requiredParams: ['goal'],
    meta: {
      category: 'orchestration',
      outputType: 'text',
      acceptsChainInput: true,
      avgLatencyMs: 30000,
      retryable: false,
      complementaryTools: [],
    },
  },
};

/**
 * Get metadata for a specific tool.
 * 
 * @param {string} toolName
 * @returns {Object|null} Tool metadata
 */
function getToolMeta(toolName) {
    const tool = availableTools[toolName];
    if (!tool) return null;
    return {
        name: toolName,
        description: tool.description,
        requiredParams: tool.requiredParams,
        ...(tool.meta || {}),
    };
}

/**
 * Get all tool names and descriptions (for LLM prompts).
 * @returns {Array} Array of { name, description }
 */
function getToolSummaries() {
    return Object.entries(availableTools).map(([name, tool]) => ({
        name,
        description: tool.description,
        category: tool.meta?.category || 'general',
    }));
}

module.exports = { availableTools, getToolMeta, getToolSummaries };