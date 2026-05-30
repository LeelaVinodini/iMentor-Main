const { decomposeTask } = require('./taskDecompositionService');
const { routeRetrieval } = require('./retrievalRouter');
// const { availableTools } = require('./toolRegistry'); // Removed to fix circular dependency
// Note: We might need to refactor toolRegistry to export individual functions if we want direct calls, 
// or use the toolRegistry.execute() method.

/**
 * Agent Orchestrator - Main Execution Entry Point
 *
 * This module orchestrates the entire process of handling a user request by:
 * 1. Breaking it down into a dependency-aware plan (via decomposeTask).
 * 2. Running plan steps in parallel 'waves' - a step executes only when all its dependencies are completed.
 * Routing retrieval tasks through the retrieval router to ensure proper tool usage and context management.
 * Collecting results and execution traces for final response construction.
 * This implements a simplified DAG executor: no topological sort upfront, but rather a dynamic check for ready steps in each iteration, allowing for parallel execution where possible.
 *
 * The design allows for flexible integration of various tools and retrieval methods, while maintaining a clear separation of concerns between task decomposition, retrieval routing, and execution orchestration.
 * @param {string} userRequest - The original user request to be processed.
 * @param {object} context - Additional context for the request, such as user info, session data, etc.
 * @returns {object} An object containing the decomposition plan and execution trace results.   
 * @returns {promise<Object>} An object containing the decomposition plan and execution trace results.
 *          -plan: The structured plan generated from the user request, detailing steps and dependencies.
 *          -execution_trace: An array of [stepId, result] pairs representing the outcome of each executed step.
 *
 * Note: This is a simplified implementation for demonstration purposes. In a production system, you would want to add error handling, logging, and more robust management of execution state and results.
 * Manages the execution of the decomposition plan.
 */
async function executeAgentTask(userRequest, context) {
    // 1. Decompose the high level request into a plan with dependencies

/*
decomposeTask returns a plan object like:
{
steps: [
    { id: 'step1', description: 'Search for X', tool: 'web_search', dependencies: [] },
    { id: 'step2', description: 'Analyze results of step1', tool: 'analysis_tool', dependencies: ['step1'] },
    { id: 'step3', description: 'Generate report', tool: 'reporting_tool', dependencies: ['step2'] }
]
}
*/
    const plan = await decomposeTask(userRequest, context);

    // 2. Execute Graph (Topological Sort / Parallel Execution)
    // For simplicity, we'll do linear or simple dependency check loop.

    const results = new Map(); // stepId -> result
    const completed = new Set();
/*
Wave Execution loop:
Each iteration, we find all steps that are not completed and whose dependencies are all in the completed set. We execute those steps in parallel (using Promise.allSettled) and then mark them as completed once done. This allows for concurrent execution of independent steps while respecting the dependency order.
Executes all ready steps concurrently, collects results, and marks them as completed before moving to the next wave of ready steps. This approach is efficient for plans with multiple independent branches, allowing for maximum parallelism while ensuring correct execution order based on dependencies.
Marks completed steps and repeats until the plan is done or stuck.
This avoids blocking on sequential execution and allows for efficient handling of complex plans with multiple dependencies, while also providing a mechanism to handle errors gracefully without crashing the entire execution flow.
*/
    // Wave-based parallel DAG: each wave runs all steps whose dependencies are met concurrently
    while (completed.size < plan.steps.length) {
        const ready = plan.steps.filter(
            s => !completed.has(s.id) && s.dependencies.every(d => completed.has(d))
        );
        if (ready.length === 0) break; // guard against cycles or unresolvable deps
/**
 * Executes all ready steps concurrently, collects results, and marks them as completed before moving to the next wave of ready steps. This approach is efficient for plans with multiple independent branches, allowing for maximum parallelism while ensuring correct execution order based on dependencies.
 * Marks completed steps and repeats until the plan is done or stuck.
 * This avoids blocking on sequential execution and allows for efficient handling of complex plans with multiple dependencies, while also providing a mechanism to handle errors gracefully without crashing the entire execution flow. 
 * Logs info about each step execution and catches errors to prevent a single failure from halting the entire process, while still recording the error in the results for visibility.
 * Routes retrieval tools
 * Catches and reports errors for each step to ensure that one failure does not stop the entire execution, while still providing visibility into what went wrong for that specific step.
 */
        const wave = await Promise.allSettled(
            ready.map(step =>
                (async () => {
                    console.log(`[Agent] Executing Step ${step.id}: ${step.description}`);
                    let result = '';
                    //Route retrieval-type steps to the appropriate search backend via the retrieval router, which handles context and tool selection. For non-retrieval steps, we simulate a result for demonstration purposes, but in a real implementation, this is where you would call the actual tool execution logic.
                    if (['web_search', 'graph_search', 'vector_search'].includes(step.tool)) {
                        const retrieval = await routeRetrieval(step.description, context);
                        result = JSON.stringify(retrieval);
                    } else {
                            // Route non-retrieval steps through toolRegistry using lazy require().
                            // Lazy require (inside the function, not at the top of the file) avoids
                             // the circular dependency — by the time this line runs, both modules are
                            // fully loaded and the cycle is no longer an issue.
                            const { availableTools } = require('./toolRegistry');
                            const tool = availableTools[step.tool];
                            if (tool) {
                            const toolResult = await tool.execute(step.params || { query: step.description }, context);
                            result = typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult);
                            } else {
                                // Tool name from decomposition didn't match any registered tool.
                                // Log it clearly so it's easy to debug which tool name is wrong.
                                console.warn(`[Agent] No tool found for "${step.tool}" — step ${step.id} skipped.`);
                                result = `No tool registered for "${step.tool}". Known tools: ${Object.keys(availableTools).join(', ')}`;
                            }
                        }
                    return { id: step.id, result };
                })().catch(e => {
                    //Graceful error handling: log the error and return an error result for this step, allowing the rest of the execution to continue unaffected. This ensures that a failure in one step does not halt the entire process, while still providing visibility into what went wrong for that specific step.
                    console.error(`[Agent] Step ${step.id} failed: ${e.message}`);
                    return { id: step.id, result: `Error: ${e.message}` };
                })
            )
        );
// Collect results from this wave and mark steps as completed. This allows the next wave of dependent steps to proceed in the next iteration of the loop.
        for (const outcome of wave) {
            //Promise.allSettled return { status: 'fulfilled', value: ... } or { status: 'rejected', reason: ... }
            const { id, result } = outcome.value;
            results.set(id, result);
            completed.add(id);
        }
        //Rejected promises are already handled in the catch block of the individual step execution, so we don't need to do anything special here for rejected outcomes. The error is logged and the result is set to an error message, allowing the execution to continue without interruption.
    }
//3. Return structured output for downstream synthesis/response construction
    // - plan: The original decomposition (useful for debugging and step-awareness in response synthesis)
    // - execution_trace: List of [stepId, result] for answer synthesis and user-facing explanations
    return {
        plan: plan,
        execution_trace: Array.from(results.entries())
    };
}
module.exports = { executeAgentTask };
