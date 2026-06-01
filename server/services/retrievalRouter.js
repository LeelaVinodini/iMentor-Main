const { getCachedRoute, setCachedRoute } = require('./routingCache');
const axios = require('axios');
const geminiService = require('./geminiService');

const RAG_SERVICE_URL = process.env.PYTHON_RAG_SERVICE_URL || 'http://localhost:5002';

/**
 * Route Retrieval
 * Decides whether to use Vector Store, Knowledge Graph, or Web Search.
 */
async function routeRetrieval(query, context) {
    console.log(`[RetrievalRouter] Routing query: "${query}"`);

    // Task 1.2.3: Routing Cache
    const cached = await getCachedRoute(query);
    if (cached) {
        console.log(`[RetrievalRouter] Cache Hit: ${cached}`);
        return await executeStrategy(cached, query, context);
    }

    // 1. Classify Query (Task 1.2.1: Semantic ML Routing)
    const { classifyIntent } = require('./semanticIntentService');
    const decision = classifyIntent(query);

    console.log(`[RetrievalRouter] Strategy selected via Semantic ML: ${decision}`);

    

    // 2. Cache and Execute
    await setCachedRoute(query, decision);
    return await executeStrategy(decision, query, context);
}

/**
 * Helper to execute the retrieval strategy
 */
async function executeStrategy(decision, query, context) {
    const _docCtx = context.documentContextName || null;
    const _hasExt = _docCtx ? /\.[a-zA-Z0-9]{2,5}$/.test(_docCtx) : false;
    const _sourceType = _hasExt ? 'user_doc' : (_docCtx ? 'course' : null);

    let results = { decision };
    try {
        if (decision === 'WEB') {
            const res = await axios.post(`${RAG_SERVICE_URL}/web_search`, { query });
            results.web = res.data;
        } else if (decision === 'GRAPH') {
            const res = await axios.post(`${RAG_SERVICE_URL}/graph_rag/search`, {
                query,
                user_id: context?.user?.id,
                documentContextName: _docCtx
            });
            results.graph = res.data.facts;
        } else if (decision === 'VECTOR') {
            const res = await axios.post(`${RAG_SERVICE_URL}/query`, {
                query,
                user_id: context?.user?.id,
                k: 3,
                documentContextName: _docCtx,
                source_type: _sourceType
            });
            results.vector = res.data.retrieved_documents_list;
        } else { // HYBRID
            const [graphRes, vectorRes] = await Promise.all([
                axios.post(`${RAG_SERVICE_URL}/graph_rag/search`, {
                    query,
                    user_id: context?.user?.id,
                    documentContextName: _docCtx
                }).catch(e => ({ data: { facts: [] } })),
                axios.post(`${RAG_SERVICE_URL}/query`, {
                    query,
                    user_id: context?.user?.id,
                    k: 3,
                    documentContextName: _docCtx,
                    source_type: _sourceType
                }).catch(e => ({ data: { retrieved_documents_list: [] } }))
            ]);
            results.graph = graphRes.data.facts;
            results.vector = vectorRes.data.retrieved_documents_list;
        }
    } catch (error) {
        console.error(`[RetrievalRouter] Execution failed: ${error.message}`);
        throw new Error(`Retrieval failed: ${error.message}`);
    }
    return results;
}

module.exports = { routeRetrieval };