// server/services/semanticRouter.js
/**
 * ROLE: Full intent classification layer — classifies query intent into named
 * categories (DEEP_RESEARCH, ACADEMIC_SEARCH, SOCRATIC_TUTORING, etc.),
 * selects tools and LLM, and handles rejections (inappropriate queries).
 * Caches example embeddings to disk on first run.
 *
 * RELATIONSHIP TO semanticRouterService.js:
 * These are two SEPARATE routing layers, not duplicates:
 *  - semanticRouter.js         → WHAT kind of query is this? (intent classification)
 *  - semanticRouterService.js  → WHERE should this query go? (route selection via prototypes + Redis)
 * Both are used by the main chat pipeline at different decision points.
 */
// server/services/semanticRouter.js
/**
 * Semantic Router - Intelligent Query Classification & Tool Selection
 *
 * Uses vector embeddings to match queries against predefined intent categories.
 * Embeddings are cached to disk on first run and reloaded on restart (fast path).
 *
 * Primary embedding: FastEmbed (ONNX, mxbai-embed-large-v1, 1024-dim) via Python RAG /embed
 * No direct Ollama or Gemini embedding dependency.
 */

const fs   = require('fs');
const path = require('path');
const axios = require('axios');
const log  = require('../utils/logger');

const CACHE_FILE = path.join(__dirname, '../data/semantic_router_cache.json');
const RAG_EMBED_URL = () => `${(process.env.PYTHON_RAG_SERVICE_URL || 'http://localhost:2001').trim()}/embed`;

// ─────────────────────────────────────────────────────────────────────────────
// INTENT CATEGORIES WITH SEMANTIC EXAMPLES
// ─────────────────────────────────────────────────────────────────────────────

const INTENT_ROUTES = {
  DEEP_RESEARCH: {
    examples: [
      // Only clear "write a full research paper / systematic review" intents
      "write a research paper on blockchain technology",
      "help me write a research paper on AI ethics",
      "write a comprehensive academic literature review on CRISPR",
      "conduct a systematic review of federated learning",
      "write a full research report on renewable energy policy",
      "generate a detailed academic report on supply chain disruptions",
      "produce a comprehensive research synthesis on protein folding",
      "create an academic research paper on quantum computing applications",
      "write a literature review on transformer architectures for my thesis",
      "compile a full research paper on climate change mitigation strategies",
      "I need a complete research document on the history of neural networks",
      "generate a scholarly report on multi-agent reinforcement learning",
    ],
    tools: ['deep_research'],
    handler: 'deepResearch',
    confidence_threshold: 0.70,  // raised from 0.60 — only explicit "write paper/report" requests score ≥0.70
  },

  ACADEMIC_SEARCH: {
    examples: [
      "find papers on attention mechanisms",
      "research articles about transformers",
      "scholarly articles on quantum entanglement",
      "peer-reviewed papers about vaccination",
      "arxiv papers on computer vision",
      "locate papers on dark matter theories",
      // Additional diverse examples
      "peer-reviewed research on mRNA vaccine safety",
      "scholarly articles on dark matter detection methods",
      "find academic research on reinforcement learning from human feedback",
      "journal articles about climate change impacts",
      "scientific literature on quantum computing algorithms",
    ],
    tools: ['academic_search'],
    handler: 'academicSearch',
    confidence_threshold: 0.60,  // was 0.70; measured scores land 0.62-0.70
  },

  WEB_SEARCH: {
    examples: [
      // Technology & Science current events
      "latest news on AI developments",
      "current trends in semiconductor industry",
      "recent breakthroughs in fusion energy",
      "newest quantum computer announcements",
      "latest developments in renewable energy",
      // World events & geopolitics
      "recent news about Israel and Iran conflict",
      "what is happening in Ukraine right now",
      "latest updates on US-China relations",
      "current situation in the Middle East",
      "recent news about Iran nuclear program",
      "what happened in the latest election",
      "recent conflict news around the world",
      // General current events
      "what are today's top news stories",
      "latest updates on the climate summit",
      "recent economic news and market updates",
    ],
    tools: ['web_search'],
    handler: 'webSearch',
    confidence_threshold: 0.52,  // was 0.60; measured scores land 0.55-0.71
  },

  TECHNICAL_CODING: {
    examples: [
      "write python code to sort array",
      "implement binary search algorithm",
      "create REST API with Express",
      "debug this JavaScript function",
      "optimize SQL query performance",
      "implement quicksort in Java",
      "create React component for login",
      "implement graph traversal algorithm",
      // Additional diverse examples
      "Write a Python function to reverse a linked list",
      "How do I implement OAuth2 in Express.js?",
      "Debug this React useEffect infinite loop",
      "What is the time complexity of merge sort?",
      "Build a REST API with JWT authentication in FastAPI",
      "Explain and implement a red-black tree in Python",
      "Write a C++ class for a thread-safe queue",
      "Fix a segmentation fault in C pointer arithmetic",
    ],
    tools: ['rag_retrieve'],
    handler: 'standardWithRAG',
    llm_preference: 'code',
    confidence_threshold: 0.57,  // was 0.75; measured scores land 0.60-0.75
  },

  MATHEMATICAL_REASONING: {
    examples: [
      "solve quadratic equation x^2 + 5x + 6 = 0",
      "calculate derivative of sin(x^2)",
      "prove Pythagorean theorem",
      "integrate x^2 * e^x dx",
      "solve system of linear equations",
      "find eigenvalues of matrix",
      "compute Fourier transform of signal",
      // Additional examples
      "Solve the integral of x^2 * sin(x) dx",
      "Find eigenvalues of the matrix [[2,1],[1,3]]",
      "Prove that sqrt(2) is irrational",
      "Compute the Fourier transform of a rectangular pulse",
      "What is the Laplace transform of t*e^(2t)?",
      "Prove by induction that n^3 - n is divisible by 6",
    ],
    tools: [],
    handler: 'standardWithRAG',
    llm_preference: 'reasoning',
    confidence_threshold: 0.68,  // was 0.80; measured scores land 0.71-0.78
  },

  CONCEPTUAL_EXPLANATION: {
    examples: [
      "explain how photosynthesis works",
      "what is quantum entanglement",
      "describe the krebs cycle",
      "how does TCP/IP protocol work",
      "explain gradient descent algorithm",
      "what causes earthquakes",
      "how do vaccines provide immunity",
      "what is general relativity",
    ],
    tools: ['rag_retrieve'],
    handler: 'standardWithRAG',
    confidence_threshold: 0.70,
  },

  SOCRATIC_TUTORING: {
    examples: [
      "teach me calculus step by step",
      "help me understand thermodynamics",
      "guide me through organic chemistry",
      "tutor me on data structures",
      "I need help learning physics",
      "teach quantum mechanics gradually",
      "guide me through circuit analysis",
      "help me understand statistical inference",
      // Additional diverse examples
      "Teach me calculus from the basics step by step",
      "Help me understand quantum mechanics gradually",
      "Guide me through learning data structures",
      "Tutor me on organic chemistry reactions",
      "Guide me through how the TCP handshake works",
      "Walk me through how a compiler works step by step",
    ],
    tools: ['rag_retrieve'],
    handler: 'socraticTutor',
    confidence_threshold: 0.72,  // was 0.75; measured scores land 0.749
  },

  DOCUMENT_RAG: {
    examples: [
      "summarize the uploaded document",
      "what does page 5 say about",
      "find information in my notes about",
      "search through the PDF for",
      "extract key points from the paper",
      "find references to machine learning in files",
      "what does the syllabus say about grading",
    ],
    tools: ['rag_retrieve'],
    handler: 'documentRAG',
    confidence_threshold: 0.70,
  },

  ENTERTAINMENT: {
    examples: [
      "recommend a good movie to watch",
      "who won the cricket match yesterday",
      "tell me about latest Netflix series",
      "recommend songs to listen to",
      "tell me about Bollywood gossip",
      "who is the best NBA player",
      "recommend video games to play",
      "Who won the cricket match yesterday?",
      "best movie streaming right now",
    ],
    rejection: true,
    rejection_message: "I'm an academic learning assistant. I cannot help with entertainment, sports, or media recommendations.",
    confidence_threshold: 0.56,  // was 0.65; measured scores land 0.587-0.72
  },

  LIFESTYLE_PERSONAL: {
    examples: [
      "give me a recipe for chocolate cake",
      "how to lose weight quickly",
      "best workout routine for beginners",
      "fashion tips for summer",
      "relationship advice for couples",
      "best skincare products",
    ],
    rejection: true,
    rejection_message: "I'm focused on academic and technical topics. I cannot provide lifestyle, fashion, or personal advice.",
    confidence_threshold: 0.60,  // was 0.65; existing examples score 0.654
  },

  INAPPROPRIATE: {
    examples: [
      "tell me a dirty joke",
      "how to hack someone's account",
      "create fake news article",
      "help me cheat on exam",
      "create malicious software",
      "write plagiarized essay",
      "How do I hack into someone's email account?",
      "help me break into a system",
      "write malware to steal passwords",
    ],
    rejection: true,
    rejection_message: "I cannot assist with inappropriate, unethical, or harmful requests.",
    confidence_threshold: 0.57,  // was 0.70; measured scores land 0.60
  },

  MEMORY_RECALL: {
    examples: [
      "what did we discuss last time",
      "remind me what I asked before",
      "continue our previous conversation",
      "recall what I told you about my project",
      "pick up where we left off",
      "what was my previous question",
      "remind me what I asked about neural networks before",
      "what did I ask you about machine learning earlier",
    ],
    tools: ['conversation_memory'],
    handler: 'memoryRecall',
    confidence_threshold: 0.75,
  },

  GREETING: {
    examples: [
      "hello",
      "hi there",
      "hey how are you",
      "good morning",
      "greetings",
      "hi AI",
      "hello assistant",
    ],
    handler: 'greeting',
    confidence_threshold: 0.70,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// EMBEDDING — FastEmbed via Python RAG service /embed endpoint
// ─────────────────────────────────────────────────────────────────────────────

async function getEmbedding(text) {
  try {
    const resp = await axios.post(RAG_EMBED_URL(), { text }, { timeout: 10000 });
    const emb = resp.data?.embedding;
    if (Array.isArray(emb) && emb.length > 0) return emb;
    throw new Error('Empty embedding returned');
  } catch (err) {
    log.error('SEMANTIC_ROUTER', `FastEmbed /embed failed: ${err.message}`);
    throw new Error('Embedding service unavailable');
  }
}

async function getBatchEmbeddings(texts) {
  try {
    const resp = await axios.post(RAG_EMBED_URL(), { texts }, { timeout: 60000 });
    const embs = resp.data?.embeddings;
    if (Array.isArray(embs) && embs.length === texts.length) return embs;
    throw new Error('Unexpected batch embed response shape');
  } catch (err) {
    log.warn('SEMANTIC_ROUTER', `Batch embed failed, falling back to serial: ${err.message}`);
  }

  // Serial fallback
  const results = [];
  for (const text of texts) {
    results.push(await getEmbedding(text));
  }
  return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// COSINE SIMILARITY
// ─────────────────────────────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ─────────────────────────────────────────────────────────────────────────────
// CACHE — persist embeddings to disk to avoid re-embedding on every restart
// ─────────────────────────────────────────────────────────────────────────────

let _cache = null;       // { [intentName]: { config, embeddings: [{text, vector}] } }
let _ready = false;
let _initPromise = null;

function _loadCacheFromDisk() {
  try {
    if (fs.existsSync(CACHE_FILE)) {
      const raw = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
      // Validate: each intent must have all examples embedded
      for (const [name, data] of Object.entries(raw)) {
        const expected = INTENT_ROUTES[name]?.examples?.length || 0;
        if (!data.embeddings || data.embeddings.length !== expected) return null;
      }
      return raw;
    }
  } catch (e) {
    log.warn('SEMANTIC_ROUTER', `Cache load failed: ${e.message}`);
  }
  return null;
}

function _saveCacheToDisk(cache) {
  try {
    const dir = path.dirname(CACHE_FILE);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(CACHE_FILE, JSON.stringify(cache, null, 2), 'utf8');
  } catch (e) {
    log.warn('SEMANTIC_ROUTER', `Cache save failed: ${e.message}`);
  }
}

async function initialize() {
  if (_ready) return;

  log.info('SEMANTIC_ROUTER', 'Initializing semantic router...');

  // Try loading from disk cache first
  const diskCache = _loadCacheFromDisk();
  if (diskCache) {
    _cache = diskCache;
    _ready = true;
    log.info('SEMANTIC_ROUTER', `Loaded ${Object.keys(_cache).length} intents from disk cache (fast path)`);
    return;
  }

  // Build cache: batch-embed all examples per intent
  log.info('SEMANTIC_ROUTER', 'No valid cache found — embedding all examples (one-time setup)...');
  _cache = {};

  for (const [intentName, intentConfig] of Object.entries(INTENT_ROUTES)) {
    if (!intentConfig.examples || intentConfig.examples.length === 0) continue;
    try {
      const vectors = await getBatchEmbeddings(intentConfig.examples);
      _cache[intentName] = {
        config: intentConfig,
        embeddings: intentConfig.examples.map((text, i) => ({ text, vector: vectors[i] })),
      };
      log.info('SEMANTIC_ROUTER', `  Embedded ${intentConfig.examples.length} examples for ${intentName}`);
    } catch (err) {
      log.error('SEMANTIC_ROUTER', `Failed to embed intent ${intentName}: ${err.message}`);
      _cache[intentName] = { config: intentConfig, embeddings: [] };
    }
  }

  _saveCacheToDisk(_cache);
  _ready = true;
  log.info('SEMANTIC_ROUTER', `Semantic router ready. Cache saved to disk.`);
}

// ─────────────────────────────────────────────────────────────────────────────
// CLASSIFICATION
// ─────────────────────────────────────────────────────────────────────────────

async function classifyIntent(query, context = {}) {
  if (!_ready) await initialize();

  // Context-awareness: use prior signals (uploaded files, user-enabled tools)
  // to bias scoring before cosine similarity decides.
  const { hasUploadedFiles = false, activeTools = {} } = context;

  let queryEmbedding;
  try {
    queryEmbedding = await getEmbedding(query);
  } catch (err) {
    log.warn('SEMANTIC_ROUTER', `Cannot embed query — defaulting to CONCEPTUAL_EXPLANATION: ${err.message}`);
    return {
      intent: 'CONCEPTUAL_EXPLANATION',
      confidence: 0.50,
      config: INTENT_ROUTES.CONCEPTUAL_EXPLANATION,
      fallback: true,
    };
  }

  let bestMatch = null;
  let bestScore = 0;

  for (const [intentName, cached] of Object.entries(_cache)) {
    if (!cached.embeddings || cached.embeddings.length === 0) continue;

    // Max similarity against all examples (top-3 mean for stability)
    const sims = cached.embeddings
      .map(ex => cosineSimilarity(queryEmbedding, ex.vector))
      .sort((a, b) => b - a);
    const topN = sims.slice(0, 3);
    let score = topN.reduce((s, v) => s + v, 0) / topN.length;

    // ── Context-aware score adjustments ────────────────────────────────────
    // Uploaded file → strong prior that query is about that document
    if (intentName === 'DOCUMENT_RAG' && hasUploadedFiles) {
      score = Math.min(score * 1.3, 1.0);
    }
    // User-enabled tool flags → the query is likely in that domain; nudge the
    // confidence up so the semantic handler also matches (avoids silent mismatch
    // between user intent and handler selection).
    if (intentName === 'WEB_SEARCH' && activeTools.webSearch) {
      score = Math.min(score * 1.15, 1.0);
    }
    if (intentName === 'ACADEMIC_SEARCH' && activeTools.academicSearch) {
      score = Math.min(score * 1.15, 1.0);
    }
    if (intentName === 'DEEP_RESEARCH' && activeTools.deepResearch) {
      score = Math.min(score * 1.15, 1.0);
    }

    if (score > bestScore && score >= cached.config.confidence_threshold) {
      bestScore = score;
      bestMatch = { intent: intentName, confidence: score, config: cached.config };
    }
  }

  if (!bestMatch || bestScore < 0.55) {
    log.info('SEMANTIC_ROUTER', `No confident match for "${query.substring(0, 50)}" — defaulting`);
    return {
      intent: 'CONCEPTUAL_EXPLANATION',
      confidence: 0.50,
      config: INTENT_ROUTES.CONCEPTUAL_EXPLANATION,
      fallback: true,
    };
  }

  log.info('SEMANTIC_ROUTER', `"${query.substring(0, 50)}" → ${bestMatch.intent} (conf=${bestScore.toFixed(3)})`);
  return bestMatch;
}

// ─────────────────────────────────────────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────────────────────────────────────────

async function routeQuery(query, context = {}) {
  const classification = await classifyIntent(query, context);

  if (classification.config.rejection) {
    return {
      shouldReject: true,
      rejectionMessage: classification.config.rejection_message,
      intent: classification.intent,
      confidence: classification.confidence,
    };
  }

  return {
    shouldReject: false,
    intent: classification.intent,
    confidence: classification.confidence,
    handler: classification.config.handler,
    tools: classification.config.tools || [],
    llmPreference: classification.config.llm_preference || 'general',
    fallback: classification.fallback || false,
  };
}

function selectLLMForIntent(intent, userPreferredProvider = 'sglang') {
  const intentConfig = INTENT_ROUTES[intent];
  if (!intentConfig) return { provider: userPreferredProvider, model: 'default' };

  const llmMapping = {
    code:      { sglang: 'qwen2.5-7b-instruct', ollama: 'qwen2.5:7b', gemini: 'gemini-2.0-flash' },
    reasoning: { sglang: 'qwen2.5-7b-instruct', ollama: 'qwen2.5:7b', gemini: 'gemini-2.0-flash' },
    general:   { sglang: 'qwen2.5-7b-instruct', ollama: 'qwen2.5:3b', gemini: 'gemini-2.0-flash' },
  };

  const preference = intentConfig.llm_preference || 'general';
  const models = llmMapping[preference] || llmMapping.general;

  return {
    provider: userPreferredProvider,
    model: models[userPreferredProvider] || models.sglang,
    reasoning: preference,
  };
}

async function shouldUseTools(query, context = {}) {
  const routing = await routeQuery(query, context);
  return {
    useWebSearch: routing.tools.includes('web_search'),
    useAcademicSearch: routing.tools.includes('academic_search'),
    useRAG: routing.tools.includes('rag_retrieve'),
    useDeepResearch: routing.tools.includes('deep_research'),
    useMemory: routing.tools.includes('conversation_memory'),
    intent: routing.intent,
    confidence: routing.confidence,
  };
}

// Start initialization in background at module load (non-blocking)
_initPromise = initialize().catch(err =>
  log.warn('SEMANTIC_ROUTER', `Background init failed: ${err.message}`)
);

module.exports = {
  initialize,
  classifyIntent,
  routeQuery,
  selectLLMForIntent,
  INTENT_ROUTES,
};

