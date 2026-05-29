const log = require('../utils/logger');
const { redisClient } = require('../config/redisClient');
const geminiService = require('./geminiService');
const ollamaService = require('./ollamaService');
const groqService = require('./groqService');
const claudeService = require('./claudeService');
const openaiService = require('./openaiService');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const {
    SOCRATIC_CLASSIFICATION_PROMPT,
    SOCRATIC_QUESTION_GENERATION_PROMPT,
    SOCRATIC_INTRO_PROMPT
} = require('../config/promptTemplates');
const reactOrchestrator = require('./reactOrchestrator');
const llmStreamingService = require('./llmStreamingService');
const { decideTeachingAction } = require('./teachingPolicyService');
const { reflectOnTeaching } = require('./teachingReflectionService');
const { safe, sanitizeGeneratedText } = require('../utils/promptSanitizer');

const ENABLE_TUTOR_REACT_WRAPPER = process.env.ENABLE_TUTOR_REACT_WRAPPER === 'true';

const TUTOR_STATE_TTL = 3600; // 1 hour
const MASTERY_THRESHOLD = 4.0; // Cumulative score needed for mastery

const COGNITIVE_LEVELS = {
    L1_CONCEPT: 'L1_CONCEPT',         // Definition, basic understanding
    L2_APPLICATION: 'L2_APPLICATION', // Real-world examples, practical use
    L3_CRITICAL: 'L3_CRITICAL',       // Edge cases, limitations, bias
    L4_EVALUATION: 'L4_EVALUATION'    // Comparison, improvement, design
};

const PEDAGOGICAL_MOVES = {
    STAY: 'STAY',           // Stay at current level (refine/correct)
    ADVANCE_LEVEL: 'ADVANCE_LEVEL', // Move up the ladder (L1 -> L2, etc.)
    JUMP_LEVEL: 'JUMP_LEVEL',       // Skip a level (L1 -> L3) for advanced students
    COMPLETE: 'COMPLETE'            // Subtopic mastery achieved
};

/**
 * SOCRATIC_STATES: The cognitive progression ladder for guided learning.
 * Each state represents a phase in the student's understanding journey, from initial introduction to mastery. The tutor uses this framework to adapt its questioning and support strategies based on the student's current state, aiming to guide them through increasingly complex levels of understanding while providing appropriate scaffolding and challenges at each step.
 * 
 * Introduction: 
 * Goal: Orient the student, build rapport, and set expectations.
 * Behavior: Brief topic overview + simple opening question to gauge prior knowledge.
 * Exit: Student engages -> advance to L1_Concept
 *
 * L1_CONCEPT:(Remembering)
 * Goal: Establish foundational understanding of the core concept.
 * Behavior: Ask recall questions ("What is X?"), confirm key terms are understood.
 *   Exit: Student demonstrates basic understanding → advance to L2_APPLICATION.
 *
 * L2_APPLICATION (Understanding/Applying):
 *   Goal: Connect concept to real-world scenarios and practical use.
 *   Behavior: Ask "How would you use X in situation Y?" or "Give an example of X".
 *   Exit: Student applies concept correctly → advance to L3_CRITICAL.
 *
 * L3_CRITICAL (Analyzing/Evaluating):
 *   Goal: Develop critical thinking about limitations, edge cases, and biases.
 *   Behavior: Ask "What are the weaknesses of X?" or "When might X fail?".
 *   Exit: Student demonstrates nuanced understanding → advance to L4_EVALUATION.
 *
 * L4_EVALUATION (Creating/Designing):
 *   Goal: Enable synthesis, comparison, and design thinking.
 *   Behavior: Ask "How would you improve X?" or "Compare X vs Y for use case Z".
 *   Exit: Student demonstrates mastery → advance to MASTERY_ACHIEVED.
 *
 * MASTERY_ACHIEVED:
 *   Goal: Celebrate progress, consolidate learning, and transition to next topic.
 *   Behavior: Summary of key takeaways + optional reflection prompt.
 *   Exit: Session advances to next subtopic or completes course.
 */
const SOCRATIC_STATES = {
    INTRODUCTION: 'INTRODUCTION',
    ...COGNITIVE_LEVELS,
    MASTERY_ACHIEVED: 'MASTERY_ACHIEVED'
};

// ─── Emotional State Detection (ported from Team1-6) ────────────────────────
 
const EMOTIONAL_STATES = {
    CURIOUS: 'CURIOUS',
    CONFIDENT: 'CONFIDENT',
    UNCERTAIN: 'UNCERTAIN',
    FRUSTRATED: 'FRUSTRATED',
    BORED: 'BORED'
};

const CONFIDENCE_LEVELS = {
    HIGH: 'HIGH',
    MEDIUM: 'MEDIUM',
    LOW: 'LOW'
};

// Richer understanding classification (superset of T1-6 + T3)
const UNDERSTANDING_LEVELS = {
    CORRECT: 'CORRECT',
    PARTIAL: 'PARTIAL',
    MISCONCEPTION: 'MISCONCEPTION',
    VAGUE: 'VAGUE',
    NO_FOUNDATION: 'NO_FOUNDATION'
};

// Adaptive support levels (from tutorStates.js Bloom's taxonomy)
const SUPPORT_LEVELS = {
    MINIMAL: 'MINIMAL',       // Pure Socratic questioning — student is doing well
    GUIDED: 'GUIDED',         // Question + hint
    SCAFFOLDED: 'SCAFFOLDED', // Example + explanation
    DIRECT: 'DIRECT'          // Direct reteaching
};

/**
 * Multi-provider LLM call with automatic fallback (ported from Team1-6).
 * Tries preferred provider first, then falls back through all available providers.
 * Logs each attempt and returns the first successful response, or throws an error if all fail. 
 *
 * Purpose: Ensures tutor reliability by trying multiple LLM providers in sequence.
 * If the preferred provider fails (timeout, auth error, rate limit), automatically
 * falls back to the next available provider until one succeeds or all are exhausted.
 *
 * Flow:
 * 1. Start with preferred provider from llmConfig (default: 'gemini')
 * 2. For each provider: check env/config → call generateContentWithHistory → validate response
 * 3. Return first successful non-empty response
 * 4. If all fail, throw aggregated error with last failure details
 *
 * Providers tried (in order): sglang → gemini → groq → claude → openai → ollama
 * Note: sglang and ollama don't require API keys; others skip if key missing.
 *
 * @param {Array} chatHistory - Prior conversation turns (for context-aware generation)
 * @param {string} currentQuery - The immediate prompt/question to answer
 * @param {string} systemPrompt - System instructions to guide LLM behavior
 * @param {Object} llmConfig - Provider preferences, API keys, model names
 * @param {Object} additionalOptions - Extra params like jsonMode, temperature, etc.
 * @returns {Promise<string>} The LLM's generated response text (trimmed)
 * @throws {Error} If all configured providers fail to return a valid response 

 */
async function generateWithFallback(chatHistory, currentQuery, systemPrompt, llmConfig, additionalOptions = {}) {
    const preferredProvider = llmConfig?.llmProvider || 'gemini';
    const allProviders = ['sglang', 'gemini', 'groq', 'claude', 'openai', 'ollama'];
    const providers = [
        preferredProvider,
        ...allProviders.filter(p => p !== preferredProvider)
    ];

    let lastError = null;

    for (const provider of providers) {
        try {
            log.info('TUTOR', `🔄 Attempting ${provider}...`);

            let llmService;
            let options = { ...additionalOptions };

            switch (provider) {
                case 'sglang': {
                    const SGLANG_ENABLED = process.env.SGLANG_ENABLED === 'true';
                    if (!SGLANG_ENABLED) {
                        log.info('TUTOR', `Skipping sglang — SGLANG_ENABLED=false.`);
                        continue;
                    }
                    llmService = require('./sglangService');
                    options.endpoint = llmConfig.sglangEndpoint || 'chat';
                    options.model = llmConfig.sglangModel || process.env.SGLANG_CHAT_MODEL || undefined;
                    break;
                }
                case 'gemini':
                    llmService = geminiService;
                    options.apiKey = llmConfig.apiKey || process.env.GEMINI_API_KEY;
                    options.geminiModel = llmConfig.geminiModel || process.env.GEMINI_MODEL || 'gemini-flash-latest';
                    break;
                case 'groq':
                    llmService = groqService;
                    options.model = llmConfig.groqModel || process.env.GROQ_MODEL || 'llama-3.1-70b-versatile';
                    options.apiKey = llmConfig.groqApiKey || llmConfig.apiKey || process.env.GROQ_API_KEY;
                    break;
                case 'claude':
                    llmService = claudeService;
                    options.apiKey = llmConfig.claudeApiKey || process.env.ANTHROPIC_API_KEY;
                    options.model = llmConfig.claudeModel || 'claude-3-sonnet-20240229';
                    break;
                case 'openai':
                    llmService = openaiService;
                    options.apiKey = llmConfig.openaiApiKey || process.env.OPENAI_API_KEY;
                    options.model = llmConfig.openaiModel || 'gpt-4o';
                    break;
                case 'ollama':
                    llmService = ollamaService;
                    options.ollamaUrl = llmConfig.ollamaUrl || process.env.OLLAMA_API_BASE_URL;
                    options.model = llmConfig.ollamaModel || process.env.OLLAMA_DEFAULT_MODEL || 'qwen3.5:9b';
                    break;
                default:
                    continue;
            }

            if (!options.apiKey && provider !== 'ollama' && provider !== 'sglang') {
                log.info('TUTOR', `Skipping ${provider} — no API key configured.`);
                continue;
            }

            const response = await llmService.generateContentWithHistory(
                chatHistory,
                currentQuery,
                systemPrompt,
                options
            );

            if (response && response.trim().length > 0) {
                log.info('TUTOR', `✅ Success with ${provider}`);
                return response.trim();
            }
        } catch (err) {
            log.warn('TUTOR', `⚠️ ${provider} attempt failed: ${err.message}`);
            lastError = err;
        }
    }

    throw new Error(`All LLM providers failed. Last error: ${lastError?.message}`);
}

/**
 * Multi-dimensional assessment of student response with emotional state detection.
 * Ported from Team1-6's assessStudentResponse — produces richer signals for the FSM.
 * Purpose: Evaluate not just correctness, but also confidence, emotional state, and effort
 * to enable adaptive, empathetic tutoring. This richer signal drives the FSM's pedagogical decisions.
 *
 * What it assesses:
 * - understanding: CORRECT | PARTIAL | MISCONCEPTION | VAGUE | NO_FOUNDATION
 * - confidence: HIGH | MEDIUM | LOW (student's self-assurance in their answer)
 * - emotionalState: CURIOUS | CONFIDENT | UNCERTAIN | FRUSTRATED | BORED
 * - effortLevel: HIGH | MEDIUM | LOW (inferred from response length/detail)
 * - specificGaps: Array of identified knowledge gaps for targeted remediation
 * - reasoning: One-sentence LLM explanation of the classification
 *
 * Design principle: Be GENEROUS — reward genuine understanding even if incomplete.
 * Only classify as PARTIAL if there's a specific, identifiable gap that matters.
 *
 * @param {string} studentResponse - The student's answer text
 * @param {string} moduleTitle - Current topic/subtopic being taught
 * @param {string} lastQuestion - The tutor's preceding question
 * @param {Object} llmConfig - LLM provider configuration for assessment call
 * @param {Array} conversationHistory - Optional prior turns for context
 * @returns {Promise<Object>} Assessment object with understanding, confidence, emotionalState, etc.
 */
async function assessStudentResponse(studentResponse, moduleTitle, lastQuestion, llmConfig, conversationHistory = []) {
    const prompt = `You are an expert educational assessor evaluating a student's response during an AI tutoring session.

Topic: ${moduleTitle}
Tutor's Last Question: ${lastQuestion}
Student's Response: ${studentResponse}

Classification guide (be GENEROUS — reward genuine understanding):
- CORRECT: Student demonstrates clear understanding of the concept with correct information. Does NOT need to be perfect — covering the key idea counts as CORRECT.
- PARTIAL: Student shows some understanding but has a specific, identifiable gap in their answer.
- MISCONCEPTION: Student gives a factually incorrect answer that shows a clear wrong mental model.
- VAGUE: Student's response is too vague or off-topic to assess understanding.
- NO_FOUNDATION: Student has no idea or says "I don't know".

IMPORTANT: If the student correctly identifies the main concept and gives a reasonable explanation or example — even if incomplete — classify as CORRECT. Only use PARTIAL if there is a SPECIFIC identifiable gap that matters.

Return ONLY valid JSON:
{
  "understanding": "CORRECT|PARTIAL|MISCONCEPTION|VAGUE|NO_FOUNDATION",
  "confidence": "HIGH|MEDIUM|LOW",
  "emotionalState": "CURIOUS|CONFIDENT|UNCERTAIN|FRUSTRATED|BORED",
  "effortLevel": "HIGH|MEDIUM|LOW",
  "specificGaps": [],
  "reasoning": "Brief one-sentence explanation of your classification"
}`;

    try {
        const responseText = await generateWithFallback(
            [],
            prompt,
            'You are an expert educational assessor. Respond with ONLY valid JSON.',
            llmConfig,
            { jsonMode: true }
        );

        const jsonMatch = responseText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }
        throw new Error('No JSON found in assessment response');
    } catch (error) {
        log.warn('TUTOR', `Assessment failed, using fallback: ${error.message}`);
        return {
            understanding: 'VAGUE',
            confidence: 'MEDIUM',
            emotionalState: 'UNCERTAIN',
            effortLevel: 'MEDIUM',
            specificGaps: [],
            reasoning: 'Fallback assessment due to provider error'
        };
    }
}

/**
 * Determine adaptive support level based on emotional state + struggle history + response timing.
 * Merges T1-6 emotional detection with tutorStates.js Bloom's support levels.
 * Enhanced with timing signals for mood estimation.
 * Purpose: Dynamically adjust how much scaffolding the tutor provides, balancing
 * Socratic questioning (minimal help) with direct instruction (maximum help) based on
 * the student's current needs.
 *
 * Support levels (from least to most supportive):
 * - MINIMAL: Pure Socratic questioning — student is confident and progressing well
 * - GUIDED: Question + subtle hint — student needs a nudge in the right direction
 * - SCAFFOLDED: Example + explanation + question — student is struggling but engaged
 * - DIRECT: Full explanation + simplified question — student is frustrated or stuck
 *
 * Decision factors:
 * 1. Timing signals: Fast+wrong = guessing → scaffold; Slow+frustrated = give up → direct
 * 2. Emotional state: Frustrated → direct; Bored → guided (re-engage); Confident → minimal
 * 3. Struggle count: 1 wrong → guided; 2 wrong → scaffolded; 3+ wrong → direct
 * 4. Understanding level: NO_FOUNDATION → scaffolded; CORRECT → minimal
 * @param {Object} sessionState - Current session state with struggleCount
 * @param {Object} assessment - Assessment result with understanding, confidence, emotionalState
 * @param {Number} responseTime - Time taken by student to respond (in seconds)
 * @returns {String} Support level: MINIMAL, GUIDED, SCAFFOLDED, or DIRECT
 */
function determineSupportLevel(sessionState, assessment, responseTime = null) {
    const struggleCount = sessionState.struggleCount || sessionState.consecutiveWrong || 0;
    const { confidence, emotionalState, understanding } = assessment;

    // ─── TIMING-BASED MOOD SIGNALS ───
    // Fast response (<15s) + low confidence = guessing → needs scaffolding
    // Slow response (>45s) + low confidence = struggling → needs guidance/scaffolding
    // Very slow (>60s) + frustrated = give up → direct answer
    
    let timingAdjustment = null;
    if (responseTime !== null) {
        if (responseTime > 60 && (emotionalState === EMOTIONAL_STATES.FRUSTRATED || confidence === CONFIDENCE_LEVELS.LOW)) {
            timingAdjustment = 'DIRECT'; // Student taking too long and struggling → direct answer
            log.info('TUTOR', `⏰ Very slow response (${responseTime}s) + struggle → DIRECT answer`);
        } else if (responseTime > 45 && confidence === CONFIDENCE_LEVELS.LOW) {
            timingAdjustment = 'SCAFFOLDED'; // Slow + unsure → provide scaffolding
            log.info('TUTOR', `⏰ Slow response (${responseTime}s) + low confidence → SCAFFOLDED`);
        } else if (responseTime < 15 && confidence === CONFIDENCE_LEVELS.LOW && understanding !== UNDERSTANDING_LEVELS.CORRECT) {
            timingAdjustment = 'SCAFFOLDED'; // Fast but wrong → likely guessing
            log.info('TUTOR', `⏰ Fast response (${responseTime}s) + incorrect → likely guessing, SCAFFOLDED`);
        }
    }

    // Apply timing adjustment if present (overrides normal flow for extreme cases)
    if (timingAdjustment) {
        return SUPPORT_LEVELS[timingAdjustment];
    }

    // ─── STANDARD SUPPORT LEVEL LOGIC ───
    // Frustrated or 3+ consecutive failures → direct reteaching
    if (struggleCount >= 3 || emotionalState === EMOTIONAL_STATES.FRUSTRATED) {
        log.info('TUTOR', `📝 ${struggleCount >= 3 ? 'Multiple failures' : 'Frustrated'} → DIRECT answer`);
        return SUPPORT_LEVELS.DIRECT;
    }
    // No foundation or 2 failures → scaffolded examples
    if (understanding === UNDERSTANDING_LEVELS.NO_FOUNDATION || struggleCount === 2) return SUPPORT_LEVELS.SCAFFOLDED;
    // Bored → switch to guided to re-engage
    if (emotionalState === EMOTIONAL_STATES.BORED) {
        log.info('TUTOR', `😴 Bored student → GUIDED with engaging content`);
        return SUPPORT_LEVELS.GUIDED;
    }
    // Low confidence or 1 failure → guided hints
    if (struggleCount === 1 || confidence === CONFIDENCE_LEVELS.LOW) return SUPPORT_LEVELS.GUIDED;
    // Student is doing well → pure Socratic questioning
    return SUPPORT_LEVELS.MINIMAL;
}

/**
 * Clean LLM response to ensure no internal thinking leaks
 *  Purpose: Strip any reasoning tags or internal monologue that the LLM might emit
 * (e.g., "<thinking>...</thinking>", "Thought: ...") before showing the response
 * to the student. Ensures a clean, tutor-appropriate output.
 *
 * @param {string} text - Raw LLM output
 * @returns {string} Sanitized response ready for student display
 */
function cleanResponse(text) {
    if (!text) return "";
    return text
        .replace(/^(Thinking Process|Thought|Analysis|Reasoning|Internal Monologue|Thought Process):?\s*/i, '')
        .replace(/\n(Thinking Process|Thought|Analysis|Reasoning|Internal Monologue|Thought Process):?\s*/i, '\n')
        .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
        .trim();
}

/**
 * getLLMService — kept for backward compat with startSocraticSession streaming path.
 * All new code should use generateWithFallback() instead.
 */
function getLLMService(llmConfig) {
    let llmService;
    if (llmConfig.llmProvider === 'ollama') {
        llmService = ollamaService;
    } else if (llmConfig.llmProvider === 'groq') {
        llmService = groqService;
    } else if (llmConfig.llmProvider === 'claude') {
        llmService = claudeService;
    } else if (llmConfig.llmProvider === 'openai') {
        llmService = openaiService;
    } else {
        llmService = geminiService;
    }

    const llmOptions = {
        ...(llmConfig.llmProvider === 'ollama' && { model: llmConfig.ollamaModel }),
        ...(llmConfig.llmProvider === 'groq' && { model: llmConfig.groqModel }),
        ...(llmConfig.llmProvider === 'gemini' && { geminiModel: llmConfig.geminiModel || process.env.GEMINI_MODEL }),
        ...(llmConfig.llmProvider === 'claude' && { model: llmConfig.claudeModel }),
        ...(llmConfig.llmProvider === 'openai' && { model: llmConfig.openaiModel }),
        apiKey: llmConfig.apiKey,
        ollamaUrl: llmConfig.ollamaUrl
    };
    return { llmService, llmOptions };
}

function fallbackLearningSteps() {
    return ['definition', 'core idea', 'example', 'application'];
}

/**
 * Fetch pre-computed Socratic content from Redis (written by Python rag_service).
 * Cache key format matches Python: im_cache:socratic_precompute:{course}:{topic_id}
 * Purpose: Enable zero-latency tutor responses by caching pre-generated intros,
 * questions, and teaching notes. Cache key format: im_cache:socratic_precompute:{course}:{topic_id}
 *
 * Fallback: Returns null if Redis unavailable or key not found → caller falls back to LLM generation.
 *
 * @param {string} courseName - Course identifier (e.g., "machine-learning-basics")
 * @param {string} topicId - Topic or subtopic identifier
 * @returns {Promise<Object|null>} Cached content object or null if not found
 */
async function getPrecomputedContent(courseName, topicId) {
    if (!courseName || !topicId) return null;
    try {
        const key = `im_cache:socratic_precompute:${courseName.toLowerCase()}:${topicId.toLowerCase()}`;
        const cached = await redisClient.get(key);
        return cached ? JSON.parse(cached) : null;
    } catch (_) {
        return null;
    }
}

/**
 * Pick a precomputed question for a given cognitive level.
 * Maps L1→easy, L2→medium, L3→hard, L4→expert.
 * 
 * Purpose: Select an appropriately-difficult Socratic question from the precomputed pool
 * based on the student's current cognitive level (Bloom's taxonomy).
 *
 * Mapping: L1_CONCEPT → easy | L2_APPLICATION → medium | L3_CRITICAL → hard | L4_EVALUATION → expert
 * preventing repetition if the student retries the same level.
 *
 * @param {Object} precomputed - Cached content from getPrecomputedContent()
 * @param {string} cognitiveLevel - One of COGNITIVE_LEVELS (L1_CONCEPT, etc.)
 * @param {Number} questionIndex - Which question in the bucket to pick (0, 1, or 2)
 * @returns {string|null} Selected question text or null if not available
 */
function pickPrecomputedQuestion(precomputed, cognitiveLevel, questionIndex = 0) {
    if (!precomputed?.questions) return null;
    const levelMap = {
        L1_CONCEPT: 'easy',
        L2_APPLICATION: 'medium',
        L3_CRITICAL: 'hard',
        L4_EVALUATION: 'expert'
    };
    const bucket = levelMap[cognitiveLevel] || 'easy';
    const questions = precomputed.questions[bucket];
    if (!Array.isArray(questions) || questions.length === 0) return null;
    return questions[Math.min(questionIndex, questions.length - 1)];
}

/**
 * Resume or start a tutor session for a user+course.
 * Returns greeting message, current position, and precomputed intro if available.
 * Purpose: Initialize or continue a learning session, providing a personalized greeting
 * and positioning the student at their last progress point (or the start if new).
 *
 * Returns:
 * - isNew: Boolean indicating if this is a first-time session
 * - greeting: Personalized welcome message with progress context
 * - position: Current curriculum position (module/topic/subtopic)
 * - precomputed: Optional cached intro + first question for instant start
 * - progress: Summary of completed items and last active date
 *
 * @param {string} userId - Student identifier
 * @param {string} courseName - Course to resume/start
 * @returns {Promise<Object>} Session initialization object
 */
async function resumeOrStartSession(userId, courseName) {
    const progress = await loadUserProgress(userId, courseName);
    const completedSubtopics = progress?.completedSubtopics || [];
    const completedTopics = progress?.completedTopics || [];
    const isReturning = completedSubtopics.length > 0 || completedTopics.length > 0;

    let position = null;
    try {
        position = await resolveCurrentPosition(courseName, completedSubtopics, completedTopics);
    } catch (e) {
        return {
            isNew: true,
            greeting: `Welcome to **${courseName}**! There was an issue loading the curriculum: ${e.message}`,
            position: null,
            error: e.message
        };
    }

    // Fetch precomputed intro from Redis (populated by Python precompute job)
    let precomputed = null;
    const lookupId = position?.topicId || position?.subtopicId;
    if (lookupId) {
        precomputed = await getPrecomputedContent(courseName, lookupId);
    }

    const subtopicLabel = position?.subtopicName || position?.topicName || 'the first topic';
    const moduleLabel = position?.moduleName || '';

    let greeting;
    if (position?.isComplete) {
        greeting = `🎉 Congratulations! You've completed the **${courseName}** course. Amazing work!`;
    } else if (isReturning) {
        const lastDate = progress?.lastActiveDate
            ? `Last session: ${new Date(progress.lastActiveDate).toLocaleDateString()}.`
            : '';
        greeting = `Welcome back! ${lastDate} You were studying **${subtopicLabel}**${moduleLabel ? ` in *${moduleLabel}*` : ''}. Let's continue where you left off.`;
    } else {
        greeting = `Welcome to **${courseName}**! 🚀 Let's start your learning journey. We'll begin with **${subtopicLabel}**${moduleLabel ? ` in *${moduleLabel}*` : ''}.`;
    }

    // Append precomputed intro summary if available
    if (precomputed?.intro_summary && !position?.isComplete) {
        greeting += `\n\n${precomputed.intro_summary}`;
    }

    return {
        isNew: !isReturning,
        greeting,
        position,
        precomputed: precomputed ? {
            intro_summary: precomputed.intro_summary,
            firstQuestion: pickPrecomputedQuestion(precomputed, 'L1_CONCEPT', 0)
        } : null,
        completedSubtopics,
        completedTopics,
        progress: {
            completedCount: completedSubtopics.length,
            lastActiveDate: progress?.lastActiveDate || null
        }
    };
}

async function buildInitialLearningPath(courseName, position = null) {
    const concept = position?.subtopicName || position?.topicName || position?.moduleName || 'general';
    try {
        if (!courseName || courseName === 'General') {
            return {
                concept,
                steps: fallbackLearningSteps(),
                currentStep: 0
            };
        }

        const structure = await getCurriculumStructure(courseName);
        const positionSteps = Array.isArray(position?.steps) && position.steps.length > 0 ? position.steps : null;
        const steps = positionSteps
            || (Array.isArray(structure?.learningSteps) && structure.learningSteps.length > 0 ? structure.learningSteps : null)
            || (Array.isArray(structure?.meta?.learningSteps) && structure.meta.learningSteps.length > 0 ? structure.meta.learningSteps : null)
            || fallbackLearningSteps();

        return {
            concept,
            steps,
            currentStep: 0
        };
    } catch (_error) {
        return {
            concept,
            steps: fallbackLearningSteps(),
            currentStep: 0
        };
    }
}

/**
 * Generate initial response for a topic/subtopic.
 * Uses precomputed intro+question from Redis when available (zero LLM latency).
 * Purpose: Start a Socratic lesson by introducing the topic conversationally and
 * asking the first guiding question. Prioritizes precomputed content for zero latency;
 * falls back to LLM generation if cache miss.
 *
 * Response structure (when generated):
 * 1. Two short paragraphs explaining the foundational concept
 * 2. One simple real-world analogy or example (max 3 sentences)
 * 3. ONE focused Socratic question requiring reasoning (no answer provided)
 *
 * Constraints: Natural dialogue tone, <200 words total, no structural headings,
 * no internal thinking tags.
 *
 * @param {string} topic - Topic/subtopic name to teach
 * @param {string} context - Background knowledge/context for factual grounding
 * @param {Object} llmConfig - LLM provider configuration
 * @param {Object} position - Optional curriculum position for contextual prompts
 * @param {Function} onToken - Optional streaming callback for real-time output
 * @returns {Promise<string>} Sanitized introductory response + first question
 */
async function startSocraticSession(topic, context, llmConfig, position = null, onToken = null) {
    // ── Fast path: use precomputed content from Redis ──────────────────────────
    const courseName = position?.courseName || position?.course;
    const topicId = position?.topicId || position?.subtopicId;
    if (courseName && topicId) {
        try {
            const precomputed = await getPrecomputedContent(courseName, topicId);
            if (precomputed?.intro_summary && precomputed?.questions?.easy?.length) {
                const firstQ = precomputed.questions.easy[0];
                const response = `${precomputed.intro_summary}\n\n${firstQ.question}`;
                log.info('TUTOR', `startSocraticSession: using precomputed content for ${courseName}/${topicId}`);
                return sanitizeGeneratedText(response);
            }
        } catch (_) { /* fall through to LLM */ }
    }

    // ── Slow path: generate via LLM ───────────────────────────────────────────
    const { llmService, llmOptions } = getLLMService(llmConfig);
    const prompt = `Start a Study Mode Socratic lesson on: ${safe(topic)}`;

    let systemPrompt = `You are iMentor, an AI-powered learning platform designed to provide guided, Socratic learning.
Your goal is to introduce the topic clearly and conversationally.

TOPIC TO INTRODUCE: "${safe(topic)}"

CONTEXT FOR FACTS:
${safe(context)}

You must respond conversationally and naturally like a human tutor, but internally logically follow this progression:
1. Explain the foundational concept in EXACTLY two short, easily digestible paragraphs.
2. Provide a simple real-world analogy or example to make the concept concrete (max 3 sentences).
3. Ask ONE focused Socratic question requiring the student to reason about what was explained. Do NOT provide the answer. Stop and wait for the student.

CRITICAL RULES:
- DO NOT use any structural headings (like "1. Concept", "### 1️⃣ CONCEPT", "Intuition", etc.).
- Render the output as natural, conversational dialogue.
- KEEP THE ENTIRE RESPONSE COMPACT. Total length should be under 200 words.
- Use paragraph breaks to separate the concept, example, and question.
- DO NOT output any internal thinking tags.`;

    if (position) {
        systemPrompt += `\n\n📍 POSITION: ${safe(position.moduleName || 'General')} -> ${safe(position.topicName || topic)}`;
    }

    try {
        if (onToken && (llmConfig.llmProvider === 'gemini' || llmConfig.llmProvider === 'groq')) {
            log.info('TUTOR', `Starting streamed intro for ${topic}...`);
            const response = await llmStreamingService.streamCompletion({
                messages: [{ role: 'user', content: prompt }],
                provider: llmConfig.llmProvider,
                model: llmOptions.geminiModel || llmOptions.model,
                apiKey: llmOptions.apiKey,
                systemPrompt: systemPrompt,
                onToken,
                options: llmOptions
            });
            console.log(`\n\n[STUDENT INPUT]\nTopic: ${topic}\n\n[TUTOR ACTION]\nExplaining fundamentals\n\n[SOCRATIC QUESTION]\n(Waiting for response)\n`);
            return sanitizeGeneratedText(response);
        }

        const response = await llmService.generateContentWithHistory(
            [],
            prompt,
            systemPrompt,
            llmOptions
        );

        console.log(`\n\n[STUDENT INPUT]\nTopic: ${topic}\n\n[TUTOR ACTION]\nExplaining fundamentals\n\n[SOCRATIC QUESTION]\n(Waiting for response)\n`);
        return sanitizeGeneratedText(cleanResponse(response));
    } catch (error) {
        log.error('TUTOR', `Start session error: ${error.message}`, error);
        return sanitizeGeneratedText(`We are going to learn about ${safe(topic)}. This is a fundamental idea to grasp.\n\nWhat do you already know about this topic?`);
    }
}

/**
 * Process the response loop
 * Purpose: Handle a student's answer, assess understanding/emotion, decide pedagogical action,
 * generate an adaptive follow-up, and update session state — all while supporting streaming,
 * fallback providers, and gamification integration.
 *
 * High-level flow:
 * 1. LOAD: Fetch current session state + student knowledge profile
 * 2. ANALYZE: Call assessStudentResponse() for multi-dimensional evaluation
 * 3. ADAPT: Call determineSupportLevel() + decideTeachingAction() for strategy
 * 4. GENERATE: Prompt LLM with rich context (emotion, support level, knowledge gaps, policy)
 * 5. UPDATE: Adjust mastery score, cognitive level, learning path, and history
 * 6. PERSIST: Save session state + update StudentKnowledgeState (fire-and-forget)
 * 7. RETURN: Structured response for UI rendering (question, classification, progress, etc.)
 *
 * Key adaptive features:
 * - Emotional awareness: Adjusts tone and support based on frustration/boredom/confidence
 * - Timing signals: Uses response time to infer guessing vs. struggling
 * - Struggle escalation: Increases scaffolding after consecutive wrong answers
 * - Mastery tracking: Cumulative score + consecutive correct answers determine advancement
 * - Gamification: Awards credits/XP on subtopic mastery (fire-and-forget)
 * - Knowledge persistence: Updates StudentKnowledgeState with misconceptions, patterns, mastery
 *
 * @param {string} studentResponse - The student's answer to the last question
 * @param {string} sessionId - Unique session identifier for state management
 * @param {Object} llmConfig - LLM provider configuration
 * @param {Function} onProgress - Optional callback for status updates
 * @param {Function} onToken - Optional streaming callback for token-by-token output
 * @param {Object} metadata - Optional metadata (e.g., responseTime for mood estimation)
 * @returns {Promise<Object>} Structured response with follow-up question, classification, progress, etc.
 */
async function processTutorResponse(studentResponse, sessionId, llmConfig, onProgress, onToken = null, metadata = {}) {
    const state = await getTutorSessionState(sessionId);
    if (!state) return null;

    // Extract response time for mood estimation (if provided)
    const responseTime = metadata.responseTime || null;
    if (responseTime) {
        log.info('TUTOR', `⏱️ Student response time: ${responseTime}s`);
    }

    // ─── StudentKnowledgeState integration (Fix #2) ───
    // Load the student's persistent knowledge profile so the tutor can adapt
    let knowledgeProfile = null;
    try {
        const StudentKnowledgeState = require('../models/StudentKnowledgeState');
        if (state.userId) {
            knowledgeProfile = await StudentKnowledgeState.findOne({ userId: state.userId });
        }
    } catch (kErr) {
        log.warn('TUTOR', `KnowledgeState load non-fatal: ${kErr.message}`);
    }

    const { moduleTitle, lastQuestion, turnCount, masteryScore = 0, history = [] } = state;
    const topic = state.subtopicName || state.teachingUnit || moduleTitle;
    const learningPath = state.learningPath || {
        concept: topic || 'general',
        steps: fallbackLearningSteps(),
        currentStep: 0
    };
    const learningSteps = Array.isArray(learningPath.steps) && learningPath.steps.length > 0
        ? learningPath.steps
        : fallbackLearningSteps();
    const currentStep = Math.max(0, Math.min(Number(learningPath.currentStep || 0), learningSteps.length - 1));
    const remainingSteps = Math.max(0, learningSteps.length - 1 - currentStep);

    const position = {
        moduleName: state.moduleName || state.moduleTitle,
        topicName: state.topicName || moduleTitle,
        subtopicName: state.subtopicName || state.teachingUnit || moduleTitle,
        moduleIndex: state.moduleIndex,
        topicIndex: state.topicIndex,
        subtopicIndex: state.subtopicIndex,
        subtopicId: state.subtopicId,
        topicId: state.topicId
    };

    let unitContext = "";
    if (state.courseName && (state.subtopicId || state.topicId)) {
        try {
            const contextData = await getSubtopicContext(state.courseName, state.subtopicId, state.topicId);
            if (contextData?.qdrant_chunks?.length > 0) {
                unitContext = contextData.qdrant_chunks.map(c => c.text).join('\n\n').substring(0, 2000);
            }
        } catch (ctxErr) {
            log.warn('TUTOR', `Context error: ${ctxErr.message}`);
        }
    }

    const { llmService, llmOptions } = getLLMService(llmConfig);

    // Optional ReAct wrapper path (feature-flagged, non-breaking)
    if (ENABLE_TUTOR_REACT_WRAPPER && reactOrchestrator?.executeReActCycle) {
        try {
            const reactResult = await reactOrchestrator.executeReActCycle({
                studentResponse,
                moduleTitle: topic,
                lastQuestion,
                llmConfig,
                masteryScore,
                cognitiveLevel: state.cognitiveLevel || COGNITIVE_LEVELS.L1_CONCEPT,
                history,
                turnCount,
                position,
                onProgress,
                context: unitContext,
                onToken
            });

            const newState = {
                ...state,
                lastQuestion: reactResult.followUpQuestion,
                turnCount: turnCount + 1,
                masteryScore: Math.max(0, reactResult.masteryScore || masteryScore),
                cognitiveLevel: reactResult.nextLevel || state.cognitiveLevel || COGNITIVE_LEVELS.L1_CONCEPT,
                learningPath: {
                    concept: learningPath.concept || topic,
                    steps: learningSteps,
                    currentStep: reactResult.isMastered
                        ? Math.min(currentStep + 1, Math.max(0, learningSteps.length - 1))
                        : currentStep
                },
                history: [...history, {
                    status: reactResult.classification,
                    response: studentResponse,
                    score: reactResult.score || 0
                }],
                socraticState: reactResult.socraticState || reactResult.classification
            };
            await setTutorSessionState(sessionId, newState);

            return {
                followUpQuestion: reactResult.followUpQuestion,
                classification: reactResult.classification,
                pedagogicalMove: reactResult.pedagogicalMove,
                reasoning: reactResult.reasoning,
                isMastered: !!reactResult.isMastered,
                socraticState: reactResult.socraticState,
                position,
                masteryProgress: { current: newState.masteryScore, required: 3.5 },
                steps: reactResult.steps || []
            };
        } catch (reactErr) {
            log.warn('TUTOR', `ReAct wrapper failed, fallback to legacy Socratic loop: ${reactErr.message}`);
        }
    }

    // ANALYZE Phase — Multi-dimensional assessment with emotional state
    if (onProgress) onProgress('Evaluating understanding and emotional state...');

    const assessment = await assessStudentResponse(
        studentResponse,
        topic,
        lastQuestion,
        llmConfig,
        history
    );

    const supportLevel = determineSupportLevel(state, assessment, responseTime);
    log.info('TUTOR', `📊 Assessment: ${assessment.understanding} | Emotion: ${assessment.emotionalState} | Support: ${supportLevel} | Confidence: ${assessment.confidence}`);

    // Map rich understanding → legacy classification object for backward compat
    const classificationStatusMap = {
        'CORRECT': 'CORRECT',
        'PARTIAL': 'PARTIAL',
        'MISCONCEPTION': 'WRONG',
        'VAGUE': 'PARTIAL',
        'NO_FOUNDATION': 'UNKNOWN'
    };
    const classification = {
        status: classificationStatusMap[assessment.understanding] || assessment.understanding || 'PARTIAL',
        reasoning: assessment.reasoning || 'Could not parse',
        action: assessment.understanding === 'CORRECT' ? 'Deepen topic' : 'Clarify concept',
        emotionalState: assessment.emotionalState || EMOTIONAL_STATES.UNCERTAIN,
        confidence: assessment.confidence || CONFIDENCE_LEVELS.MEDIUM,
        specificGaps: assessment.specificGaps || []
    };

    // MANDATORY CLEAN LOGGING
    console.log(`\n\n[STUDENT INPUT]\nTopic: ${topic}\nInput: ${studentResponse}\n`);
    console.log(`[TUTOR ACTION]\nStatus: ${classification.status}\nAction: ${classification.action}\n`);

    // ADAPT Phase
    if (onProgress) onProgress('Adapting teaching strategy...');

    // Load state machine state to inject cognitive-level guidance into system prompt
    let smState = null;
    let tutorSM = null;
    try {
        tutorSM = require('./tutorStateMachine');
        smState = await tutorSM.getSessionState(sessionId);
    } catch (_e) { /* non-fatal — continue without smState */ }

    const cognitiveNote = smState
        ? `\nCOGNITIVE LEVEL: ${smState.cognitiveLevelName || 'L1'} | Mastery: ${Math.round((smState.masteryScore || 0) * 25)}% | Hints Used: ${smState.hintsGiven || 0}\nCalibrate your question difficulty to this level: L1=recall, L2=explain, L3=apply, L4=design.\n`
        : '';

    // Emotional state & support level guidance injected into teaching policy
    const emotionalNote = `
STUDENT EMOTIONAL STATE: ${classification.emotionalState || 'UNCERTAIN'}
CONFIDENCE: ${classification.confidence || 'MEDIUM'}
SUPPORT LEVEL: ${supportLevel}
${classification.specificGaps?.length ? `SPECIFIC KNOWLEDGE GAPS: ${classification.specificGaps.join(', ')}` : ''}
${supportLevel === 'DIRECT' ? 'IMPORTANT: Student is frustrated or struggling. Use simple language, give direct explanations, and be extra encouraging.' : ''}
${supportLevel === 'SCAFFOLDED' ? 'NOTE: Student needs scaffolding. Provide worked examples before asking questions.' : ''}
${classification.emotionalState === 'BORED' ? 'NOTE: Student appears disengaged. Use a surprising fact, real-world connection, or challenge to re-engage.' : ''}`;

    // ─── Inject persistent knowledge state into teaching context ───
    let knowledgeNote = '';
    if (knowledgeProfile) {
        try {
            const conceptMatch = knowledgeProfile.getConcept(topic);
            const struggling = knowledgeProfile.getStrugglingConcepts().slice(0, 3).map(c => c.conceptName);
            const mastered = knowledgeProfile.getMasteredConcepts().slice(0, 3).map(c => c.conceptName);
            const misconceptions = conceptMatch?.misconceptions?.filter(m => m.stillPresent).map(m => m.description) || [];
            const learningPatterns = conceptMatch?.learningPatterns || {};

            knowledgeNote = `
STUDENT KNOWLEDGE PROFILE:
- Learning Pace: ${knowledgeProfile.learningProfile?.learningPace || 'moderate'}
- Preferred Depth: ${knowledgeProfile.learningProfile?.preferredDepth || 'balanced'}
- Challenge Response: ${knowledgeProfile.learningProfile?.challengeResponse || 'needs_encouragement'}
${conceptMatch ? `- Current Topic Mastery: ${conceptMatch.masteryScore}% (${conceptMatch.understandingLevel})` : ''}
${misconceptions.length > 0 ? `- ACTIVE MISCONCEPTIONS to address: ${misconceptions.join('; ')}` : ''}
${learningPatterns.respondsWellTo?.length ? `- Student responds well to: ${learningPatterns.respondsWellTo.join(', ')}` : ''}
${struggling.length > 0 ? `- Struggling with: ${struggling.join(', ')}` : ''}
${mastered.length > 0 ? `- Already mastered: ${mastered.join(', ')}` : ''}`;
        } catch (kErr) {
            log.warn('TUTOR', `KnowledgeState injection non-fatal: ${kErr.message}`);
        }
    }

    const policyAction = decideTeachingAction({
        masteryScore: Math.max(0, Math.min(1, (smState?.masteryScore || 0) / 4)),
        consecutiveWrong: smState?.consecutiveWrong || 0,
        hintUsage: smState?.hintsGiven || 0,
        cognitiveLevel: smState?.cognitiveLevelName || state.cognitiveLevel || COGNITIVE_LEVELS.L1_CONCEPT,
        currentStep,
        remainingSteps
    });

    const reflectionAction = reflectOnTeaching({
        consecutiveWrong: state.consecutiveWrong,
        hintsGiven: state.hintsGiven,
        masteryScore: state.masteryScore,
        learningPath: state.learningPath
    });

    const finalAction = reflectionAction || policyAction;

    log.info('TUTOR_REFLECTION', {
        sessionId,
        policyAction,
        reflectionAction,
        finalAction
    });

    const consecutiveWrongNow = (state.consecutiveWrong || 0) + (
        (classification.status === 'WRONG' || classification.status === 'UNKNOWN') ? 1 : 0
    );
    // Visual aid is generated by the caller (tutorHandler) when struggle >= 2
    const visualAidNeeded = consecutiveWrongNow >= 2;

    const policyInstruction = {
        ASK_QUESTION: 'Ask one focused Socratic question and wait for the student response.',
        GIVE_HINT: 'Give a concise hint before asking the next Socratic question.',
        EXPLAIN_CONCEPT: 'Briefly re-explain the core concept in simpler language, then ask a check question.',
        SIMPLIFY_PROBLEM: 'Reduce complexity and ask a foundational sub-question first.',
        ADVANCE_DIFFICULTY: 'Increase difficulty slightly with a deeper application question.',
        RETEACH_CONCEPT: 'Reteach the current concept step-by-step and verify understanding with one focused question.',
        SKIP_AHEAD: 'Briefly bridge to the next concept step and ask one application-focused question.'
    };

    // ── Struggle-aware encouragement prefix injected into every non-correct response ──
    const struggleEncouragement = consecutiveWrongNow === 1
        ? `You're getting there — this concept takes a bit of practice. Here's a nudge in the right direction:\n\n`
        : consecutiveWrongNow === 2
        ? `No worries at all — this is genuinely one of the trickier ideas. Let me break it down a different way:\n\n`
        : consecutiveWrongNow >= 3
        ? `You're building persistence — that's key in ML! Let me give you the full picture step by step, and we'll work through it together:\n\n`
        : '';

    const systemPrompt = `You are iMentor's interactive cognitive learning system (Study Mode).
Your task is to adaptively teach and guide the student. ALWAYS be warm, encouraging, and patient. NEVER give up on the student.
${cognitiveNote}
${emotionalNote}
${knowledgeNote}
TEACHING POLICY DECISION: ${policyAction}
REFLECTION ADJUSTMENT: ${reflectionAction || "none"}
FINAL TEACHING ACTION: ${finalAction}
TEACHING POLICY INSTRUCTION: ${policyInstruction[finalAction] || policyInstruction.ASK_QUESTION}
TOPIC: "${safe(topic)}"
LAST QUESTION: "${safe(lastQuestion)}"
STUDENT'S ANSWER: "${safe(studentResponse)}"
CONSECUTIVE STRUGGLES ON THIS SUBTOPIC: ${consecutiveWrongNow}

Internal Evaluation:
Status: ${safe(classification.status)}
Reasoning: ${safe(classification.reasoning)}

${supportLevel === 'DIRECT' ? `
CRITICAL: DIRECT ANSWER MODE ACTIVATED
The student has struggled ${consecutiveWrongNow} times. You MUST:
1. Start with genuine encouragement — acknowledge this is hard
2. Provide the COMPLETE, DIRECT answer to the previous question
3. Explain it step-by-step using the simplest possible language
4. Use a concrete real-world analogy or example (e.g., everyday objects, cooking, sports)
5. Include a SHORT mermaid flowchart or pseudocode to visualise the concept if helpful. Use fenced code block:
   \`\`\`mermaid
   flowchart TD
       A[Start] --> B[Step]
   \`\`\`
   Keep it under 8 nodes.
6. Then ask ONE simple comprehension check question (must be very easy — just verifying they followed)
7. Remind them: "Don't worry, this is a tough concept. It's completely normal to need a few tries."

DO NOT continue Socratic questioning until they've seen the direct explanation.
` : consecutiveWrongNow === 2 ? `
SCAFFOLDED EXPLANATION MODE:
The student has struggled twice. You MUST:
1. Start with encouragement (1 sentence)
2. Gently acknowledge what they got right (even if partial)
3. Re-explain the concept using a NEW angle or analogy — NOT the same explanation again
4. Give a worked example: walk through a concrete case step by step
5. Include a SHORT mermaid flowchart OR pseudocode block to visualise the concept. Use a fenced code block like:
   \`\`\`mermaid
   flowchart TD
       A[Start] --> B[Step]
   \`\`\`
   Keep it under 8 nodes. Only include a diagram if it meaningfully clarifies the concept.
6. Ask ONE simpler, more-guided question (break the original question into a smaller piece)
` : `
ADAPTIVE INSTRUCTION:
${classification.status === 'CORRECT' ? 'The student answered correctly. Briefly celebrate with genuine warmth, then deepen the topic.' : ''}
${classification.status === 'PARTIAL' ? `The student has partial understanding. Start by acknowledging what they GOT RIGHT. Then clarify the specific gap: "${safe(classification.reasoning || 'the concept')}". Ask a more focused follow-up.` : ''}
${classification.status === 'WRONG' ? `The student answered incorrectly. Be encouraging — say something like "Good attempt!". Give a hint that points toward the right answer without giving it away. Ask a simpler version of the question.` : ''}
${classification.status === 'UNKNOWN' ? `The student doesn't know. That's OK! Provide a direct hint and break the concept into the smallest possible sub-question first.` : ''}

You must respond like a warm, patient human tutor:
1. Acknowledge the student's effort (1 sentence)
2. Correct or affirm with explanation + analogy (MAX 2 paragraphs)
3. Ask ONE focused Socratic question requiring reasoning. Stop and wait.
`}

CRITICAL RULES:
- DO NOT use structural headings (like "### Phase 1", "Step 1:", etc.)
- Write as natural dialogue, NOT a lecture.
- KEEP IT CONCISE. Total length must be under 250 words to avoid cognitive overload.
- Use paragraph breaks between explanation, analogy, and question.
- DO NOT output internal thinking tags.
- Be specific to the topic: "${safe(topic)}"
- Adapt complexity to the student's preferred depth: ${knowledgeProfile?.learningProfile?.preferredDepth || 'balanced'}.`;

    const prompt = `Please respond to the student focusing on ${safe(topic)}.`;

    // ── Inject precomputed hint when student is struggling ─────────────────────
    // If consecutive wrong answers ≥ 1, append expected_answer_nature as a hint
    let precomputedHint = '';
    if ((state.consecutiveWrong || 0) >= 1 && state.courseName && (state.topicId || state.subtopicId)) {
        try {
            const pc = await getPrecomputedContent(state.courseName, state.topicId || state.subtopicId);
            if (pc) {
                const levelMap = { L1_CONCEPT: 'easy', L2_APPLICATION: 'medium', L3_CRITICAL: 'hard', L4_EVALUATION: 'expert' };
                const bucket = levelMap[state.cognitiveLevel] || 'easy';
                const qIdx = Math.min((state.consecutiveWrong || 1) - 1, 2);
                const qEntry = pc.questions?.[bucket]?.[qIdx];
                if (qEntry?.expected_answer_nature) {
                    precomputedHint = `\n\n💡 *Hint: ${qEntry.expected_answer_nature}*`;
                }
            }
        } catch (_) { /* non-fatal */ }
    }

    let followUpQuestion = "";
    try {
        if (onToken && (llmConfig.llmProvider === 'gemini' || llmConfig.llmProvider === 'groq')) {
            followUpQuestion = await llmStreamingService.streamCompletion({
                messages: [{ role: 'user', content: prompt }],
                provider: llmConfig.llmProvider,
                model: llmOptions.geminiModel || llmOptions.model,
                apiKey: llmOptions.apiKey,
                systemPrompt: systemPrompt,
                onToken,
                options: llmOptions
            });
        } else {
            // Use generateWithFallback (tries all configured providers with automatic fallback)
            // instead of single-provider llmService which has no retry mechanism.
            followUpQuestion = await generateWithFallback(
                [],
                prompt,
                systemPrompt,
                llmConfig,
                { maxOutputTokens: 600 }
            );
        }
        followUpQuestion = sanitizeGeneratedText(cleanResponse(followUpQuestion));
        // Prepend encouragement prefix when student is struggling (non-correct)
        if (struggleEncouragement && classification.status !== 'CORRECT' && followUpQuestion) {
            followUpQuestion = struggleEncouragement + followUpQuestion;
        }
        if (precomputedHint) followUpQuestion += precomputedHint;
    } catch (e) {
        // Log to aid debugging — if this still triggers the LLM call is failing
        log.warn('TUTOR', `Question generation FAILED (provider=${llmConfig?.llmProvider}): ${e.message}`);
        // Fallback must be topic-specific, never the generic "rethink" phrase
        const topicSafe = safe(topic) || 'this concept';
        followUpQuestion = sanitizeGeneratedText(
            `${struggleEncouragement}Let me approach **${topicSafe}** from a different angle.\n\nThink about it this way: what would happen in a simple real-world scenario where ${topicSafe} is involved? Start with what you already know and describe it in your own words.`
        );
    }

    console.log(`[SOCRATIC RESPONSE GENERATED]\n${followUpQuestion}\n`);
    console.log(`[WAITING FOR RESPONSE]\n\n`);

    // Update state progression
    let consecutiveCorrect = state.consecutiveCorrect || 0;
    if (classification.status === 'CORRECT') {
        consecutiveCorrect += 1;
    } else if (classification.status === 'WRONG' || classification.status === 'UNKNOWN') {
        consecutiveCorrect = 0; // Reset on definitively wrong answers
    }
    // PARTIAL does NOT increment nor reset consecutiveCorrect

    // Calculate mastery out of 5.0
    // Only CORRECT answers increase mastery meaningfully; PARTIAL gives a small boost
    const masteryDelta = classification.status === 'CORRECT' ? 1.0
        : (classification.status === 'PARTIAL' ? 0.25
        : (classification.status === 'WRONG' || classification.status === 'UNKNOWN' ? -0.5 : 0));
    const newMastery = Math.min(5.0, masteryScore + masteryDelta);
    const projectedMastery = Math.max(0, newMastery);
    // Mastery requires BOTH consecutive correct answers AND a minimum mastery score
    const isMastered = (consecutiveCorrect >= 2 && projectedMastery >= 2.0) || projectedMastery >= 3.5;

    // ── Award gamification credits on subtopic mastery (fire-and-forget) ──────
    if (isMastered && !state.masteryAwarded && state.userId) {
        setImmediate(async () => {
            try {
                const gamificationService = require('./gamificationService');
                const cogLevel = state.cognitiveLevel || COGNITIVE_LEVELS.L1_CONCEPT;
                // Credits scale with Bloom's taxonomy level reached
                const creditsByLevel = { L1_CONCEPT: 10, L2_APPLICATION: 20, L3_CRITICAL: 30, L4_EVALUATION: 40 };
                const credits = creditsByLevel[cogLevel] || 10;
                const reasonMap = { L1_CONCEPT: 'remembering', L2_APPLICATION: 'applying', L3_CRITICAL: 'analyzing', L4_EVALUATION: 'evaluating_creating' };
                await gamificationService.awardLearningCredits(state.userId, credits, reasonMap[cogLevel] || 'understanding', topic);
                log.info('TUTOR', `Awarded ${credits} credits to ${state.userId} for mastering ${topic}`);
                // Mark mastery awarded to prevent double-awarding in same session
                await setTutorSessionState(sessionId, { ...state, masteryAwarded: true });
            } catch (gmErr) {
                log.warn('TUTOR', `Gamification award non-fatal: ${gmErr.message}`);
            }
        });
    }

    let nextCognitiveLevel = state.cognitiveLevel || COGNITIVE_LEVELS.L1_CONCEPT;
    if (reflectionAction === 'SIMPLIFY_PROBLEM') {
        try {
            if (tutorSM?.downgradeCognitiveLevel) {
                const downgraded = await tutorSM.downgradeCognitiveLevel(sessionId);
                nextCognitiveLevel = downgraded?.cognitiveLevelName || downgraded?.cognitiveLevel || nextCognitiveLevel;
            }
        } catch (reflectionErr) {
            log.warn('TUTOR_REFLECTION', `Cognitive downgrade failed (non-fatal): ${reflectionErr.message}`);
        }
    }

    const keepSameStep = reflectionAction === 'RETEACH_CONCEPT';
    const stepIncrement = keepSameStep ? 0 : (isMastered ? 1 : 0);

    const newState = {
        ...state,
        lastQuestion: followUpQuestion,
        turnCount: turnCount + 1,
        consecutiveCorrect,
        consecutiveWrong: (classification.status === 'WRONG' || classification.status === 'UNKNOWN')
            ? (state.consecutiveWrong || 0) + 1
            : (classification.status === 'CORRECT' ? 0 : (state.consecutiveWrong || 0)),
        hintsGiven: reflectionAction === 'RETEACH_CONCEPT' ? 0 : (state.hintsGiven || 0),
        masteryScore: projectedMastery,
        cognitiveLevel: nextCognitiveLevel,
        learningPath: {
            concept: learningPath.concept || topic,
            steps: learningSteps,
            currentStep: Math.min(currentStep + stepIncrement, Math.max(0, learningSteps.length - 1))
        },
        history: [...history, { 
            status: classification.status, 
            response: studentResponse
        }],
        socraticState: classification.status
    };
    await setTutorSessionState(sessionId, newState);

    if (reflectionAction === 'SKIP_AHEAD') {
        try {
            if (tutorSM?.advanceLearningStep) {
                await tutorSM.advanceLearningStep(sessionId);
            }
        } catch (reflectionErr) {
            log.warn('TUTOR_REFLECTION', `Skip-ahead step advance failed (non-fatal): ${reflectionErr.message}`);
        }
    }

    if (reflectionAction === 'RETEACH_CONCEPT') {
        try {
            if (tutorSM?.resetHints) {
                await tutorSM.resetHints(sessionId);
            }
        } catch (reflectionErr) {
            log.warn('TUTOR_REFLECTION', `Hint reset failed (non-fatal): ${reflectionErr.message}`);
        }
    }

    // ─── Fire-and-forget: persist assessment to StudentKnowledgeState ───
    if (state.userId && topic) {
        setImmediate(async () => {
            try {
                const StudentKnowledgeState = require('../models/StudentKnowledgeState');
                let ks = await StudentKnowledgeState.findOne({ userId: state.userId });
                if (!ks) {
                    ks = new StudentKnowledgeState({ userId: state.userId, concepts: [] });
                }
                ks.updateConcept({
                    conceptName: topic,
                    masteryScore: Math.round(newState.masteryScore * 28.6), // Scale 3.5→100
                    understandingLevel: isMastered ? 'mastered' :
                        (classification.status === 'CORRECT' ? 'comfortable' :
                            classification.status === 'PARTIAL' ? 'learning' : 'struggling'),
                    totalInteractions: (ks.getConcept(topic)?.totalInteractions || 0) + 1,
                    successfulInteractions: (ks.getConcept(topic)?.successfulInteractions || 0) +
                        (classification.status === 'CORRECT' ? 1 : 0),
                    lastInteractionDate: new Date()
                });
                // Record misconceptions from assessment
                if (classification.specificGaps?.length > 0 && classification.status !== 'CORRECT') {
                    const concept = ks.getConcept(topic);
                    if (concept) {
                        for (const gap of classification.specificGaps) {
                            const existing = concept.misconceptions?.find(m => m.description === gap);
                            if (!existing) {
                                concept.misconceptions = concept.misconceptions || [];
                                concept.misconceptions.push({ description: gap, stillPresent: true });
                            }
                        }
                    }
                }
                // Mark misconceptions as corrected when student gets it right
                if (classification.status === 'CORRECT') {
                    const concept = ks.getConcept(topic);
                    if (concept?.misconceptions) {
                        concept.misconceptions.forEach(m => { m.stillPresent = false; m.correctedAt = new Date(); });
                    }
                }
                ks.engagementMetrics.totalSessions = (ks.engagementMetrics.totalSessions || 0) + 1;
                ks.engagementMetrics.lastActiveDate = new Date();
                await ks.save();
            } catch (ksErr) {
                log.warn('TUTOR', `KnowledgeState update non-fatal: ${ksErr.message}`);
            }
        });
    }

    return {
        followUpQuestion: followUpQuestion,
        classification: classification.status,
        pedagogicalMove: classification.action,
        teachingPolicyAction: finalAction,
        reflectionAction,
        reasoning: classification.reasoning,
        isMastered: isMastered,
        socraticState: classification.status,
        position: position,
        masteryProgress: { current: newState.masteryScore, required: 3.5 },
        emotionalState: classification.emotionalState,
        supportLevel,
        confidence: classification.confidence,
        specificGaps: classification.specificGaps,
        visualAidNeeded,
        consecutiveWrong: consecutiveWrongNow,
        topic,
        steps: []
    };
}

/**
 * Fetch topic context from RAG service
 */
async function getTopicContext(course, topicId) {
    try {
        const pythonServiceUrl = process.env.PYTHON_RAG_SERVICE_URL;
        if (!pythonServiceUrl) return null;
        const url = `${pythonServiceUrl}/course/${encodeURIComponent(course)}/topic/${encodeURIComponent(topicId)}/context`;
        const response = await axios.get(url, { timeout: 20000 });
        return response.data;
    } catch (error) {
        log.error('TUTOR', `Error fetching topic context: ${error.message}`);
        return null;
    }
}

/**
 * Fetch Subtopic Teaching Notes (STN) from Redis via the RAG service.
 * STN is pre-generated from course material (marker-pdf Markdown) offline —
 * returns in ~5ms vs ~300ms for a live Qdrant vector search.
 * Falls back to live Qdrant context if no STN is cached.
 * 
 * Purpose: Provide instant access to pre-generated teaching context (offline-processed
 * from course materials) instead of waiting for live vector search. Falls back to
 * Qdrant search if STN cache miss.
 *
 * Performance: ~5ms cache hit vs ~300ms live search — critical for responsive tutoring.
 *
 * Return shape: Matches caller expectations with teaching_notes + qdrant_chunks + source flag.
 *
 * @param {string} course - Course identifier
 * @param {string} subtopicId - Subtopic identifier (preferred)
 * @param {string} topicId - Topic identifier (fallback if subtopicId missing)
 * @returns {Promise<Object|null>} Context object with teaching notes and chunks, or null
 */
async function getSubtopicContext(course, subtopicId, topicId) {
    const pythonServiceUrl = process.env.PYTHON_RAG_SERVICE_URL;
    if (!pythonServiceUrl) return null;

    // ── Fast path: STN cache ──────────────────────────────────────────────────
    const lookupId = subtopicId || topicId;
    if (course && lookupId) {
        try {
            const stnUrl = `${pythonServiceUrl}/stn/${encodeURIComponent(course)}/${encodeURIComponent(lookupId)}`;
            const stnResp = await axios.get(stnUrl, { timeout: 3000 });
            if (stnResp.data?.cached && stnResp.data?.data?.teaching_context) {
                const notes = stnResp.data.data;
                log.info('TUTOR', `STN cache HIT for ${course}/${lookupId}`);
                // Return in the same shape the caller expects (qdrant_chunks)
                return {
                    teaching_notes: notes,
                    qdrant_chunks: [{ text: notes.teaching_context }],
                    source: 'stn_cache'
                };
            }
        } catch (_) { /* fall through to live Qdrant */ }
    }

    // ── Slow path: live Qdrant search ────────────────────────────────────────
    if (!topicId) return null;
    try {
        const url = `${pythonServiceUrl}/course/${encodeURIComponent(course)}/topic/${encodeURIComponent(topicId)}/context`;
        log.info('TUTOR', `STN miss — falling back to Qdrant for ${topicId}`);
        const response = await axios.get(url, { timeout: 15000 });
        return response.data;
    } catch (error) {
        log.warn('TUTOR', `Context fetch failed: ${error.message}`);
        return null;
    }
}

/**
 * Progress & State Management
 */
async function saveUserProgress(userId, courseName, progress) {
    try {
        const User = require('../models/User');
        const update = {
            $set: {
                [`curriculumProgress.${courseName}`]: {
                    completedSubtopics: progress.completedSubtopics || [],
                    completedTopics: progress.completedTopics || [],
                    completedModules: progress.completedModules || [],
                    currentPosition: progress.currentPosition,
                    lastActiveDate: new Date()
                }
            }
        };
        await User.findByIdAndUpdate(userId, update, { upsert: true });
        return true;
    } catch (error) {
        log.error('TUTOR', `Error saving user progress: ${error.message}`);
        return false;
    }
}

async function loadUserProgress(userId, courseName) {
    try {
        const User = require('../models/User');
        const user = await User.findById(userId);
        return user?.curriculumProgress?.get(courseName) || null;
    } catch (error) {
        log.error('TUTOR', `Error loading user progress: ${error.message}`);
        return null;
    }
}

/**
 * Resolves the user's current position in the curriculum
 */
async function resolveCurrentPosition(courseName, completedSubtopics = [], completedTopics = [], targetModuleId = null) {
    log.info('TUTOR', `Resolving position for course: ${courseName}`);
    const structure = await getCurriculumStructure(courseName);
    
    if (!structure || !structure.modules || structure.modules.length === 0) {
        log.error('TUTOR', `No curriculum structure found for '${courseName}'. Cannot start structured tutor session.`);
        const err = new Error(`Curriculum not found for course "${courseName}". Please ask your admin to upload the syllabus CSV for this course.`);
        err.code = 'CURRICULUM_EMPTY';
        throw err;
    }

    // Also verify there is at least one module with topics
    const hasAnyTopics = structure.modules.some(m => m.topics && m.topics.length > 0);
    if (!hasAnyTopics) {
        log.error('TUTOR', `Curriculum for '${courseName}' has ${structure.modules.length} module(s) but no topics.`);
        const err = new Error(`Curriculum for "${courseName}" is incomplete — modules were found but no topics exist. Please re-upload the syllabus CSV.`);
        err.code = 'CURRICULUM_EMPTY';
        throw err;
    }

    // Find the first topic/subtopic that hasn't been completed
    for (const module of structure.modules) {
        // If a specific module target was provided, restrict search to that module only
        if (targetModuleId && module.id !== targetModuleId) {
            continue;
        }

        const topics = module.topics || [];
        for (const topic of topics) {
            if (!completedTopics.includes(topic.id)) {
                // Check prerequisites (subtopics)
                const prerequisites = topic.subtopics || topic.prerequisites || [];
                for (const sub of prerequisites) {
                    if (!completedSubtopics.includes(sub.id)) {
                        return {
                            moduleName: module.name,
                            topicName: topic.name,
                            subtopicName: sub.name,
                            teachingUnit: sub.name, // Added for chat.js compatibility
                            teachingUnitType: 'subtopic', // Added for chat.js compatibility
                            moduleId: module.id,
                            topicId: topic.id,
                            subtopicId: sub.id,
                            isLastInTopic: prerequisites.indexOf(sub) === prerequisites.length - 1,
                            isLastInModule: topics.indexOf(topic) === topics.length - 1 && prerequisites.indexOf(sub) === prerequisites.length - 1
                        };
                    }
                }
                
                // All subtopics are completed, so the topic is effectively complete.
                // We should skip teaching the topic umbrella itself and move to the next topic.
                continue;
            }
        }
    }

    if (targetModuleId) {
        return {
            moduleName: 'Module Complete',
            topicName: 'Module Complete',
            subtopicName: 'Summary',
            teachingUnit: 'Module Complete',
            teachingUnitType: 'finish',
            isLastInTopic: true,
            isLastInModule: true,
            isComplete: true
        };
    }

    return {
        moduleName: 'Success',
        topicName: 'Course Complete',
        subtopicName: 'Summary',
        teachingUnit: 'Course Complete',
        teachingUnitType: 'finish',
        isLastInTopic: true,
        isLastInModule: true,
        isComplete: true
    };
}

/**
 * Get the full curriculum structure from the RAG service, with Redis caching
 * Purpose: Fetch and cache the course's module/topic/subtopic hierarchy for navigation
 * and progress tracking. Cache TTL: 1 hour to balance freshness vs. performance.
 *
 * Cache strategy: Only cache non-empty structures with at least one topic to avoid
 * propagating incomplete curriculum data.
 *
 * @param {string} courseName - Course identifier
 * @returns {Promise<Object>} Curriculum structure with modules array, or empty fallback
 */
async function getCurriculumStructure(courseName) {
    try {
        const cacheKey = `curriculum:structure:${encodeURIComponent(courseName)}`;
        
        // 1. Check Redis Cache First
        try {
            const cachedData = await redisClient.get(cacheKey);
            if (cachedData) {
               // log.info('TUTOR', `Curriculum structure loaded from cache for ${courseName}`);
               return JSON.parse(cachedData); 
            }
        } catch(cacheErr) {
            log.warn('TUTOR', `Redis cache read error: ${cacheErr.message}`);
        }

        // 2. Fetch from Python Service if not cached
        // log.info('TUTOR', `Fetching curriculum structure for ${courseName} from Python backend...`);
        const pythonServiceUrl = process.env.PYTHON_RAG_SERVICE_URL;
        if (!pythonServiceUrl) return { modules: [] };
        const url = `${pythonServiceUrl}/curriculum/${encodeURIComponent(courseName)}/structure`;
        const response = await axios.get(url, { timeout: 30000 });
        const structure = response.data.curriculum || response.data;
        
        // 3. Save to Redis Cache only when data is non-empty (TTL: 1 hour)
        if (structure && structure.modules && structure.modules.length > 0) {
           const hasTopics = structure.modules.some(m => m.topics && m.topics.length > 0);
           if (hasTopics) {
               try {
                   await redisClient.setEx(cacheKey, 3600, JSON.stringify(structure));
               } catch (saveErr) {
                   log.warn('TUTOR', `Redis cache write error: ${saveErr.message}`);
               }
           } else {
               log.warn('TUTOR', `Curriculum for '${courseName}' has modules but no topics — skipping cache to prevent stale empty data.`);
           }
        } else {
            log.warn('TUTOR', `Curriculum for '${courseName}' returned empty modules — not caching.`);
        }
        
        return structure;
    } catch (error) {
        log.error('TUTOR', `Error fetching curriculum structure: ${error.message}`);
        return { modules: [] };
    }
}

function normalizeTutorSessionState(state = {}, sessionId = null) {
    const topic = state.topic || state.subtopicName || state.teachingUnit || state.moduleTitle || 'general';
    const learningPath = state.learningPath || {
        concept: topic,
        steps: fallbackLearningSteps(),
        currentStep: 0
    };
    const steps = Array.isArray(learningPath.steps) && learningPath.steps.length > 0
        ? learningPath.steps
        : fallbackLearningSteps();
    const currentStep = Number.isFinite(Number(learningPath.currentStep)) ? Number(learningPath.currentStep) : 0;

    return {
        ...state,
        sessionId: state.sessionId || sessionId,
        topic,
        cognitiveLevel: state.cognitiveLevel || COGNITIVE_LEVELS.L1_CONCEPT,
        consecutiveWrong: Number(state.consecutiveWrong || 0),
        hintsGiven: Number(state.hintsGiven || 0),
        masteryScore: Number(state.masteryScore || 0),
        turnCount: Number(state.turnCount || 0),
        learningPath: {
            concept: learningPath.concept || topic,
            steps,
            currentStep: Math.max(0, Math.min(currentStep, steps.length - 1))
        }
    };
}

/**
 * Session State Accessors
 */
async function getTutorSessionState(sessionId) {
    try {
        const stateStr = await redisClient.get(`tutor:session:${sessionId}`);
        if (!stateStr) return null;
        const normalized = normalizeTutorSessionState(JSON.parse(stateStr), sessionId);
        await redisClient.setEx(`tutor:session:${sessionId}`, TUTOR_STATE_TTL, JSON.stringify(normalized));
        return normalized;
    } catch (e) {
        log.error('TUTOR', `Redis get state error: ${e.message}`, e);
        return null;
    }
}

async function setTutorSessionState(sessionId, state) {
    try {
        const normalized = normalizeTutorSessionState(state, sessionId);
        await redisClient.setEx(`tutor:session:${sessionId}`, TUTOR_STATE_TTL, JSON.stringify(normalized));
        return true;
    } catch (e) {
        log.error('TUTOR', `Redis set state error: ${e.message}`, e);
        return false;
    }
}

async function clearTutorSessionState(sessionId) {
    try {
        await redisClient.del(`tutor:session:${sessionId}`);
        return true;
    } catch (e) {
        log.error('TUTOR', `Redis clear state error: ${e.message}`, e);
        return false;
    }
}

module.exports = {
    startSocraticSession,
    processTutorResponse,
    clearTutorSessionState,
    getTutorSessionState,
    setTutorSessionState,
    saveUserProgress,
    loadUserProgress,
    resolveCurrentPosition,
    getSubtopicContext,
    getTopicContext,
    // Add placeholders to prevent chat.js crashes
    getCurriculumStructure,
    buildInitialLearningPath,
    getNextTopic: (course, currentTopicId) => {
        // This is used for simple jump-ahead, we can still use the Python service here
        return null; // Placeholder for now, resolveCurrentPosition is the primary driver
    },
    advanceToNextSubtopic: async (course, current, completedS, completedT) => {
        const cSub = [...(completedS || [])];
        const cTop = [...(completedT || [])];

        // Mark current teaching unit as completed
        if (current.subtopicId && !cSub.includes(current.subtopicId)) {
            cSub.push(current.subtopicId);
        }

        const isTopicCompleted = current.subtopicName === current.topicName || current.isLastInTopic;
        if (isTopicCompleted && current.topicId && !cTop.includes(current.topicId)) {
            cTop.push(current.topicId);
        }

        const nextPosition = await resolveCurrentPosition(course, cSub, cTop);
        return {
            completedSubtopics: cSub,
            completedTopics: cTop,
            topicJustCompleted: isTopicCompleted,
            moduleJustCompleted: current.isLastInModule,
            nextPosition: nextPosition,
            topicCompletedName: isTopicCompleted ? current.topicName : null,
            moduleCompletedName: current.isLastInModule ? current.moduleName : null
        };
    },
    findCurrentTeachingUnit: (pos) => {
        return { 
            name: pos.subtopicName || pos.topicName || 'General', 
            type: pos.subtopicId === pos.topicId ? 'topic' : 'subtopic' 
        };
    },
    validateSessionState: () => true,
    resumeOrStartSession,
    getPrecomputedContent,
    pickPrecomputedQuestion,
    COGNITIVE_LEVELS,
    PEDAGOGICAL_MOVES,
    SOCRATIC_STATES,
    // Emotional state & support (ported from T1-6)
    EMOTIONAL_STATES,
    CONFIDENCE_LEVELS,
    UNDERSTANDING_LEVELS,
    SUPPORT_LEVELS,
    assessStudentResponse,
    determineSupportLevel,
    generateWithFallback
};
