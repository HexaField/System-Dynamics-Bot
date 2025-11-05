import { expect, test } from 'vitest'
import { generateCausalLoopDiagram } from '../dist/sage'

// This e2e test runs against a local Ollama HTTP API. It is skipped unless
// you explicitly enable it by setting the following environment variables:
//
//   USE_OLLAMA=1
//   OLLAMA_URL=http://localhost:11434
//   RUN_OLLAMA_E2E=1
//
// Additionally, ensure Ollama has a model that supports chat completions and
// embeddings. For example, if you use `gpt-oss:20b` or another model that
// provides both endpoints, pull it locally with `ollama pull <model>`.
//
// The test purposefully avoids mocking and exercises the actual network calls.

const shouldRun = Boolean(process.env.RUN_OLLAMA_E2E) && Boolean(process.env.USE_OLLAMA)

// Increase timeout because real model calls can take longer
;(shouldRun ? test : test.skip)(
  'end-to-end run using Ollama (real LLM)',
  async () => {
    // Preflight determinism check: ensure the configured Ollama model honors seed + deterministic params.
    const utils = require('../dist/utils')
    const checkSeed = Number(process.env.SEED) || 42
    const probeMsg = [{ role: 'user', content: `Return exactly this JSON: {"seed": ${checkSeed}} and nothing else.` }]
    const a = await utils.getCompletionFromMessages(probeMsg, process.env.OLLAMA_CHAT_MODEL, false, 0, 1, checkSeed)
    const b = await utils.getCompletionFromMessages(probeMsg, process.env.OLLAMA_CHAT_MODEL, false, 0, 1, checkSeed)
    if (a !== b) {
      throw new Error(
        `Preflight determinism check failed: model responses differed. Ensure your Ollama model supports deterministic seeds and set OLLAMA_CHAT_MODEL accordingly. Got:\nA=${a}\nB=${b}`
      )
    }

    // Preflight schema compliance: ensure model returns the required JSON schema for a simple example
    const schemaProbe = [{ role: 'user', content: `Extract causal relationships from: "When X increases, Y decreases." Return ONLY valid JSON following the schema: {"causalRelationships": [{"cause":"<text>","effect":"<text>","direction":"increase|decrease|positive|negative","reasoning":"<text>","relevant":"<text>"}] }` }]
    const schemaOut = await utils.getCompletionFromMessages(schemaProbe, process.env.OLLAMA_CHAT_MODEL, false, 0, 1, checkSeed)
    const parsedSchema = utils.loadJson(schemaOut)
    if (!parsedSchema || !Array.isArray(parsedSchema.causalRelationships)) {
      throw new Error(
        `Preflight schema check failed: model did not return the expected JSON schema. Received: ${schemaOut}. Please use an Ollama model that supports structured JSON outputs and set OLLAMA_CHAT_MODEL accordingly.`
      )
    }

    // Larger prompt to exercise CLD parsing
    const largePrompt = `Engineers compare the work remaining to be done against the time remaining before the deadline. The larger the gap, the more Schedule Pressure they feel. When schedule pressure builds up, engineers can work overtime which increases completion rate but also increases fatigue, which lowers productivity.`

    const result = await generateCausalLoopDiagram({
      verbose: false,
      diagram: false,
      xmile: false,
      threshold: 0.85,
      question: largePrompt
    })
    console.log(result)

    // Expect a non-empty response and at least one extracted relationship line
    expect(result).toBeDefined()
    expect(typeof result.response).toBe('string')
    expect(result.response.length).toBeGreaterThan(0)
    expect(Array.isArray(result.lines)).toBe(true)
    expect(result.lines.length).toBeGreaterThan(0)

    // Normalize lines for robust matching
    const norm = (s: string) =>
      s
        .toLowerCase()
        .replace(/["'()\.,]/g, '')
        .replace(/\s+/g, ' ')
        .trim()
    const parsed = result.lines.map((l: string) => {
      const parts = l.split('-->')
      return {
        raw: l,
        left: parts[0] ? norm(parts[0]) : '',
        right: parts[1] ? norm(parts[1]) : ''
      }
    })

    // Expected causal pairs implied by the prompt. Use keyword sets for fuzzy matching.
    const expectedKeywordPairs: Array<[string[], string[]]> = [
      [['gap between work remaining', 'gap', 'work remaining', 'work_remaining', 'workremaining'], ['schedule pressure', 'schedule_pressure', 'schedulepressure']],
      [['schedule pressure', 'schedule_pressure'], ['overtime', 'working overtime', 'work overtime', 'engineers work overtime']],
      [['overtime', 'working overtime'], ['completion rate', 'completion_rate', 'increased completion rate', 'completionrate']],
      [['overtime', 'working overtime'], ['fatigue']],
      [['fatigue'], ['productivity', 'lowered productivity', 'productivity']]
    ]

    let found = 0
    for (const [causeKeys, effectKeys] of expectedKeywordPairs) {
      const ok = parsed.some((p: { left: string; right: string }) => {
        const left = p.left
        const right = p.right
        const causeOk = causeKeys.some((k) => left.includes(k))
        const effectOk = effectKeys.some((k) => right.includes(k))
        return causeOk && effectOk
      })
      if (ok) found++
    }

    // Require that the model produced most of the expected causal links (tolerate 1 missing)
    expect(found).toBeGreaterThanOrEqual(expectedKeywordPairs.length - 1)

    console.log('test finished')
  },
  120000
)
