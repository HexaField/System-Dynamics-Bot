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
    // Larger prompt to exercise CLD parsing
    const largePrompt = `Engineers compare the work remaining to be done against the time remaining before the deadline. The larger the gap, the more Schedule Pressure they feel. When schedule pressure builds up, engineers can work overtime which increases completion rate but also increases fatigue, which lowers productivity.`

    const result = await generateCausalLoopDiagram({
      verbose: false,
      diagram: false,
      write_relationships: false,
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
  },
  120000
)
