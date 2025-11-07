import { expect, test } from 'vitest'
import { generateCausalLoopDiagram } from '../src/sage'

//   OLLAMA_URL=http://localhost:11434

// Increase timeout because real model calls can take longer
test('end-to-end run using Ollama (real LLM)', async () => {
  // Run the full CLD extraction 5 times in a row and require validity each time
  for (let i = 0; i < 5; i++) {
    // Preflight determinism check: ensure the configured Ollama model honors seed + deterministic params.
    const checkSeed = Number(process.env.SEED) || 42
    // Larger prompt to exercise CLD parsing
    const largePrompt = `Engineers compare the work remaining to be done against the time remaining before the deadline. The larger the gap, the more Schedule Pressure they feel. When schedule pressure builds up, engineers can work overtime which increases completion rate but also increases fatigue, which lowers productivity.`

    // Expected causal pairs implied by the prompt. Use keyword sets for fuzzy matching.
    const expectedKeywordPairs: Array<[string[], string[]]> = [
      [
        ['gap between work remaining', 'gap', 'work remaining', 'work_remaining', 'workremaining'],
        ['schedule pressure', 'schedule_pressure', 'schedulepressure']
      ],
      [
        ['schedule pressure', 'schedule_pressure'],
        ['overtime', 'working overtime', 'work overtime', 'engineers work overtime']
      ],
      [
        ['overtime', 'working overtime'],
        ['completion rate', 'completion_rate', 'increased completion rate', 'completionrate']
      ],
      [['overtime', 'working overtime'], ['fatigue']],
      [['fatigue'], ['productivity', 'lowered productivity', 'productivity']]
    ]

    const norm = (s: string) =>
      s
        .toLowerCase()
        .replace(/["'()\.,]/g, '')
        .replace(/\s+/g, ' ')
        .trim()

    const result = await generateCausalLoopDiagram({
      verbose: true,
      diagram: false,
      xmile: false,
      threshold: 0.85,
      llmModel: 'github-copilot/gpt-5-mini',
      question: largePrompt,
      seed: checkSeed,
      temperature: 0,
      top_p: 1
    })
    console.log(result)

    // Expect a non-empty response and structured lines
    expect(result).toBeDefined()
    expect(typeof result.response).toBe('string')
    expect(result.response.length).toBeGreaterThan(0)
    expect(Array.isArray(result.lines)).toBe(true)
    expect(result.lines.length).toBeGreaterThan(0)

    const parsed = result.lines.map((l: string) => {
      const parts = l.split('-->')
      return {
        raw: l,
        left: parts[0] ? norm(parts[0]) : '',
        right: parts[1] ? norm(parts[1]) : '',
        hasValence: /\(\+\)|\(-\)/.test(l)
      }
    })

    // Ensure every produced relationship includes valence (+) or (-)
    for (const p of parsed) {
      expect(p.hasValence).toBe(true)
    }

    let found = 0
    for (const [causeKeys, effectKeys] of expectedKeywordPairs) {
      const ok = parsed.some((p: { left: string; right: string; hasValence: boolean }) => {
        const left = p.left
        const right = p.right
        const causeOk = causeKeys.some((k) => left.includes(k))
        const effectOk = effectKeys.some((k) => right.includes(k))
        return causeOk && effectOk && p.hasValence
      })
      if (ok) found++
    }

    // Require that the model produced most of the expected causal links (tolerate 1 missing) for each run
    expect(found).toBeGreaterThanOrEqual(expectedKeywordPairs.length - 1)
  }

  console.log('test finished')
}, 600_000)
