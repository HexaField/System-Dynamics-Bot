import { beforeEach, describe, expect, it, vi } from 'vitest'
import { generateCausalLoopDiagram } from '../src/sage'
import * as utils from '../dist/utils'

describe('GreatSage.think', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('writes relationships, xmile and dot when flags set', async () => {
    const sampleText = 'When death rate goes up, population decreases.'

    vi.spyOn(utils, 'getEmbedding').mockResolvedValue(new Array(8).fill(1))

    const response1 = JSON.stringify({
      '1': {
        reasoning: 'Because higher death reduces population',
        'causal relationship': 'death rate -->(-) population',
        'relevant text': 'When death rate goes up, population decreases.'
      }
    })
    vi.spyOn(utils, 'getCompletionFromMessages').mockResolvedValueOnce(response1).mockResolvedValueOnce('{}')

    const res = await generateCausalLoopDiagram({
      verbose: false,
      diagram: true,
      xmile: true,
      threshold: 0.85,
      question: sampleText
    })

    expect(res).toBeDefined()
    expect(res.response).toContain('1. death rate -->(-) population')
    expect(res.lines).toContain('death rate -->(-) population')
    expect(res.xmile).toContain('<xmile')
    expect(res.dot).toContain('digraph')
  })
})
