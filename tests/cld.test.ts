import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('CLD.generateCausalRelationships', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('parses model JSON response and returns numbered relationships', async () => {
    const sampleText = 'When death rate goes up, population decreases.'

    // Require the built utils so we can mock functions used by the built CLD
    const utils = require('../dist/utils')

    // Mock embedding to deterministic small vectors
    // @ts-ignore
    vi.spyOn(utils, 'getEmbedding').mockImplementation(async (text: string) => {
      const len = text.length
      return new Array(8).fill(0).map((_, i) => (i === 0 ? len : 1))
    })

    // Mock getCompletionFromMessages to return a JSON response string
    const response1 = JSON.stringify({
      '1': {
        reasoning: 'Because higher death reduces population',
        'causal relationship': 'death rate -->(-) population',
        'relevant text': 'When death rate goes up, population decreases.'
      }
    })

    vi.spyOn(utils, 'getCompletionFromMessages').mockResolvedValueOnce(response1).mockResolvedValueOnce('{}')

    // Now require the built CLD after mocks are in place so the built module picks up the mocked utils
    const CLD = require('../dist/cld').default

    const cld = new CLD(sampleText, 0.85, false)
    const output = await cld.generateCausalRelationships()
    expect(output).toContain('1. death rate -->(-) population')
  })
})
