import { beforeEach, describe, expect, it, vi } from 'vitest'
import CLD from '../dist/cld'
import * as utils from '../dist/utils'

describe('CLD.generateCausalRelationships', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('parses model JSON response and returns numbered relationships', async () => {
    const sampleText = 'When death rate goes up, population decreases.'

    // Mock embedding to deterministic small vectors
    vi.spyOn(utils, 'getEmbedding').mockImplementation(async (text: string) => {
      // return different embeddings depending on length to allow getLine to pick a sentence
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

    const cld = new CLD(sampleText, 0.85, false)
    const output = await cld.generateCausalRelationships()
    expect(output).toContain('1. death rate -->(-) population')
  })
})
