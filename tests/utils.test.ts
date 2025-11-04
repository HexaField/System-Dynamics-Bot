import { describe, expect, it } from 'vitest'
import { cosineSimilarity, loadJson } from '../dist/utils'

describe('utils', () => {
  it('cosineSimilarity returns 1 for identical vectors', () => {
    const a = [1, 2, 3]
    const b = [1, 2, 3]
    const sim = cosineSimilarity(a, b)
    expect(sim).toBeCloseTo(1, 6)
  })

  it('cosineSimilarity handles orthogonal vectors', () => {
    const a = [1, 0]
    const b = [0, 1]
    const sim = cosineSimilarity(a, b)
    expect(sim).toBeCloseTo(0, 6)
  })

  it('loadJson parses JSON string', () => {
    const text = '{"a":1}'
    const obj = loadJson(text as any)
    expect(obj).toEqual({ a: 1 })
  })
})
