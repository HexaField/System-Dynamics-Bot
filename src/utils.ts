import axios from 'axios'
import * as fs from 'fs'

require('dotenv').config()

export const RED = '\u001b[31m'
export const RESET = '\u001b[0m'

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434'

// getEmbedding now uses Ollama exclusively - no fallbacks
export async function getEmbedding(text: string, model?: string) {
  const clean = text.replace(/\n/g, ' ')
  const embedModel = model || process.env.OLLAMA_EMBEDDING_MODEL || 'bge-m3:latest'
  const payload = { model: embedModel, input: clean }
  
  try {
    const resp = await axios.post(`${OLLAMA_URL}/api/embed`, payload, { timeout: 60000 })
    const data = resp.data
    
    // Ollama /api/embed response format: { "embeddings": [[...]] }
    if (data && Array.isArray(data.embeddings) && data.embeddings[0] && Array.isArray(data.embeddings[0])) {
      return data.embeddings[0]
    }
    
    // No fallbacks - fail immediately if embeddings not available
    throw new Error(
      `Embeddings API returned unexpected format for model ${embedModel}. Response: ${JSON.stringify(data)}. Ensure Ollama is running at ${OLLAMA_URL} and the model supports embeddings.`
    )
  } catch (error: any) {
    if (error.response) {
      throw new Error(
        `Ollama embeddings API error (${error.response.status}): ${JSON.stringify(error.response.data)}. Ensure model ${embedModel} is installed (run: ollama pull ${embedModel}) and Ollama is running at ${OLLAMA_URL}.`
      )
    }
    throw error
  }
}

export function cosineSimilarity(a: number[], b: number[]) {
  const dot = a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0)
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0))
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0))
  if (magA === 0 || magB === 0) throw new Error('Zero magnitude vector')
  return dot / (magA * magB)
}

export function extractNumbers(input: string) {
  const m = input.match(/\d+/g)
  return m || []
}

export function xmileName(displayName: string) {
  return displayName.split(/\s+/).join('_')
}

export function cleanSymbol(symbol: string) {
  return symbol.replace(/[()]/g, '')
}

export function loadJson(text: string) {
  if (!text || typeof text !== 'string') return null
  // Try to extract codefence JSON first
  const m = text.match(/```json\n([\s\S]*?)```/i)
  if (m && m[1]) {
    try {
      return JSON.parse(m[1])
    } catch (e) {
      // fall through
    }
  }
  // Try plain JSON parse
  try {
    return JSON.parse(text)
  } catch (e) {
    // Try to find the first { ... } block
    const first = text.indexOf('{')
    const last = text.lastIndexOf('}')
    if (first !== -1 && last !== -1 && last > first) {
      const sub = text.substring(first, last + 1)
      try {
        return JSON.parse(sub)
      } catch (e2) {
        return null
      }
    }
    return null
  }
}

export function cleanUp(text: string) {
  // Extract numbered list items
  const pattern = /\d+\.[^\n]*(?:\n(?!\d+\.).*)*/g
  const items = text.match(pattern) || []
  return items.map((i) => i.replace(/\n\s*/g, ' ').trim()).join('\n')
}

export function saveFile(path: string, content: string) {
  fs.writeFileSync(path, content, { encoding: 'utf8' })
}
