import axios from 'axios'
import * as fs from 'fs'
import * as OpenAI from 'openai'

require('dotenv').config()

export const RED = '\u001b[31m'
export const RESET = '\u001b[0m'

const USE_OLLAMA = Boolean(process.env.USE_OLLAMA) || Boolean(process.env.OLLAMA_URL)
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434'

const openaiKey = process.env.OPENAI_API_KEY || ''
const openaiClient = openaiKey ? new OpenAI.OpenAI({ apiKey: openaiKey }) : null

export async function getCompletionFromMessages(
  messages: any[],
  model?: string,
  responseFormat = false,
  temperature = 0
) {
  if (USE_OLLAMA) {
    // prefer an explicit model argument first, then env var fallback
    const chatModel = model || process.env.OLLAMA_CHAT_MODEL || 'gpt-oss:20b'
    const payload = { model: chatModel, messages, temperature }
    let resp
    try {
      resp = await axios.post(`${OLLAMA_URL}/api/chat`, payload, {
        timeout: 120000
      })
    } catch (err: any) {
      // If Ollama instance doesn't expose /api/chat for this model, try /api/generate as a fallback
      if (err && err.response && err.response.status === 404) {
        resp = await axios.post(`${OLLAMA_URL}/api/generate`, payload, {
          timeout: 120000
        })
      } else {
        throw err
      }
    }
    const data = resp.data
    // Ollama often streams NDJSON. Handle string streaming output by parsing lines.
    if (typeof data === 'string') {
      const lines = data
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean)
      const parts: string[] = []
      for (const line of lines) {
        try {
          const obj = JSON.parse(line)
          if (obj && obj.message && typeof obj.message.content === 'string') {
            parts.push(obj.message.content)
          } else if (obj && obj.output && typeof obj.output === 'string') {
            parts.push(obj.output)
          }
        } catch (e) {
          // ignore partial JSON parse errors
        }
      }
      const joined = parts.join('')
      if (joined) return joined
    }
    // Handle JSON-like shapes
    if (data && data.choices && Array.isArray(data.choices)) {
      const first = data.choices[0]
      if (first && first.message && first.message.content) return first.message.content
    }
    if (typeof data === 'object') return JSON.stringify(data)
    if (typeof data === 'string') return data
    return ''
  } else {
    if (!openaiClient) throw new Error('OpenAI API key not set')
    // use chat.completions.create style wrapper (openai npm v4 provides chat.completions.create)
    const chosen = model || 'gpt-4-1106-preview'
    const response = await openaiClient.chat.completions.create({
      model: chosen,
      messages,
      temperature
    })
    return response.choices[0].message.content
  }
}

export async function getEmbedding(text: string, model?: string) {
  const clean = text.replace(/\n/g, ' ')
  if (USE_OLLAMA) {
    // prefer explicit model argument first, then env var, then sensible default for Ollama
    const embedModel = model || process.env.OLLAMA_EMBEDDING_MODEL || 'bge-m3:latest'
    const payload = { model: embedModel, input: [clean] }
    const resp = await axios.post(`${OLLAMA_URL}/api/embeddings`, payload, { timeout: 60000 })
    const data = resp.data
    // Common Ollama shapes:
    // 1) { data: [{ embedding: [...] }] }
    if (
      data &&
      data.data &&
      Array.isArray(data.data) &&
      data.data[0] &&
      Array.isArray(data.data[0].embedding) &&
      data.data[0].embedding.length
    ) {
      return data.data[0].embedding
    }
    // 2) { embedding: [...] }
    if (data && Array.isArray(data.embedding) && data.embedding.length) {
      return data.embedding
    }
    // If Ollama returned empty embeddings (some setups), fall back to deterministic local embedding
    return hashEmbedding(clean)
  } else {
    if (!openaiClient) throw new Error('OpenAI API key not set')
    const chosen = model || 'text-embedding-3-small'
    const resp = await openaiClient.embeddings.create({ model: chosen, input: [clean] })
    return resp.data[0].embedding
  }
}

function hashEmbedding(text: string, dim = 1536) {
  // Deterministic pseudo-embedding so downstream code can compute similarities.
  let h = 2166136261 >>> 0
  for (let i = 0; i < text.length; i++) {
    h = Math.imul(h ^ text.charCodeAt(i), 16777619) >>> 0
  }
  const vec: number[] = new Array(dim)
  let seed = h
  for (let i = 0; i < dim; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0
    vec[i] = ((seed % 1000) - 500) / 500
  }
  return vec
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
