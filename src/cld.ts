import { cosineSimilarity, getCompletionFromMessages, getEmbedding, loadJson } from './utils'

function simpleSentenceSplit(text: string): string[] {
  // very simple sentence splitter
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

export class CLD {
  question: string
  threshold: number
  verbose: boolean
  sentences: string[]
  embeddings: number[][]
  llmModel?: string
  embeddingModel?: string

  constructor(question: string, threshold = 0.85, verbose = false, llmModel?: string, embeddingModel?: string) {
    this.question = question
    this.threshold = threshold
    this.verbose = verbose
    this.sentences = simpleSentenceSplit(question)
    this.embeddings = []
    this.llmModel = llmModel
    this.embeddingModel = embeddingModel
  }

  async initEmbeddings() {
    const jobs = this.sentences.map((s) => getEmbedding(s, this.embeddingModel))
    this.embeddings = (await Promise.all(jobs as any)) as number[][]
  }

  async getLine(query: string) {
    if (this.embeddings.length === 0) await this.initEmbeddings()
    const qEmb = await getEmbedding(query, this.embeddingModel)
    let best = 0
    let idx = 0
    for (let i = 0; i < this.embeddings.length; i++) {
      const sim = cosineSimilarity(qEmb as number[], this.embeddings[i])
      if (sim > best) {
        best = sim
        idx = i
      }
    }
    return this.sentences[idx]
  }

  async generateCausalRelationships() {
    const systemPrompt = `You are a System Dynamics Professional Modeler.\nUsers will give text, and it is your job to generate causal relationships from that text.\nFollow the instructions and return only JSON as in examples.`
    const context: any[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: this.question }
    ]
    const response1 = await getCompletionFromMessages(context, this.llmModel)
    let parsed1 = loadJson(response1 as string)
    if (!parsed1) {
      throw new Error('Input text did not have any causal relationships')
    }
    // Normalize array responses into the expected keyed object format
    if (Array.isArray(parsed1)) {
      const obj: any = {}
      parsed1.forEach((entry: any, idx: number) => {
        const cause = entry.cause || entry.variable1 || ''
        const effect = entry.effect || entry.variable2 || ''
        const rel = (entry.relationship || '').toLowerCase()
        const symbol =
          rel.includes('increase') || rel.includes('positive')
            ? '(+)'
            : rel.includes('decrease') || rel.includes('negative')
              ? '(-)'
              : ''
        const causal = `${cause} -->${symbol} ${effect}`.trim()
        obj[(idx + 1).toString()] = {
          reasoning: entry.reasoning || '',
          'causal relationship': causal,
          'relevant text': entry.relevant || this.question
        }
      })
      parsed1 = obj
    }
    // Normalize object with array under common keys, e.g., { causalRelationships: [...] }
    if (
      parsed1 &&
      !Array.isArray(parsed1) &&
      parsed1.causalRelationships &&
      Array.isArray(parsed1.causalRelationships)
    ) {
      const arr = parsed1.causalRelationships
      const obj: any = {}
      arr.forEach((entry: any, idx: number) => {
        const cause = entry.cause || entry.variable1 || ''
        const effect = entry.effect || entry.variable2 || ''
        const rel = (entry.direction || entry.relationship || '').toLowerCase()
        const symbol =
          rel.includes('increase') || rel.includes('positive')
            ? '(+)'
            : rel.includes('decrease') || rel.includes('negative')
              ? '(-)'
              : ''
        const causal = `${cause} -->${symbol} ${effect}`.trim()
        obj[(idx + 1).toString()] = {
          reasoning: entry.reasoning || '',
          'causal relationship': causal,
          'relevant text': entry.relevant || this.question
        }
      })
      parsed1 = obj
    }
    // Generic: find any top-level array of relationship-like objects (keys: cause/effect/sign/direction)
    if (parsed1 && !Array.isArray(parsed1)) {
      const keys = Object.keys(parsed1)
      for (const k of keys) {
        const val = (parsed1 as any)[k]
        if (
          Array.isArray(val) &&
          val.length > 0 &&
          (val[0].cause || val[0].effect || val[0].sign || val[0].direction)
        ) {
          const arr = val
          const obj: any = {}
          arr.forEach((entry: any, idx: number) => {
            const cause = entry.cause || entry.variable1 || ''
            const effect = entry.effect || entry.variable2 || ''
            const rel = (entry.sign || entry.direction || entry.relationship || '').toLowerCase()
            const symbol =
              rel.includes('increase') || rel.includes('positive')
                ? '(+)'
                : rel.includes('decrease') || rel.includes('negative')
                  ? '(-)'
                  : ''
            const causal = `${cause} -->${symbol} ${effect}`.trim()
            obj[(idx + 1).toString()] = {
              reasoning: entry.reasoning || '',
              'causal relationship': causal,
              'relevant text': entry.relevant || this.question
            }
          })
          parsed1 = obj
          break
        }
      }
    }
    // secondary check for loops
    const query = `Find out if there are any possibilities of forming closed loops that are implied in the text. If yes, then close the loops by adding the extra relationships and provide them in a JSON format please.`
    context.push({ role: 'user', content: query })
    const response2 = await getCompletionFromMessages(context, this.llmModel)
    const parsed2 = loadJson(response2 as string)
    const merged = { ...(parsed1 as any), ...(parsed2 as any) }

    // Build lines: tuple of (relationship, reasoning, relevant text)
    const lines: Array<[string, string, string]> = []
    for (const k of Object.keys(merged)) {
      const entry = (merged as any)[k]
      const rel = entry['causal relationship'] || entry['relationship'] || ''
      const reason = entry['reasoning'] || ''
      const relevant = entry['relevant text'] || entry['relevant_text'] || entry['relevantText'] || this.question
      const snippet = await this.getLine(relevant)
      lines.push([rel, reason, snippet])
    }

    // simple corrected response: list each relationship numbered
    const corrected = lines.map((l, i) => `${i + 1}. ${l[0]}`).join('\n')
    return corrected
  }

  extractVariables(relationship: string) {
    const parts = relationship.split('-->')
    if (parts.length < 2) return ['', '', '']
    let var1 = parts[0].trim().toLowerCase()
    let right = parts[1]
    let symbol = ''
    let var2 = right
      .replace(/\(\+\)|\(-\)/g, (s) => {
        symbol = s
        return ''
      })
      .trim()
      .toLowerCase()
    var1 = var1.replace(/[!.,;:]/g, '')
    var2 = var2.replace(/[!.,;:]/g, '')
    return [var1, var2, symbol]
  }
}

export default CLD
