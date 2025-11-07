import { callLLM } from './llm'
import { cosineSimilarity, getEmbedding, loadJson } from './utils'

function simpleSentenceSplit(text: string): string[] {
  // very simple sentence splitter
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

const systemPrompt = `You are a System Dynamics Professional Modeler.
Users will give text, and it is your job to extract causal relationships from that text.
You will conduct a multi-step process:

1. Identify variables (entities) that participate in cause-effect relationships. Name variables concisely (no more than 2 words), avoid sentiment (neutral names), and minimize the number of unique variables by preferring canonical/shorter names when synonyms appear.

2. Represent each causal relationship as an object with subject, predicate, and object. Use ONLY these predicate values:
   - positive: subject and object move in the same direction (↑subject -> ↑object, ↓subject -> ↓object)
   - negative: subject and object move in opposite directions (↑subject -> ↓object, ↓subject -> ↑object)
   - increase: subject causes object to increase (directional effect)
   - decrease: subject causes object to decrease (directional effect)

3. When three variables are related in a sentence, ensure the relation between the second and third variable is correct. For example, in "X inhibits Y, leading to less Z", Y and Z have a positive relationship.

4. If there are no causal relationships in the provided text, return an empty array for causalRelationships.

OUTPUT FORMAT (return ONLY JSON, nothing else):
{
  "causalRelationships": [
    {
      "subject": "<variable>",
      "predicate": "increase|decrease|positive|negative",
      "object": "<variable>"
    }
  ]
}

Example 1 input:
"when death rate goes up, population decreases"

Example 1 JSON response:
{
  "causalRelationships": [
    {
      "subject": "death rate",
      "predicate": "negative",
      "object": "population"
    }
  ]
}

Example 2 input:
"lower death rate increases population"

Example 2 JSON response:
{
  "causalRelationships": [
    {
      "subject": "death rate",
      "predicate": "negative",
      "object": "population"
    }
  ]
}

Example 3 input:
"The engineers compare the work remaining to be done against the time remaining before the deadline. The larger the gap, the more Schedule Pressure they feel. When schedule pressure builds up, engineers can work overtime. Overtime raises completion rate but also increases fatigue, which lowers productivity."

Example 3 JSON response (truncated):
{
  "causalRelationships": [
    {"subject": "work remaining", "predicate": "positive", "object": "schedule pressure"},
    {"subject": "time remaining", "predicate": "negative", "object": "schedule pressure"},
    {"subject": "schedule pressure", "predicate": "increase", "object": "overtime"},
    {"subject": "overtime", "predicate": "increase", "object": "completion rate"},
    {"subject": "overtime", "predicate": "increase", "object": "fatigue"},
    {"subject": "fatigue", "predicate": "decrease", "object": "productivity"}
  ]
}

Example 4 input (no causal relationships):
"[Text with no causal relationships]"

Example 4 JSON response:
{ "causalRelationships": [] }

Return ONLY the JSON in the exact schema shown above.`

export class CLD {
  question: string
  threshold: number
  verbose: boolean
  sentences: string[]
  embeddings: number[][]
  llmModel?: string
  embeddingModel?: string
  temperature: number
  top_p: number
  seed: number | null

  constructor(
    question: string,
    threshold = 0.85,
    verbose = false,
    llmModel?: string,
    embeddingModel?: string,
    temperature = 0,
    top_p = 1,
    seed: number | null = Number(process.env.SEED) || 42
  ) {
    this.question = question
    this.threshold = threshold
    this.verbose = verbose
    this.sentences = simpleSentenceSplit(question)
    this.embeddings = []
    this.llmModel = llmModel
    this.embeddingModel = embeddingModel
    this.temperature = temperature
    this.top_p = top_p
    this.seed = seed
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
    const humanize = (s: string) =>
      s
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/_/g, ' ')
        .toLowerCase()
        .trim()

    // Single minimal LLM call: expect JSON { causalRelationships: [ { subject, predicate, object } ] }
    const response = await callLLM(systemPrompt, this.question, 'opencode', this.llmModel)
    const parsed = loadJson(response.data!)
    if (!parsed || !Array.isArray(parsed.causalRelationships)) {
      throw new Error('Assistant did not return causalRelationships in the expected JSON schema')
    }

    const outLines: string[] = []
    let idxOut = 1
    for (const entry of parsed.causalRelationships) {
      const subject = entry.subject || entry.cause || entry.variable1 || ''
      const object = entry.object || entry.effect || entry.variable2 || ''
      const predicate = (entry.predicate || entry.direction || entry.relationship || '').toLowerCase()
      const symbol =
        predicate.includes('increase') || predicate.includes('positive')
          ? '(+)'
          : predicate.includes('decrease') || predicate.includes('negative')
            ? '(-)'
            : ''
      const ssub = humanize(subject)
      const oobj = humanize(object)
      if (ssub && oobj) {
        outLines.push(`${idxOut}. ${ssub} -->${symbol} ${oobj}`)
        idxOut++
      }
    }
    return outLines.join('\n')
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
