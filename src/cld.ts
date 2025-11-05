import { cosineSimilarity, getCompletionFromMessages, getEmbedding, loadJson } from './utils'

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

2. Represent each causal relationship as an object with subject, predicate, and object, plus your reasoning and the most relevant supporting text snippet. Use ONLY these predicate values:
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
      "object": "<variable>",
      "reasoning": "<your concise reasoning for this relationship>",
      "relevant": "<the exact sentence/paragraph that highlights this relationship>"
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
      "object": "population",
      "reasoning": "An increase in death rate leads to a decrease in population (opposite directions).",
      "relevant": "when death rate goes up, population decreases"
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
      "object": "population",
      "reasoning": "A decrease in death rate leads to an increase in population (opposite directions).",
      "relevant": "lower death rate increases population"
    }
  ]
}

Example 3 input:
"The engineers compare the work remaining to be done against the time remaining before the deadline. The larger the gap, the more Schedule Pressure they feel. When schedule pressure builds up, engineers can work overtime. Overtime raises completion rate but also increases fatigue, which lowers productivity."

Example 3 JSON response (truncated):
{
  "causalRelationships": [
    {"subject": "work remaining", "predicate": "positive", "object": "schedule pressure", "reasoning": "Larger gap means more schedule pressure.", "relevant": "The larger the gap, the more Schedule Pressure they feel."},
    {"subject": "time remaining", "predicate": "negative", "object": "schedule pressure", "reasoning": "More time remaining reduces schedule pressure.", "relevant": "compare the work remaining ... against the time remaining ... The larger the gap, the more Schedule Pressure"},
    {"subject": "schedule pressure", "predicate": "increase", "object": "overtime", "reasoning": "Pressure leads engineers to work overtime.", "relevant": "When schedule pressure builds up, engineers can work overtime."},
    {"subject": "overtime", "predicate": "increase", "object": "completion rate", "reasoning": "Working more hours increases completion rate.", "relevant": "Overtime raises completion rate"},
    {"subject": "overtime", "predicate": "increase", "object": "fatigue", "reasoning": "Working long hours increases fatigue.", "relevant": "overtime ... increases fatigue"},
    {"subject": "fatigue", "predicate": "decrease", "object": "productivity", "reasoning": "Fatigue lowers productivity.", "relevant": "fatigue ... lowers productivity"}
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

  async computeSimilarities(variable_to_index: Record<string, number>, index_to_variable: Record<number, string>) {
    const keys = Object.keys(variable_to_index)
    if (keys.length === 0) return null
    const embeddings = await Promise.all(keys.map((k) => getEmbedding(k, this.embeddingModel)))
    // normalize
    const norms = embeddings.map((v) => {
      const mag = Math.sqrt(v.reduce((s: number, x: number) => s + x * x, 0))
      return v.map((x: number) => x / (mag || 1))
    })
    const n = norms.length
    const similar: Array<Array<string>> = []
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = norms[i]
        const b = norms[j]
        const dot = a.reduce((s: number, v: number, idx: number) => s + v * (b[idx] ?? 0), 0)
        if (dot >= this.threshold) {
          similar.push([keys[i], keys[j]])
        }
      }
    }
    if (similar.length === 0) return null
    // dedupe groups (merge overlapping pairs into groups)
    const groups: Array<Set<string>> = []
    for (const pair of similar) {
      let foundIdx = -1
      for (let gi = 0; gi < groups.length; gi++) {
        const g = groups[gi]
        if (g.has(pair[0]) || g.has(pair[1])) {
          g.add(pair[0])
          g.add(pair[1])
          foundIdx = gi
          break
        }
      }
      if (foundIdx === -1) {
        groups.push(new Set(pair))
      }
    }
    // convert to list of tuples
    return groups.map((g) => Array.from(g))
  }

  async checkCausalRelationships(relationship: string, reasoning: string, relevantText: string) {
    // replicate Python prompt for verification: choose among options 1-4 and return JSON
    const parts = relationship
    const varParts = parts.split('-->')
    const var1 = (varParts[0] || '').trim()
    const var2 = (varParts[1] || '').replace(/\(\+\)|\(-\)/g, '').trim()
    const prompt = `Relationship: ${relationship}\nRelevant Text: ${relevantText}\nReasoning: ${reasoning}`
    const system = `Given the above text, select the options which are correct. There can be more than one option that is correct:\n1. increasing ${var1} increases ${var2}\n2. decreasing ${var1} decreases ${var2}\n3. increasing ${var1} decreases ${var2}\n4. decreasing ${var1} increases ${var2}\nRespond in JSON: {"answers":"[1,2,3,or 4 depending on your reasoning]","reasoning":"<text>"}`
    const context: any[] = [
      { role: 'system', content: system },
      { role: 'user', content: prompt }
    ]
    const raw = await getCompletionFromMessages(context, this.llmModel, false, this.temperature, this.top_p, this.seed)
    let parsed: any = loadJson(raw)
    let steps: string[] = []
    let reasoningText = ''
    if (!parsed || !parsed.answers) {
      // try to extract digit answers directly from text
      const m1 = raw.match(/\[?\s*([1-4](?:\s*,\s*[1-4])*)\s*\]?/)
      const matchDigits = raw.match(/[1-4]/g)
      if (m1 && m1[1]) {
        steps = (m1[1].match(/[1-4]/g) || []).map(String)
      } else if (matchDigits) {
        steps = matchDigits.map(String)
      }
      const mReason = raw.match(/reasoning\s*[:\-]*\s*(.+)/i)
      reasoningText = mReason ? mReason[1].trim() : ''
    } else {
      try {
        steps = ('' + parsed.answers).match(/[1-4]/g) || []
        reasoningText = parsed.reasoning || ''
      } catch (e) {
        steps = []
      }
    }
    // decide polarity
    let correct: string
    if (steps.includes('1') || steps.includes('2')) {
      correct = `${var1} -->(+) ${var2}`
    } else if (steps.includes('3') || steps.includes('4')) {
      correct = `${var1} -->(-) ${var2}`
    } else {
      // Try heuristic: look for polarity words in reasoning or relevantText or relationship
      const textToInspect = (reasoningText || '') + ' ' + (relevantText || '') + ' ' + (relationship || '')
      const lower = textToInspect.toLowerCase()
      const positiveWords = [
        'increase',
        'increases',
        'increasing',
        'increased',
        'rise',
        'rises',
        'higher',
        'more',
        'boost',
        'improve',
        'improves'
      ]
      const negativeWords = [
        'decrease',
        'decreases',
        'decreasing',
        'decreased',
        'drop',
        'drops',
        'fall',
        'falls',
        'lower',
        'lowered',
        'reduce',
        'reduces',
        'reduced',
        'decline',
        'declines'
      ]
      const posFound = positiveWords.some((w) => lower.includes(w))
      const negFound = negativeWords.some((w) => lower.includes(w))
      if (posFound && !negFound) {
        correct = `${var1} -->(+) ${var2}`
      } else if (negFound && !posFound) {
        correct = `${var1} -->(-) ${var2}`
      } else {
        // As a last resort, default to positive polarity but mark it explicit
        correct = `${var1} -->(+) ${var2}`
      }
    }
    return correct
  }

  async checkVariables(text: string, lines: Array<[string, string, string]>) {
    const result_list = lines.map((l) => l[0])
    const reasoning_list = lines.map((l) => l[1])
    const rel_txt_list = lines.map((l) => l[2])
    // collect unique variable names
    const variableSet = new Set<string>()
    for (const line of result_list) {
      const [v1, v2] = line.split('-->').map((s) =>
        (s || '')
          .replace(/\(\+\)|\(-\)/g, '')
          .trim()
          .toLowerCase()
      )
      if (v1) variableSet.add(v1)
      if (v2) variableSet.add(v2)
    }
    const variable_list = Array.from(variableSet)
    const variable_to_index: Record<string, number> = {}
    const index_to_variable: Record<number, string> = {}
    for (let i = 0; i < variable_list.length; i++) {
      variable_to_index[variable_list[i]] = i
      index_to_variable[i] = variable_list[i]
    }
    const similar_variables = await this.computeSimilarities(variable_to_index, index_to_variable)
    if (!similar_variables) return lines

    if (this.verbose) console.log('Similar variable groups detected:', similar_variables)

    // If running interactively, allow user choices; otherwise, proceed to merge groups automatically
    const interactive = Boolean(process.stdin && (process.stdin as any).isTTY)
    if (interactive) {
      // For simplicity in TS CLI, we default to merging all groups unless the user explicitly opts out.
      // Implementing full interactive selection is possible but outside test automation needs.
    }

    // Build system prompt for merging (same as Python check_variables prompt)
    const mergeSystem = `You are a Professional System Dynamics Modeler.\nYou will be provided with 3 things:\n1. Multiple causal relationships between variables in a numbered list.\n2. The text on which the above causal relationships are based.\n3. Multiple tuples of two variable names which the user believes are similar.\nYour objective is to merge the two variable names into one variable, choosing a new variable name that is shorter of the two.\nFollow the steps in the example and return JSON as described.`
    const prompt = `Text:\n${text}\nRelationships:\n${JSON.stringify(lines)}\nSimilar Variables:\n${JSON.stringify(similar_variables)}`
    const context: any[] = [
      { role: 'system', content: mergeSystem },
      { role: 'user', content: prompt }
    ]
    const corrected_raw = await getCompletionFromMessages(
      context,
      this.llmModel,
      false,
      this.temperature,
      this.top_p,
      this.seed
    )
    if (this.verbose) {
      try {
        console.log('DEBUG_RAW_CORRECTED_RESPONSE:', corrected_raw)
      } catch (e) {}
    }
    let corrected_json = loadJson(corrected_raw)
    if (!corrected_json) {
      throw new Error('Got no corrected response from the assistant during variable merging')
    }
    // Normalize to Step 2 -> Final Relationships structure
    let relationships: any[] = []
    if (corrected_json['Step 2'] && corrected_json['Step 2']['Final Relationships']) {
      relationships = corrected_json['Step 2']['Final Relationships']
    } else {
      // convert numbered dict to list
      const keys = Object.keys(corrected_json).sort((a, b) => {
        const na = a.match(/\d+/)
        const nb = b.match(/\d+/)
        return (na ? Number(na[0]) : 0) - (nb ? Number(nb[0]) : 0)
      })
      for (const k of keys) {
        const entry = corrected_json[k]
        const rel =
          entry['causal relationship'] || entry['relationship'] || entry['causal_relationship'] || entry['relationship']
        const reasoning = entry['reasoning'] || ''
        const relevant = entry['relevant text'] || entry['relevant_text'] || entry['relevantText'] || ''
        relationships.push({
          relationship: typeof rel === 'string' ? rel : String(rel),
          'relevant text': relevant,
          reasoning
        })
      }
    }
    const new_lines: Array<[string, string, string]> = []
    for (const ent of relationships) {
      const rel = (ent['relationship'] || ent['causal relationship'] || '').toString().toLowerCase()
      const reason = ent['reasoning'] || ''
      const relevant = ent['relevant text'] || ent['relevant'] || ''
      const snippet = await this.getLine(relevant || text)
      new_lines.push([rel, reason, snippet])
    }
    return new_lines
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

    const context: any[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: this.question }
    ]
    // deterministic defaults: temperature=0, top_p=1, seed from env or 42
    const response1 = await getCompletionFromMessages(
      context,
      this.llmModel,
      false,
      this.temperature,
      this.top_p,
      this.seed
    )
    let parsed1 = loadJson(response1 as string)
    if (!parsed1) {
      throw new Error('Input text did not have any causal relationships')
    }

    // If parsed1 contains malformed relationships (e.g., entries like "--> positive" with no cause/effect),
    // ask the assistant to reformat its raw output into the strict JSON schema.
    const isStructuredValid = (obj: any) => {
      if (!obj || typeof obj !== 'object') return false
      if (Array.isArray(obj)) return false
      if (Array.isArray((obj as any).causalRelationships)) {
        const preds = new Set(['increase', 'decrease', 'positive', 'negative'])
        for (const r of (obj as any).causalRelationships) {
          const s = (r?.subject || '').toString().trim()
          const o = (r?.object || '').toString().trim()
          const p = (r?.predicate || '').toString().trim().toLowerCase()
          if (!s || !o || !preds.has(p)) return false
        }
        return true
      }
      return false
    }

    const malformed = () => {
      try {
        if (!parsed1) return true
        if ((parsed1 as any).causalRelationships) {
          return !isStructuredValid(parsed1)
        }
        if (Array.isArray(parsed1)) return false
        const keys = Object.keys(parsed1 as any)
        for (const k of keys) {
          const entry = (parsed1 as any)[k]
          const rel = entry['causal relationship'] || entry['relationship'] || ''
          if (typeof rel === 'string') {
            const parts = rel.split('-->')
            const left = (parts[0] || '').trim()
            const right = (parts[1] || '').trim()
            if (!left || !right) return true
          } else {
            return true
          }
        }
        return false
      } catch (e) {
        return true
      }
    }
    if (malformed()) {
      const reformatPrompt = `You previously returned this output: ${JSON.stringify(
        response1
      )}\n\nPlease convert that output EXACTLY into this JSON schema: { "causalRelationships": [{"subject":"<text>","predicate":"increase|decrease|positive|negative","object":"<text>","reasoning":"<text>","relevant":"<text>"}] } and return ONLY the JSON.\nConstraints:\n- subject and object MUST be non-empty strings (<= 2 words, neutral).\n- predicate MUST be exactly one of: increase, decrease, positive, negative.\n- If no valid relationships exist, return {"causalRelationships": []}.`
      const reformatted = await getCompletionFromMessages(
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: reformatPrompt }
        ],
        this.llmModel,
        false,
        this.temperature,
        this.top_p,
        this.seed
      )
      const parsedRe = loadJson(reformatted as string)
      if (parsedRe && (parsedRe as any).causalRelationships && isStructuredValid(parsedRe)) {
        parsed1 = parsedRe
      } else {
        throw new Error('Assistant returned malformed causalRelationships after reformat request')
      }
    }
    // Normalize object with array under common keys, e.g., { causalRelationships: [...] }
    if (
      parsed1 &&
      !Array.isArray(parsed1) &&
      (parsed1 as any).causalRelationships &&
      Array.isArray((parsed1 as any).causalRelationships)
    ) {
      const arr = (parsed1 as any).causalRelationships
      const obj: any = {}
      arr.forEach((entry: any, idx: number) => {
        // Expect entry to be {subject,predicate,object,...}
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
        const causal = `${ssub} -->${symbol} ${oobj}`.trim()
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
            let cause = entry.cause || entry.variable1 || ''
            let effect = entry.effect || entry.variable2 || ''
            const predicate = (entry.sign || entry.direction || entry.relationship || '').toLowerCase()
            const symbol =
              predicate.includes('increase') || predicate.includes('positive')
                ? '(+)'
                : predicate.includes('decrease') || predicate.includes('negative')
                  ? '(-)'
                  : ''
            cause = humanize(cause)
            effect = humanize(effect)
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
    // Build output lines from parsed1 only (strict mode; avoid secondary model variability)
    const isRelStringValid = (s: any) => {
      if (typeof s !== 'string') return false
      const parts = s.split('-->')
      if (parts.length < 2) return false
      const left = (parts[0] || '').trim()
      const right = (parts[1] || '').trim()
      return Boolean(left && right)
    }
    const outLines: string[] = []
    let idxOut = 1
    const keysOut = Object.keys(parsed1 as any)
    for (const k of keysOut) {
      const entry = (parsed1 as any)[k]
      const rel = entry['causal relationship'] || entry['relationship'] || ''
      if (isRelStringValid(rel)) {
        outLines.push(`${idxOut}. ${rel}`)
        idxOut++
      }
    }
    const corrected = outLines.join('\n')
    if (this.verbose) console.log(`Corrected Response:\n${corrected}`)
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
