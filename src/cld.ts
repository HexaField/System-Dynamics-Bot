import { cosineSimilarity, getCompletionFromMessages, getEmbedding, loadJson } from './utils'

function simpleSentenceSplit(text: string): string[] {
  // very simple sentence splitter
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean)
}

const systemPrompt = `You are a System Dynamics Professional Modeler.
Users will give text, and it is your job to generate causal relationships from that text.
You will conduct a multistep processs:

1. You will identify all the words that have cause and effect between two entities in the text. These entities are variables. \
Name these variables in a concise manner. A variable name should not be more than 2 words. Make sure that you minimize the number of variables used. Variable names should be neutral, i.e., \
it shouldn't have positive or negative meaning in their names.

2. For each variable, represent the causal relationships with other variables. There are two types of causal relationships: positive and negative.\
A positive relationship exits if a decline in variable1 leads to a decline in variable2. Also a positive relationship exists if an increase in variable1 leads to an increase in variable2.\
If there is a positive relationship, use the format: "Variable1" -->(+) "Variable2".\
A negative relationship exists if an increase in variable1 leads to a decline in variable2. Also a negative relationship exists if a decline in variable1 leads to an increase in variable2.\
If there is a negative relationship, use the format: "Variable1" -->(-) "Variable2".

3. Not all variables may have any relationship with any other variables.

4. When three variables are related in a sentence, make sure the relationship between second and third variable is correct.\
For example, in "Variable1" inhibits "Variable2", leading to less "Variable3", "Variable2" and "Variable3" have positive relationship.


5. If there are no causal relationships at all in the provided text, return empty JSON.

Example 1 of a user input:
"when death rate goes up, population decreases"

Corresponding JSON response:
{"1": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "Death rate --> (-) population",  "relevant text": "[the full text/paragraph that highlights this relationship]"}}

Example 2 of a user input:
"increased death rate reduces population"

Corresponding JSON response:
{"1": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "Death rate --> (-) population",  "relevant text": "[the full text/paragraph that highlights this relationship]"}}

Example 3 of a user input:
"lower death rate increases population"

Corresponding JSON response:
{"1": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "Death rate --> (-) population",  "relevant text": "[the full text/paragraph that highlights this relationship]"}}

Example 4 of a user input:
"The engineers compare the work remaining to be done against the time remaining before the deadline. The larger the gap, the more Schedule Pressure they feel. \
When schedule pressure builds up, engineers have several choices. First, they can work overtime. Instead of the normal 50 hours per week, they can come to work early, \
skip lunch, stay late, and work through the weekend. By burning the Midnight Oil, the increase the rate at which they complete their tasks, cut the backlog of work, \
and relieve the schedule pressure. However, if the workweek stays too high too long, fatigue sets in and productivity suffers. As productivity falls, the task completion rate drops, \
which increase schedule pressure and leads to still longer hours. Another way to complete the work faster is to reduce the time spent on each task. \
Spending less time on each task boosts the number of tasks done per hour (productivity) and relieve schedule pressure. \
Lower time per task increases error rate, which leads to rework and lower productivity in the long run."

Corresponding JSON response:
{
  "1": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "work remaining -->(+) Schedule Pressure", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "2": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "time remaining -->(-) Schedule Pressure", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "3": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "Schedule Pressure --> (+) overtime", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "4": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "overtime --> (+) completion rate", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "5": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "completion rate --> (-) work remaining", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "6": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "overtime --> (+) fatigue", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "7": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "fatigue --> (-) productivity", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "8": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "productivity --> (+) completion rate", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "9": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "Schedule Pressure --> (-) Time per task", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "10": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "Time per task --> (-) error rate", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "11": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "error rate --> (-) productivity", "relevant text": "[the full text/paragraph that highlights this relationship]"}
}

Example 5 of a user input:
"Congestion (i.e., travel time) creates pressure for new roads; after the new capacity is added, travel time falls, relieving the pressure. \
New roads are built to relieve congestion. In the short run, travel time falls and atractiveness of driving goes up—the number of cars in the region hasn’t changed and -\
people’s habits haven’t adjusted to the new, shorter travel times. \
As people notice that they can now get around much faster than before, they will take more Discretionary trips (i.e., more trips per day). They will also travel extra miles, leading to higher trip length. \
Over time, seeing that driving is now much more attractive than other modes of transport such as the public transit system, some people will give up the bus or subway and buy a car. \
The number of cars per person rises as people ask why they should take the bus.

Corresponding JSON response:
{
  "1": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "travel time --> (+) pressure for new roads", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "2": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "pressure for new roads --> (+) road construction", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "3": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "road construction --> (+) Highway capacity", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "4": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "Highway capacity --> (-) travel time", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "5": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "travel time --> (-) attractiveness of driving", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "6": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "attractiveness of driving --> (+) trips per day", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "7": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "trips per day --> (+) traffic volume", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "8": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "traffic volume --> (+) travel time", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "9": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "attractiveness of driving --> (+) trip length", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "10": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "trip length --> (+) traffic volume", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "11": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "attractiveness of driving --> (-) public transit", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "12": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "public transit --> (-) cars per person", "relevant text": "[the full text/paragraph that highlights this relationship]"},
  "13": {"reasoning": "[your reasoning for this causal relationship]", "causal relationship": "cars per person --> (+) traffic volume", "relevant text": "[the full text/paragraph that highlights this relationship]"}
}

Example 6 of a user input:
"[Text with no causal relationships]"

Corresponding JSON response:
{}

Please ensure that you only provide the appropriate JSON response format and nothing more. Ensure that you follow the example JSON response formats provided in the examples.
`

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
      // fallback: keep original
      correct = relationship
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
    // Normalize array responses into the expected keyed object format
    if (Array.isArray(parsed1)) {
      const obj: any = {}
      parsed1.forEach((entry: any, idx: number) => {
        let cause = entry.cause || entry.variable1 || ''
        let effect = entry.effect || entry.variable2 || ''
        const rel = (entry.relationship || '').toLowerCase()
        const symbol =
          rel.includes('increase') || rel.includes('positive')
            ? '(+)'
            : rel.includes('decrease') || rel.includes('negative')
              ? '(-)'
              : ''
        // normalize variable names: split camelCase/PascalCase and underscores, lowercase
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
        let cause = entry.cause || entry.variable1 || ''
        let effect = entry.effect || entry.variable2 || ''
        const rel = (entry.direction || entry.relationship || '').toLowerCase()
        const symbol =
          rel.includes('increase') || rel.includes('positive')
            ? '(+)'
            : rel.includes('decrease') || rel.includes('negative')
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
            const rel = (entry.sign || entry.direction || entry.relationship || '').toLowerCase()
            const symbol =
              rel.includes('increase') || rel.includes('positive')
                ? '(+)'
                : rel.includes('decrease') || rel.includes('negative')
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
    // secondary check for loops
    const query = `Find out if there are any possibilities of forming closed loops that are implied in the text. If yes, then close the loops by adding the extra relationships and provide them in a JSON format please.`
    context.push({ role: 'user', content: query })
    const response2 = await getCompletionFromMessages(
      context,
      this.llmModel,
      false,
      this.temperature,
      this.top_p,
      this.seed
    )
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

    // Check for similar variables and possibly merge via assistant-guided merging
    const checked_response = await this.checkVariables(this.question, lines)

    if (this.verbose) {
      console.log('After checking for similar variables:')
      for (let i = 0; i < checked_response.length; i++) {
        const vals = checked_response[i]
        console.log(`${i + 1}. ${vals[0]}`)
        console.log(`Reasoning: ${vals[1]}`)
        console.log(`Relevant Text: ${vals[2]}`)
      }
    }

    // For each relationship, validate its polarity by asking the assistant (replicates Python check)
    const corrected_pairs: string[] = []
    for (let i = 0; i < checked_response.length; i++) {
      const vals = checked_response[i]
      if (this.verbose) console.log(`Checking relationship #${i + 1}...`)
      const correctedRel = await this.checkCausalRelationships(vals[0], vals[1], vals[2])
      corrected_pairs.push(`${i + 1}. ${correctedRel}`)
    }

    const corrected = corrected_pairs.join('\n')
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
