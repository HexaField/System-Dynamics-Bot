import { generateCausalLoopDiagram as generateCLD, extractVariables } from './cld'
import { cleanSymbol, xmileName } from './utils'

type SageOpts = {
  verbose?: boolean
  diagram?: boolean
  xmile?: boolean
  threshold?: number
  question?: string
  llmModel?: string
  embeddingModel?: string
  temperature?: number
  top_p?: number
  seed?: number
}

function generateXmile(resultList: string[]) {
  let variablesDict: Record<string, string[]> = {}
  let connectors = ''
  for (const line of resultList) {
    const [v1, v2, symbol] = extractVariables(line)
    if (!v1 || !v2 || v1 === v2) continue
    if (!variablesDict[v2]) variablesDict[v2] = []
    variablesDict[v2].push(v1)
    connectors += `\t\t\t\t<connector polarity=\"${cleanSymbol(symbol)}\">\n`
    connectors += `\t\t\t\t\t<from>${xmileName(v1)}</from>\n`
    connectors += `\t\t\t\t\t<to>${xmileName(v2)}</to>\n`
    connectors += `\t\t\t\t</connector>\n`
  }

  let xmileVariables = ''
  for (const [variable, causers] of Object.entries(variablesDict)) {
    xmileVariables += `\t\t\t<aux name=\"${variable}\">\n`
    xmileVariables += `\t\t\t\t<eqn>NAN(${causers.map((c) => xmileName(c)).join(',')})</eqn>\n`
    xmileVariables += `\t\t\t\t<isee:delay_aux/>\n`
    xmileVariables += `\t\t\t</aux>\n`
  }

  const xmile = `<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<xmile version=\"1.0\">\n\t<model>\n\t\t<variables>\n${xmileVariables}\t\t</variables>\n\t\t<views>\n\t\t\t${connectors}\t\t</views>\n\t</model>\n</xmile>`
  return xmile
}

function generateDot(resultList: string[]) {
  let dot = 'digraph G {\n  rankdir=LR;\n  node [shape=box];\n'
  for (const line of resultList) {
    const [v1, v2, symbol] = extractVariables(line)
    if (!v1 || !v2 || v1 === v2) continue
    const label = symbol || ''
    dot += `  \"${v1}\" -> \"${v2}\" [label=\"${label}\"];\n`
  }
  dot += '}\n'
  return dot
}

export async function generateCausalLoopDiagram(opts: SageOpts) {
  const verbose = !!opts.verbose
  const diagram = !!opts.diagram
  const xmileFlag = !!opts.xmile
  const threshold = opts.threshold ?? 0.85
  const question = opts.question
  const llmModel = opts.llmModel
  const embeddingModel = opts.embeddingModel

  if (!question) throw new Error('No question provided')

  const response = await generateCLD({
    question,
    threshold,
    verbose,
    llmModel,
    embeddingModel,
    temperature: opts.temperature ?? 0,
    top_p: opts.top_p ?? 1,
    seed: opts.seed ?? (Number(process.env.SEED) || 42)
  })

  // dedupe and normalize lines
  const lines = response
    .split('\n')
    .map((l) => l.replace(/^[0-9]+\.\s*/, '').trim())
    .filter(Boolean)
  const uniq = Array.from(new Set(lines))

  function normalizeLine(line: string): string | null {
    let s = line.trim()
    if (!s.includes('-->')) return null
    const parts = s.split('-->')
    let left = parts[0].trim()
    let right = parts.slice(1).join('-->').trim()

    // detect existing symbol
    const symbolMatch = right.match(/\(\+\)|\(\-\)/)
    let symbol = symbolMatch ? symbolMatch[0] : ''
    // remove any existing symbol from right
    right = right.replace(/\(\+\)|\(\-\)/g, '').trim()

    left = left
      .replace(/["'()\.,]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
    right = right
      .replace(/["'()\.,]/g, '')
      .replace(/\s+/g, ' ')
      .trim()

    if (!left || !right) return null
    return `${left} -->${symbol} ${right}`
  }

  const normalized: string[] = []
  for (const l of uniq) {
    const n = normalizeLine(l)
    if (n) normalized.push(n)
  }
  // Deduplicate normalized relationships
  const uniqNormalized = Array.from(new Set(normalized))

  const result: any = {
    response,
    lines: uniqNormalized
  }

  if (xmileFlag) {
    result.xmile = generateXmile(uniqNormalized)
  }

  if (diagram) {
    result.dot = generateDot(uniqNormalized)
  }

  return result
}
