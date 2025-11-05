#!/usr/bin/env node
import { Command } from 'commander'
import fs from 'fs'
import inquirer from 'inquirer'
import { generateCausalLoopDiagram } from './sage'

const program = new Command()

program
  .option('-v, --verbose', 'enable verbosity')
  .option('-d, --diagram', 'enable causal loop diagram generation')
  .option('-w, --write-relationships', 'write the final relationships to a text file')
  .option('-x, --xmile', 'save the generated diagram as XMILE')
  .option('-t, --threshold <n>', 'similarity threshold', parseFloat, 0.85)
  .option('--llm-model <model>', 'LLM/chat model to use (overrides OLLAMA_CHAT_MODEL)')
  .option('--embedding-model <model>', 'Embedding model to use (overrides OLLAMA_EMBEDDING_MODEL)')
  .option('--seed <n>', 'random seed for deterministic runs', (v) => Number(v))
  .option('--temperature <n>', 'LLM temperature', parseFloat)
  .option('--top_p <n>', 'LLM top_p', parseFloat)
  .option('-i, --input <file>', 'read input text from file')
  .parse(process.argv)

async function main() {
  const opts = program.opts()
  let question: string | undefined = undefined
  if (opts.input) {
    question = fs.readFileSync(opts.input, 'utf8')
  } else {
    const res = await inquirer.prompt([
      {
        type: 'editor',
        name: 'q',
        message: 'Enter your problem description here:'
      }
    ])
    question = res.q
  }

  const result = await generateCausalLoopDiagram({
    verbose: opts.verbose,
    diagram: !!opts.diagram,
    xmile: !!opts.xmile,
    threshold: opts.threshold,
    question,
    llmModel: opts.llmModel,
    embeddingModel: opts.embeddingModel
    ,
    temperature: opts.temperature !== undefined ? Number(opts.temperature) : undefined,
    top_p: opts.top_p !== undefined ? Number(opts.top_p) : undefined,
    seed: opts.seed !== undefined ? Number(opts.seed) : undefined
  })

  if (result) {
    console.log('Final Relationship:\n', result.response)
    if (opts.writeRelationships || opts.write_relationships) {
      fs.writeFileSync('relationships.txt', result.response, 'utf8')
    }
    if (opts.xmile && result.xmile) {
      fs.writeFileSync('diagram.xmile', result.xmile, 'utf8')
    }
    if (opts.diagram && result.dot) {
      fs.writeFileSync('diagram.dot', result.dot, 'utf8')
    }
  }
}

main().catch((err) => {
  console.error('Error:', err.message || err)
  process.exit(1)
})
