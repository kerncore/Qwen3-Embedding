import fs from 'fs/promises';
import path from 'path';
import minimist from 'minimist';
import { getLlama } from 'node-llama-cpp';

async function loadTaskDesc(taskName) {
  const data = JSON.parse(await fs.readFile('task_prompts.json', 'utf8'));
  let desc = data[taskName];
  if (typeof desc === 'object') {
    desc = desc.query ?? Object.values(desc)[0];
  }
  if (!desc) {
    throw new Error(`Task '${taskName}' not found`);
  }
  return desc;
}

async function main() {
  const argv = minimist(process.argv.slice(2), {
    string: ['task', 'query', 'document', 'model'],
    alias: { t: 'task', q: 'query', d: 'document', m: 'model' },
    default: { model: 'Qwen/Qwen3-Embedding-0.6B' },
  });
  if (!argv.task || !argv.query || !argv.document) {
    console.error('Usage: yarn embed --task <task> --query <text> --document <file>');
    process.exit(1);
  }

  const taskDesc = await loadTaskDesc(argv.task);
  const docText = await fs.readFile(argv.document, 'utf8');
  const prompt = `Instruct: ${taskDesc}\nQuery:${argv.query}`;

  const llama = await getLlama();
  const model = await llama.loadModel({ modelPath: argv.model });
  const ctx = await model.createEmbeddingContext();

  const promptEmb = await ctx.getEmbeddingFor(prompt);
  const docEmb = await ctx.getEmbeddingFor(docText);
  const similarity = promptEmb.calculateCosineSimilarity(docEmb);

  console.log(`Similarity: ${similarity.toFixed(4)}`);
  await ctx.dispose();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
