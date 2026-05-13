#!/usr/bin/env node

const DEFAULT_TEXT =
  "Careful source evaluation matters because reliable arguments depend on evidence, context, and transparent reasoning.";

function parseArgs(argv) {
  const args = {
    url: "http://127.0.0.1:8000/predict",
    requests: 50,
    text: DEFAULT_TEXT,
  };
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === "--url") args.url = value;
    if (key === "--requests") args.requests = Number(value);
    if (key === "--text") args.text = value;
  }
  return args;
}

function percentile(values, q) {
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.min(ordered.length - 1, Math.max(0, Math.round((ordered.length - 1) * q)));
  return ordered[index];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const latencies = [];
  for (let index = 0; index < args.requests; index += 1) {
    const started = performance.now();
    const response = await fetch(args.url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: args.text }),
    });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    await response.json();
    latencies.push((performance.now() - started) / 1000);
  }
  const median = percentile(latencies, 0.5);
  const p95 = percentile(latencies, 0.95);
  console.log(`requests: ${latencies.length}`);
  console.log(`median_seconds: ${median.toFixed(4)}`);
  console.log(`p95_seconds: ${p95.toFixed(4)}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
