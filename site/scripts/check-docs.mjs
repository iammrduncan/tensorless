// What the build cannot check: `npm run check`, from site/.
// - every page's `type` matches its folder (one Diátaxis type per folder);
// - no lab-only term, private name or old model name appears on the public site;
// - every `tensorless COMMAND` and every flag used with it exists in the launcher's code;
// - the install script installs the release the site describes.
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const site = fileURLToPath(new URL("../", import.meta.url));
const repo = path.resolve(site, "..");
const docs = path.join(site, "src/content/docs");
const problems = [];

function walk(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = path.join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : /\.(md|mdx|astro|mjs|ts|sh)$/.test(n) ? [p] : [];
  });
}

// 1. Types by folder.
const typeOf = { tutorials: "tutorial", "how-to": "how-to", reference: "reference", explanation: "explanation", project: "project" };
const pages = walk(docs).filter((f) => /\.mdx?$/.test(f));
for (const f of pages) {
  const rel = path.relative(docs, f);
  const fm = readFileSync(f, "utf8").match(/^---\n([\s\S]*?)\n---/);
  const type = fm?.[1].match(/^type:\s*"?([\w-]+)"?/m)?.[1];
  const folder = rel.split(path.sep)[1];
  const want = typeOf[folder] ?? "hub";
  if (type !== want) problems.push(`${rel}: type is ${type}, but pages in ${folder ?? "the top"} are ${want}`);
}

// 2. Terms that must not reach the public site. Model names: use the Hugging Face names.
// Generic patterns live here. Names that must never reach the public site but are private
// themselves (hosts, repositories, people) go in site/.private-terms, which git ignores: one
// regular expression per line, optionally followed by a tab and the reason.
const banned = [
  [/PN3M19/, "an old model name (Qwen3.8-27B-PN3)"],
  [/\bA1\b(?![-_.])/, "an old model name (Qwen3.8-27B-UD-IQ3_XXS-Q3Floor)"],
  [/\/Users\/|\/home\/(?!you\b|user\b)[a-z]+\//, "a personal home directory"],
];
try {
  for (const line of readFileSync(path.join(site, ".private-terms"), "utf8").split("\n")) {
    const [re, why] = line.split("\t");
    if (re.trim() && !re.startsWith("#")) banned.push([new RegExp(re.trim(), "i"), why?.trim() || "a private term"]);
  }
} catch {
  // No local list: only the generic patterns are checked.
}
const publicFiles = [...walk(path.join(site, "src")), path.join(site, "public/install.sh")];
for (const f of publicFiles) {
  const text = readFileSync(f, "utf8");
  const lines = text.split("\n");
  lines.forEach((line, i) => {
    for (const [re, what] of banned) {
      if (re.test(line)) problems.push(`${path.relative(site, f)}:${i + 1}: ${what}: ${line.trim().slice(0, 100)}`);
    }
  });
}

// 3. Commands and flags against the launcher's code.
const cli = readFileSync(path.join(repo, "internal/cli/cli.go"), "utf8");
const opts = readFileSync(path.join(repo, "internal/cli/options.go"), "utf8");
const commands = new Set([...cli.matchAll(/\{name: "([a-z]+)"/g)].map((m) => m[1]).concat(["help"]));
const flags = new Set(
  [...opts.matchAll(/fs\.(?:String|Bool|Int)Var\([^,]+,\s*"([a-z-]+)"/g), ...opts.matchAll(/fs\.Var\([^,]+,\s*"([a-z-]+)"/g)].map((m) => m[1]),
);
flags.add("h");
flags.add("help");
flags.add("version");
for (const f of pages) {
  const rel = path.relative(docs, f);
  const text = readFileSync(f, "utf8");
  const blocks = [...text.matchAll(/```(?:sh|bash|shell)\b[^\n]*\n([\s\S]*?)```/g)].map((m) => m[1]); // text blocks are output
  const inline = [...text.matchAll(/`(tensorless [^`]+)`/g)].map((m) => m[1]);
  for (const chunk of [...blocks, ...inline]) {
    for (const line of chunk.split("\n")) {
      const m = line.match(/(?:^|[\s$;&|(])(?:\.\/)?tensorless\s+([a-z][a-z-]*)(?=\s|$)(.*)$/);
      if (!m) continue;
      const [, cmd, rest] = m;
      if (!commands.has(cmd)) {
        if (/^(is|runs|run's|was|has|can|does|will|and|on|the|engine|launcher|recipe|recipes)$/.test(cmd)) continue; // prose
        problems.push(`${rel}: unknown command "tensorless ${cmd}" in: ${line.trim().slice(0, 100)}`);
        continue;
      }
      const own = rest.split(/\s--\s/)[0]; // after a bare --, the flags are llama-server's
      for (const fm of own.matchAll(/(?:^|\s)--?([a-z][a-z-]*)/g)) {
        if (!flags.has(fm[1])) problems.push(`${rel}: unknown flag --${fm[1]} for "tensorless ${cmd}" in: ${line.trim().slice(0, 100)}`);
      }
    }
  }
}

// 4. The install script installs the release the site describes.
const { release } = await import(path.join(site, "src/release.mjs"));
const install = readFileSync(path.join(site, "public/install.sh"), "utf8");
const def = install.match(/TENSORLESS_VERSION:-([0-9A-Za-z.-]+)\}/)?.[1];
if (def !== release.launcher) problems.push(`public/install.sh installs ${def} by default, but the site describes ${release.launcher}`);

if (problems.length) {
  console.error(`${problems.length} problem${problems.length === 1 ? "" : "s"}:\n- ${problems.join("\n- ")}`);
  process.exit(1);
}
console.log(`docs check: ${pages.length} pages, ${commands.size} commands and ${flags.size} flags known; no problems`);
