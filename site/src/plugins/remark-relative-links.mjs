// Pages link to each other with relative `.md` paths (`../reference/settings.md#models`), which
// work on GitHub and in the Markdown copies. On the site, each page lives at its slug with a
// trailing slash, so this plugin rewrites those links to site addresses.
import path from "node:path";
import { fileURLToPath } from "node:url";

const docsRoot = fileURLToPath(new URL("../content/docs/", import.meta.url));
const relativePage = /^(?![a-z][a-z0-9+.-]*:|\/|#)([^#?]+\.mdx?)(#.*)?$/i;

/** @param {{ base: string }} options */
export function remarkRelativeLinks({ base }) {
  const prefix = base.replace(/\/$/, "");
  /** @param {any} tree @param {any} file */
  return (tree, file) => {
    if (!file.path) return;
    /** @param {any} node */
    const visit = (node) => {
      if ((node.type === "link" || node.type === "definition") && typeof node.url === "string") {
        const match = relativePage.exec(node.url);
        if (match) {
          const target = path.resolve(path.dirname(file.path), match[1] ?? "");
          const slug = path
            .relative(docsRoot, target)
            .split(path.sep)
            .join("/")
            .replace(/\.mdx?$/i, "")
            .replace(/(^|\/)index$/, "")
            .toLowerCase();
          node.url = `${prefix}/${slug}${slug ? "/" : ""}${match[2] ?? ""}`;
        }
      }
      if (Array.isArray(node.children)) node.children.forEach(visit);
    };
    visit(tree);
  };
}
