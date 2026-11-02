// @ts-check
// tensorless.dev: the home page (src/pages/index.astro) and the documentation under /docs/
// (src/content/docs/docs/, one Diátaxis type per page; src/content.config.ts).
import starlight from "@astrojs/starlight";
import { defineConfig } from "astro/config";
import mermaid from "astro-mermaid";
import starlightLinksValidator from "starlight-links-validator";
import starlightLlmsTxt from "starlight-llms-txt";
import { remarkRelativeLinks } from "./src/plugins/remark-relative-links.mjs";
import { fathom } from "./src/analytics.mjs";
import { release } from "./src/release.mjs";

const site = "https://tensorless.dev";
const base = "/";
const repo = "https://github.com/iammrduncan/tensorless";

export default defineConfig({
  site,
  base,
  trailingSlash: "always",
  devToolbar: { enabled: false },
  vite: {
    // Mermaid is one large script, loaded only on pages with a diagram.
    build: { chunkSizeWarningLimit: 4096 },
    define: { __TENSORLESS_VERSION__: JSON.stringify(release.launcher) },
  },
  markdown: {
    // Pages link to each other with relative `.md` paths, so the links also work on GitHub.
    // This turns them into site addresses.
    remarkPlugins: [[remarkRelativeLinks, { base }]],
  },
  integrations: [
    // astro-mermaid must come before Starlight.
    mermaid({ theme: "neutral", autoTheme: true }),
    starlight({
      title: "tensorless",
      description:
        "Run a model tuned for your GPU, on the engine it was tuned with, in one command. Custom engine and quantization for the NVIDIA Tesla P100.",
      favicon: "/favicon.svg",
      head: [
        // The social-preview image (scripts/og.html renders it).
        { tag: "meta", attrs: { property: "og:image", content: `${site}/og.png` } },
        { tag: "meta", attrs: { name: "twitter:image", content: `${site}/og.png` } },
        { tag: "meta", attrs: { name: "twitter:card", content: "summary_large_image" } },
        // Fathom Analytics (src/analytics.mjs).
        { tag: "script", attrs: fathom },
      ],
      components: {
        SiteTitle: "./src/components/SiteTitle.astro",
        // Dark only, like the home page.
        ThemeProvider: "./src/components/starlight/ThemeProvider.astro",
        ThemeSelect: "./src/components/starlight/ThemeSelect.astro",
        SocialIcons: "./src/components/starlight/SocialIcons.astro",
      },
      social: [
        { icon: "github", label: "GitHub", href: repo },
        { icon: "discord", label: "Discord", href: "https://discord.gg/3Qs2uejUf9" },
      ],
      editLink: { baseUrl: `${repo}/edit/main/site/` },
      lastUpdated: true,
      customCss: [
        "@fontsource-variable/geist",
        "@fontsource-variable/geist-mono",
        "./src/styles/docs.css",
      ],
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
      sidebar: [
        { label: "Start here", link: "/docs/" },
        { label: "Tutorials", items: [{ autogenerate: { directory: "docs/tutorials" } }] },
        {
          label: "How-to guides",
          items: [
            { label: "Set up", items: [{ autogenerate: { directory: "docs/how-to/setup" } }] },
            { label: "Run models", items: [{ autogenerate: { directory: "docs/how-to/run" } }] },
            { label: "Connect and measure", items: [{ autogenerate: { directory: "docs/how-to/use" } }] },
            { label: "Go further", items: [{ autogenerate: { directory: "docs/how-to/advanced" } }] },
          ],
        },
        { label: "Reference", items: [{ autogenerate: { directory: "docs/reference" } }] },
        { label: "Explanation", items: [{ autogenerate: { directory: "docs/explanation" } }] },
        { label: "Project", collapsed: true, items: [{ autogenerate: { directory: "docs/project" } }] },
      ],
      plugins: [
        starlightLinksValidator({
          // Relative links are turned into site addresses before the validator sees them.
          errorOnRelativeLinks: false,
          // Pages link to a server on the reader's own computer (127.0.0.1:8080).
          errorOnLocalLinks: false,
          // The home page is an Astro page outside the docs collection.
          exclude: ["/"],
        }),
        starlightLlmsTxt({
          projectName: "tensorless",
          exclude: ["404"],
          description:
            "tensorless runs a model tuned for a GPU, on the engine it was tuned with, in one command: `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF`. The first card is the NVIDIA Tesla P100 (16 GB, sm_60). A recipe pins the engine image by digest and the model by revision and sha256, with the measured settings and numbers.",
          customSets: [
            { label: "Tutorials", paths: ["docs/tutorials/**"], description: "learn tensorless by doing" },
            { label: "How-to guides", paths: ["docs/how-to/**"], description: "the steps for one task each" },
            {
              label: "Reference",
              paths: ["docs/reference/**"],
              description: "commands, recipes, images, the PN format and receipts, described completely",
            },
            { label: "Explanation", paths: ["docs/explanation/**"], description: "how tensorless works, and why" },
          ],
        }),
      ],
    }),
  ],
});
