# tensorless.dev

The tensorless website: a home page (`src/pages/index.astro`) and the documentation under `/docs/`, built with [Astro](https://astro.build/) and [Starlight](https://starlight.astro.build/). It needs Node.js 22.22 or later.

```sh
cd site
npm ci
npm run dev      # a live preview at http://127.0.0.1:4321/
npm run build    # the site, in dist/; fails on a broken internal link
npm run check    # what the build cannot check; run it after a docs change
```

## How it is put together

- **The home page** is `src/pages/index.astro`, made of the sections in `src/components/home/`, with `src/layouts/HomeLayout.astro` around them and `src/styles/home.css` for the shared styles: light paper and ink, Geist, one blue accent, and a dark theme that shares the docs' theme setting. Each section keeps its own styles and script. Animations stop for readers who ask for reduced motion (`src/scripts/motion.ts`), and the model map has its own pause button.
- **The model map and the PN blocks are real data.** `scripts/build-model-map.py` reads the headers of both released GGUF files from Hugging Face (byte ranges, at the recipes' pinned revisions) and writes `src/data/model-map.json`: every tensor's shape, type and size in both models, which inputs the engine rotates, the 19 promoted tensors, the bench prompt tokenized with the model's own vocabulary (it must come to the 429 tokens the server counted), and a few real PN blocks. Run it again with `python3 -I scripts/build-model-map.py` when a recipe moves to another file.
- **Numbers come from the repository.** `src/release.mjs` reads the recipes in `../recipes/` and holds the release's versions and its measured numbers, with their sources. The home page takes its numbers from there, and the build fails if a released recipe pins another image than the release's. When a release changes, update `src/release.mjs` and the pages that quote it.
- **The docs** are Markdown in `src/content/docs/docs/`, one [Diátaxis](https://diataxis.fr/) type per page, and the folder says which: `tutorials/`, `how-to/<group>/`, `reference/`, `explanation/`, `project/`. Every page needs a `title`, a one-line `description` and its `type`. The sidebar follows the folders; `sidebar.order` sets a page's place.
- **The install script** is `public/install.sh`, served at `https://tensorless.dev/install.sh`. Its default version must be the release the site describes (`npm run check` compares them).
- **The social image** is `public/og.png`, a 1200 × 630 screenshot of the home page's hero. **The README banner** is `../docs/assets/banner.png`: Bitsy and the name, over a render of the hero's tower.
- **The logo** is Bitsy, `src/assets/logo.svg`. `public/favicon.svg` holds two versions: a hand-placed 16 × 16 pixel Bitsy for browser tabs (24 px and below) and the smooth one above.
- **Response headers** for Cloudflare Pages, including the Content Security Policy, are in `public/_headers`. A page that loads anything from another host needs that host added there.
- **Analytics** are [Fathom](https://usefathom.com/): `src/analytics.mjs` holds the script's settings, and both the home page and the docs load it. It counts only `tensorless.dev`, not `pages.dev` deployments or local previews.

## Writing docs

- Start with what the reader will do or learn, and why. Plain words, short sentences, a working example near the top.
- Only what exists: every command, flag, setting, message and number must be in the code, the recipes or the receipts. Never invent output; run the launcher, or quote its format strings.
- Link pages with relative `.md` paths (`../reference/cli.md#run`); they work on GitHub and on the site. Link files outside the site with full GitHub URLs.
- Code blocks name their language. Placeholders in angle brackets go inside backticks, or the site hides them as HTML.
- Use the models' Hugging Face names: Qwen3.8-27B-PN3 and Qwen3.8-27B-UD-IQ3_XXS-Q3Floor.
- The site is public: made-up host names in examples (`gpu-host-1`), and no private hosts, repositories or names. `npm run check` refuses old model names and home paths, and any term listed in `site/.private-terms`: a local file, one regular expression per line, that git ignores so the private names themselves are never published.

`npm run check` fails on a page whose type does not match its folder, a private or old name, a `tensorless` command or flag that the launcher does not have, and an install script that installs another release.

## Deploying

The site is static. Build it with `npm run build` in `site/` and serve `dist/`. On Cloudflare Pages: root directory `site`, build command `npm run build`, output directory `dist`, Node.js 22.22 or later.
