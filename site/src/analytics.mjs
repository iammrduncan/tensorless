// Fathom Analytics, site tensorless.dev: one script on every page, the home page
// (src/layouts/HomeLayout.astro) and the docs (Starlight's `head` in astro.config.mjs).
// Only tensorless.dev is counted, so pages.dev deployments and local previews are not.
// The script loads from cdn.usefathom.com and sends page views there; public/_headers allows that host.
export const fathom = {
  src: "https://cdn.usefathom.com/script.js",
  "data-site": "ELKPCFMN",
  "data-included-domains": "tensorless.dev",
  defer: true,
};
