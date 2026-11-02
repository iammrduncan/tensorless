---
title: "Contributing"
description: "How to help: report problems, share benchmarks, propose recipes, and change the launcher, the engine or these docs."
type: project
sidebar:
  order: 2
---

There are many ways to help tensorless, and most of them don't need any code. This page says what each kind of contribution needs, so that it can be used.

Contributions are made under the project's license, Apache-2.0 ([License](license.md)).

## Report a problem

Open an [issue](https://github.com/iammrduncan/tensorless/issues) with:

- what you ran, the whole command line;
- what happened, with the error message as printed;
- what `tensorless version` and `tensorless doctor` print;
- your card, driver version, distribution, and podman or docker.

A wrong number, a broken link and a confusing page are all problems worth reporting.

## Share a benchmark

Results from other people's cards are how a receipt gets checked. Run `tensorless bench` on a tuned run (no `--context`, no extra arguments), and post the output on [Discord](https://discord.gg/3Qs2uejUf9) or in an issue. [Share your results](../how-to/use/share-results.md) lists what to include.

A result that is far from the receipt is the most useful kind. Please share those too.

## Propose a recipe

A recipe is a promise: this model, on this card, with these settings, measured this fast. So a new recipe, or a change to an existing one, needs evidence:

- **A new recipe** needs the model on Hugging Face at a fixed revision, its sha256, and a receipt: what you measured, on which card, with which image digest, and how.
- **A change to a recipe's `[run]` or `[engine]` section** needs a new receipt, because the old numbers no longer apply.
- **A change to other fields,** such as the title or the license text, needs no measurement.

You can try a recipe on your own machine before proposing it: [Try a recipe of your own](../how-to/advanced/write-a-recipe.md). Recipes are reviewed like code. Open an issue or say hello on Discord first if you are planning a new model or card, so the work fits what is already under way.

## Change the launcher

The launcher is Go, in `cmd/` and `internal/`, with the recipes embedded from `recipes/`. Every pull request that touches it runs the same checks as the `launcher` workflow:

```sh
test -z "$(gofmt -l cmd internal *.go)"
go vet ./...
go test ./...
```

Run them before you open the pull request. Keep the launcher's rules: it holds no inference code, it pins what it downloads, and it refuses with a reason rather than guessing. [How tensorless works](../explanation/how-tensorless-works.md) explains them.

## Change the engine

The engine in `engine/` is not edited in place. It is developed in a separate research repository and exported into tensorless at each release, as one snapshot of the exact commit that was measured ([Where the engine comes from](../explanation/the-engine.md)). A pull request against `engine/` would be replaced by the next export.

So for an engine change, open an issue first, or bring it to Discord: a bug, a kernel idea, a measurement that disagrees with a receipt. Engine changes reach tensorless through the research line and a release, measured under the same guards as everything else.

## Improve these docs

The pages are Markdown files in the repository's `site/` folder, and every page has an edit link at the bottom. Each page is one of four types: a tutorial, a how-to guide, reference or explanation. A change keeps a page to its type, and describes only what exists in the code today.
