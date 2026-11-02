// Package tensorless holds the files that the tensorless launcher ships with.
// The launcher itself is in cmd/tensorless and internal/.
package tensorless

import "embed"

// Recipes is the recipes directory, embedded at build time. The launcher reads
// recipes/<org>/<repo>/<card>.toml from it and ignores every other file.
//
//go:embed recipes
var Recipes embed.FS

// License is the Apache-2.0 license text. tensorless bench builds its prompt
// from it, so the prompt is the same on every host.
//
//go:embed LICENSE
var License string
