// Package recipe reads and checks tensorless recipes: one model on one card,
// recipes/<org>/<repo>/<card>.toml, schema 1 (see recipes/README.md). A model
// is named by its Hugging Face repo id. Its weights are either one file
// (model.file) or a pack: several files of the repository that the engine
// reads in place, after a prepare step ([model.pack]).
package recipe

import (
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"

	"github.com/BurntSushi/toml"
)

// Schema is the recipe schema this launcher reads.
const Schema = 1

// Recipe statuses.
const (
	StatusDraft    = "draft"
	StatusReleased = "released"
)

// Recipe is one model on one card.
type Recipe struct {
	Schema  int     `toml:"schema" json:"schema"`
	Status  string  `toml:"status" json:"status"`
	Model   Model   `toml:"model" json:"model"`
	Card    Card    `toml:"card" json:"card"`
	Engine  Engine  `toml:"engine" json:"engine"`
	Run     Run     `toml:"run" json:"run"`
	Receipt Receipt `toml:"receipt" json:"receipt"`

	// Source says where the recipe was read from.
	Source string `toml:"-" json:"source"`
}

// Model is the [model] table. It names one file (File, SHA256, SizeBytes) or
// a pack (Pack), never both.
type Model struct {
	Name      string `toml:"name" json:"name"`
	Title     string `toml:"title" json:"title"`
	HFRepo    string `toml:"hf_repo" json:"hf_repo"`
	Revision  string `toml:"revision" json:"revision"`
	File      string `toml:"file" json:"file,omitempty"`
	SHA256    string `toml:"sha256" json:"sha256,omitempty"`
	SizeBytes int64  `toml:"size_bytes" json:"size_bytes"`
	License   string `toml:"license" json:"license"`
	Pack      *Pack  `toml:"pack" json:"pack,omitempty"`
}

// Pack is the [model.pack] table: files of the repository, at the model's
// revision, that the engine reads in place, and the command that prepares
// them. The launcher runs Prepare in the engine image with two more
// arguments, the pack's folder and an empty output folder; the last line it
// prints is the file llama-server is given as -m.
type Pack struct {
	Prepare []string   `toml:"prepare" json:"prepare"`
	Files   []PackFile `toml:"files" json:"files"`
}

// PackFile is one file of a pack.
type PackFile struct {
	Name      string `toml:"name" json:"name"`
	SHA256    string `toml:"sha256" json:"sha256"`
	SizeBytes int64  `toml:"size_bytes" json:"size_bytes"`
}

// Card is the [card] table.
type Card struct {
	ID                string `toml:"id" json:"id"`
	Title             string `toml:"title" json:"title"`
	ComputeCapability string `toml:"compute_capability" json:"compute_capability"`
	MinMemoryMiB      int    `toml:"min_memory_mib" json:"min_memory_mib"`
	MinDriver         string `toml:"min_driver" json:"min_driver"`
}

// Engine is the [engine] table.
type Engine struct {
	Release string `toml:"release" json:"release"`
	Arch    string `toml:"arch" json:"arch"`
	Image   string `toml:"image" json:"image"`
}

// Run is the [run] table.
type Run struct {
	Alias      string            `toml:"alias" json:"alias"`
	Context    int               `toml:"context" json:"context"`
	ThreadsMax int               `toml:"threads_max" json:"threads_max"`
	Args       []string          `toml:"args" json:"args"`
	Env        map[string]string `toml:"env" json:"env"`
}

// Receipt is the [receipt] table. Every field is optional.
type Receipt struct {
	Source     string             `toml:"source" json:"source,omitempty"`
	Summary    string             `toml:"summary" json:"summary,omitempty"`
	DecodeTPS  map[string]float64 `toml:"decode_tps" json:"decode_tps,omitempty"`
	PrefillTPS map[string]float64 `toml:"prefill_tps" json:"prefill_tps,omitempty"`
	Notes      []string           `toml:"notes" json:"notes,omitempty"`
}

// ManagedFlags are the llama-server flags the launcher sets itself. A recipe's
// args and the extra arguments given after "--" must not contain them.
var ManagedFlags = []string{
	"-m", "--model",
	"--host", "--port",
	"-c", "--ctx-size",
	"-a", "--alias",
	"-t", "--threads",
	"-tb", "--threads-batch",
}

// ManagedFlag returns the managed flag that arg sets, or "".
func ManagedFlag(arg string) string {
	for _, f := range ManagedFlags {
		if arg == f || strings.HasPrefix(arg, f+"=") {
			return f
		}
	}
	return ""
}

var (
	reName     = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
	reRepo     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
	reRevision = regexp.MustCompile(`^[0-9a-f]{40}$`)
	reSHA256   = regexp.MustCompile(`^[0-9a-f]{64}$`)
	reImage    = regexp.MustCompile(`^[^\s@]+@sha256:[0-9a-f]{64}$`)
	reCompute  = regexp.MustCompile(`^[0-9]+\.[0-9]+$`)
	reVersion  = regexp.MustCompile(`^[0-9]+(\.[0-9]+)*$`)
	reAlias    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_./-]*$`)
	reEnvKey   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

// Parse decodes and validates one recipe. source names it in error messages.
func Parse(data []byte, source string) (*Recipe, error) {
	var r Recipe
	md, err := toml.Decode(string(data), &r)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", source, err)
	}
	if r.Schema != Schema {
		return nil, fmt.Errorf("%s: schema = %d, but this launcher reads schema %d; update tensorless", source, r.Schema, Schema)
	}
	if und := md.Undecoded(); len(und) > 0 {
		keys := make([]string, len(und))
		for i, k := range und {
			keys[i] = k.String()
		}
		return nil, fmt.Errorf("%s: unknown keys: %s", source, strings.Join(keys, ", "))
	}
	if r.Model.Pack != nil {
		var both []string
		for _, k := range []string{"file", "sha256", "size_bytes"} {
			if md.IsDefined("model", k) {
				both = append(both, "model."+k)
			}
		}
		if len(both) > 0 {
			return nil, fmt.Errorf("%s: %s and [model.pack] cannot both be set: a recipe names one file or a pack", source, strings.Join(both, ", "))
		}
	}
	r.Source = source
	if err := r.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", source, err)
	}
	return &r, nil
}

// Validate checks the recipe against schema 1. It does not reject placeholder
// values; see Unpublished.
func (r *Recipe) Validate() error {
	var p []string
	bad := func(format string, args ...any) { p = append(p, fmt.Sprintf(format, args...)) }

	if r.Schema != Schema {
		bad("schema must be %d", Schema)
	}
	if r.Status != StatusDraft && r.Status != StatusReleased {
		bad("status must be %q or %q, not %q", StatusDraft, StatusReleased, r.Status)
	}

	m := r.Model
	if m.Name != "" && !reName.MatchString(m.Name) {
		bad("model.name %q (an optional short alias) must be lowercase letters, digits, '.', '_' or '-'", m.Name)
	}
	if !reRepo.MatchString(m.HFRepo) {
		bad("model.hf_repo %q must look like org/name", m.HFRepo)
	}
	if !reRevision.MatchString(m.Revision) {
		bad("model.revision must be a commit: 40 lowercase hex characters, not %q", m.Revision)
	}
	if m.Pack == nil {
		if err := checkFile(m.File); err != "" {
			bad("model.file %q %s", m.File, err)
		}
		if !reSHA256.MatchString(m.SHA256) {
			bad("model.sha256 must be 64 lowercase hex characters, not %q", m.SHA256)
		}
		if m.SizeBytes < 0 {
			bad("model.size_bytes must not be negative")
		}
	} else {
		if m.File != "" || m.SHA256 != "" || m.SizeBytes != 0 {
			bad("model.file, model.sha256 and model.size_bytes must be left out when [model.pack] names the files")
		}
		m.Pack.validate(bad)
	}
	if strings.TrimSpace(m.License) == "" {
		bad("model.license is required")
	}

	c := r.Card
	if !reName.MatchString(c.ID) {
		bad("card.id %q must be lowercase letters, digits, '.', '_' or '-'", c.ID)
	}
	if !reCompute.MatchString(c.ComputeCapability) {
		bad("card.compute_capability must look like 6.0, not %q", c.ComputeCapability)
	}
	if c.MinMemoryMiB <= 0 {
		bad("card.min_memory_mib must be positive")
	}
	if !reVersion.MatchString(c.MinDriver) {
		bad("card.min_driver must be a dotted version such as 525.60.13, not %q", c.MinDriver)
	}

	e := r.Engine
	if strings.TrimSpace(e.Release) == "" {
		bad("engine.release is required")
	}
	if strings.TrimSpace(e.Arch) == "" {
		bad("engine.arch is required")
	}
	if !reImage.MatchString(e.Image) {
		bad("engine.image must be pinned by digest (name@sha256:<64 hex>), not %q", e.Image)
	}

	run := r.Run
	if !reAlias.MatchString(run.Alias) {
		bad("run.alias %q must be letters, digits, '.', '_', '-' or '/', such as the repo id", run.Alias)
	}
	if run.Context <= 0 {
		bad("run.context must be positive")
	}
	if run.ThreadsMax <= 0 {
		bad("run.threads_max must be positive")
	}
	for _, a := range run.Args {
		if f := ManagedFlag(a); f != "" {
			bad("run.args must not set %s; the launcher sets it", f)
		}
	}
	for _, k := range sortedKeys(run.Env) {
		if !reEnvKey.MatchString(k) {
			bad("run.env key %q is not a valid variable name", k)
		}
		if k == "OMP_NUM_THREADS" {
			bad("run.env must not set OMP_NUM_THREADS; the launcher sets it")
		}
	}

	for k, v := range r.Receipt.DecodeTPS {
		if v < 0 {
			bad("receipt.decode_tps.%s must not be negative", k)
		}
	}
	for k, v := range r.Receipt.PrefillTPS {
		if v < 0 {
			bad("receipt.prefill_tps.%s must not be negative", k)
		}
	}

	if len(p) == 0 {
		return nil
	}
	return fmt.Errorf("invalid recipe:\n  - %s", strings.Join(p, "\n  - "))
}

// MaxPackFiles is the most files a pack may name.
const MaxPackFiles = 4096

func (pk *Pack) validate(bad func(string, ...any)) {
	if len(pk.Prepare) == 0 {
		bad("model.pack.prepare is required: the command, run in the engine image, that prepares the pack")
	}
	for i, a := range pk.Prepare {
		switch {
		case a == "":
			bad("model.pack.prepare[%d] must not be empty", i)
		case strings.ContainsAny(a, "\x00\n\r"):
			bad("model.pack.prepare[%d] must not contain newlines or NUL", i)
		}
	}
	switch {
	case len(pk.Files) == 0:
		bad("model.pack.files must name at least one file")
	case len(pk.Files) > MaxPackFiles:
		bad("model.pack.files names %d files; at most %d are allowed", len(pk.Files), MaxPackFiles)
	}
	seen := map[string]string{} // lowercased name -> name
	for i, f := range pk.Files {
		if err := checkFile(f.Name); err != "" {
			bad("model.pack.files[%d].name %q %s", i, f.Name, err)
		} else if other, ok := seen[strings.ToLower(f.Name)]; ok {
			if other == f.Name {
				bad("model.pack.files names %q twice", f.Name)
			} else {
				bad("model.pack.files names %q and %q, which differ only in case", other, f.Name)
			}
		} else {
			seen[strings.ToLower(f.Name)] = f.Name
		}
		if !reSHA256.MatchString(f.SHA256) {
			bad("model.pack.files[%d].sha256 (%s) must be 64 lowercase hex characters, not %q", i, f.Name, f.SHA256)
		}
		if f.SizeBytes < 0 {
			bad("model.pack.files[%d].size_bytes (%s) must not be negative", i, f.Name)
		}
	}
	// A name cannot be both a file and a folder of other files. Only clean,
	// relative names reach this loop, so path.Dir ends at ".".
	for _, f := range pk.Files {
		if checkFile(f.Name) != "" {
			continue
		}
		for dir := path.Dir(f.Name); dir != "."; dir = path.Dir(dir) {
			if other, ok := seen[strings.ToLower(dir)]; ok {
				bad("model.pack.files names %q, but %q is inside it", other, f.Name)
			}
		}
	}
}

func checkFile(f string) string {
	switch {
	case f == "":
		return "is required"
	case strings.HasPrefix(f, "/"), strings.Contains(f, `\`):
		return "must be a relative path with forward slashes"
	case path.Clean(f) != f || f == "." || strings.HasPrefix(f, "../") || f == "..":
		return "must be a clean path inside the repository"
	case strings.ContainsAny(f, ":\n"):
		return "must not contain ':' or newlines"
	}
	return ""
}

// Unpublished lists the fields that still hold a placeholder: an all-zero
// revision, sha256 or image digest, or a size_bytes of 0. In a pack, one
// file's placeholder is enough. A recipe with any of them cannot be pulled or
// run.
func (r *Recipe) Unpublished() []string {
	var zeroSHA, zeroSize bool
	for _, f := range r.Files() {
		zeroSHA = zeroSHA || allZero(f.SHA256)
		zeroSize = zeroSize || f.SizeBytes == 0
	}
	var out []string
	if allZero(r.Model.Revision) {
		out = append(out, "revision")
	}
	if zeroSHA {
		out = append(out, "sha256")
	}
	if allZero(r.ImageDigest()) {
		out = append(out, "image digest")
	}
	if zeroSize {
		out = append(out, "size_bytes")
	}
	return out
}

// IsPack reports whether the model is a pack rather than one file.
func (r *Recipe) IsPack() bool { return r.Model.Pack != nil }

// Files returns the model's files: the pack's, or the one file as a
// one-element list.
func (r *Recipe) Files() []PackFile {
	if r.Model.Pack != nil {
		return r.Model.Pack.Files
	}
	return []PackFile{{Name: r.Model.File, SHA256: r.Model.SHA256, SizeBytes: r.Model.SizeBytes}}
}

// TotalBytes is the sum of the model's file sizes.
func (r *Recipe) TotalBytes() int64 {
	var n int64
	for _, f := range r.Files() {
		n += f.SizeBytes
	}
	return n
}

// Published reports whether every pinned value is real.
func (r *Recipe) Published() bool { return len(r.Unpublished()) == 0 }

// Draft reports whether the recipe's status is draft.
func (r *Recipe) Draft() bool { return r.Status == StatusDraft }

// ImageDigest returns the 64 hex characters after "@sha256:" in the image.
func (r *Recipe) ImageDigest() string {
	_, d, ok := strings.Cut(r.Engine.Image, "@sha256:")
	if !ok {
		return ""
	}
	return d
}

// ID is "org/repo/card", the recipe's path under recipes/ without ".toml".
func (r *Recipe) ID() string { return r.Model.HFRepo + "/" + r.Card.ID }

// Repo is the model's Hugging Face repo id, the name users type.
func (r *Recipe) Repo() string { return r.Model.HFRepo }

func allZero(s string) bool {
	return s != "" && strings.Trim(s, "0") == ""
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// EnvKeys returns the recipe's env keys in sorted order.
func (r *Recipe) EnvKeys() []string { return sortedKeys(r.Run.Env) }

// receiptOrder is the order receipt measurements are shown in; other keys
// follow alphabetically.
var receiptOrder = []string{"short", "ctx_4k", "ctx_32k", "ctx_65k"}

// MeasurementKeys returns the keys of a receipt measurement table in display
// order.
func MeasurementKeys(m map[string]float64) []string {
	var out []string
	for _, k := range receiptOrder {
		if _, ok := m[k]; ok {
			out = append(out, k)
		}
	}
	for _, k := range sortedKeys(m) {
		known := false
		for _, o := range receiptOrder {
			known = known || k == o
		}
		if !known {
			out = append(out, k)
		}
	}
	return out
}
