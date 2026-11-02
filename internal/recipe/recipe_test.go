package recipe

import (
	"io/fs"
	"os"
	"reflect"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/iammrduncan/tensorless"
)

const good = `
schema = 1
status = "released"

[model]
name       = "m1"
title      = "Model one"
hf_repo    = "org/repo"
revision   = "0123456789abcdef0123456789abcdef01234567"
file       = "dir/m1.gguf"
sha256     = "1111111111111111111111111111111111111111111111111111111111111111"
size_bytes = 1234
license    = "apache-2.0"

[card]
id                 = "p100"
title              = "NVIDIA Tesla P100 16 GB"
compute_capability = "6.0"
min_memory_mib     = 16000
min_driver         = "525.60.13"

[engine]
release = "engine-v0.1.0"
arch    = "sm60"
image   = "ghcr.io/x/y@sha256:2222222222222222222222222222222222222222222222222222222222222222"

[run]
alias       = "org/repo"
context     = 4096
threads_max = 8
args        = ["-ngl", "99"]
env         = { A = "1", B = "2" }

[receipt]
decode_tps = { short = 30.0, ctx_32k = 20.0 }
`

func TestParseGood(t *testing.T) {
	r, err := Parse([]byte(good), "good.toml")
	if err != nil {
		t.Fatal(err)
	}
	if r.Model.File != "dir/m1.gguf" || r.Model.SizeBytes != 1234 || r.Card.MinMemoryMiB != 16000 {
		t.Errorf("unexpected decode: %+v", r)
	}
	if !reflect.DeepEqual(r.EnvKeys(), []string{"A", "B"}) {
		t.Errorf("env keys = %v", r.EnvKeys())
	}
	if !r.Published() || r.Draft() {
		t.Errorf("published=%v draft=%v", r.Published(), r.Draft())
	}
	if got := MeasurementKeys(r.Receipt.DecodeTPS); !reflect.DeepEqual(got, []string{"short", "ctx_32k"}) {
		t.Errorf("measurement keys = %v", got)
	}
}

func TestParseInvalid(t *testing.T) {
	tests := []struct {
		name, old, new, want string
	}{
		{"schema", `schema = 1`, `schema = 2`, "schema = 2"},
		{"status", `status = "released"`, `status = "beta"`, "status must be"},
		{"short revision", `revision   = "0123456789abcdef0123456789abcdef01234567"`, `revision = "main"`, "model.revision"},
		{"upper revision", `revision   = "0123456789abcdef0123456789abcdef01234567"`, `revision = "0123456789ABCDEF0123456789abcdef01234567"`, "model.revision"},
		{"short sha", `sha256     = "1111111111111111111111111111111111111111111111111111111111111111"`, `sha256 = "1111"`, "model.sha256"},
		{"image tag", `image   = "ghcr.io/x/y@sha256:2222222222222222222222222222222222222222222222222222222222222222"`, `image = "ghcr.io/x/y:latest"`, "engine.image"},
		{"image short digest", `image   = "ghcr.io/x/y@sha256:2222222222222222222222222222222222222222222222222222222222222222"`, `image = "ghcr.io/x/y@sha256:2222"`, "engine.image"},
		{"bad repo", `hf_repo    = "org/repo"`, `hf_repo = "repo"`, "model.hf_repo"},
		{"escaping file", `file       = "dir/m1.gguf"`, `file = "../m1.gguf"`, "model.file"},
		{"absolute file", `file       = "dir/m1.gguf"`, `file = "/m1.gguf"`, "model.file"},
		{"negative size", `size_bytes = 1234`, `size_bytes = -1`, "size_bytes"},
		{"compute", `compute_capability = "6.0"`, `compute_capability = "sm60"`, "compute_capability"},
		{"driver", `min_driver         = "525.60.13"`, `min_driver = "r525"`, "min_driver"},
		{"memory", `min_memory_mib     = 16000`, `min_memory_mib = 0`, "min_memory_mib"},
		{"managed arg", `args        = ["-ngl", "99"]`, `args = ["-ngl", "99", "-c", "8192"]`, "must not set -c"},
		{"managed arg with =", `args        = ["-ngl", "99"]`, `args = ["--port=9"]`, "must not set --port"},
		{"omp env", `env         = { A = "1", B = "2" }`, `env = { OMP_NUM_THREADS = "4" }`, "OMP_NUM_THREADS"},
		{"alias", `alias       = "org/repo"`, `alias = "org repo"`, "run.alias"},
		{"bad name", `name       = "m1"`, `name = "M 1"`, "model.name"},
		{"context", `context     = 4096`, `context = 0`, "run.context"},
		{"unknown key", `license    = "apache-2.0"`, "license = \"x\"\nlicence = \"y\"", "unknown keys: model.licence"},
		{"no license", `license    = "apache-2.0"`, ``, "model.license"},
		{"wrong type", `context     = 4096`, `context = "4096"`, "context"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := strings.Replace(good, tc.old, tc.new, 1)
			if src == good {
				t.Fatalf("test did not change the recipe")
			}
			_, err := Parse([]byte(src), "bad.toml")
			if err == nil {
				t.Fatalf("expected an error containing %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "bad.toml") {
				t.Errorf("error %q does not contain %q and the source", err, tc.want)
			}
		})
	}
}

func TestUnpublishedFixtures(t *testing.T) {
	s, err := Load(os.DirFS("testdata/placeholders"), "testdata")
	if err != nil {
		t.Fatal(err) // placeholders are valid values
	}
	want := map[string][]string{
		"Org/AllZero-GGUF": {"revision", "sha256", "image digest", "size_bytes"},
		"Org/Partial-GGUF": {"revision"},
	}
	if len(s.Recipes) != len(want) {
		t.Fatalf("loaded %d recipes", len(s.Recipes))
	}
	for _, r := range s.Recipes {
		if got := r.Unpublished(); !reflect.DeepEqual(got, want[r.Repo()]) {
			t.Errorf("%s: Unpublished() = %v, want %v", r.Repo(), got, want[r.Repo()])
		}
		if r.Published() || !r.Draft() {
			t.Errorf("%s: published %v, draft %v", r.Repo(), r.Published(), r.Draft())
		}
	}
}

func embedded(t *testing.T) *Set {
	t.Helper()
	sub, err := fs.Sub(tensorless.Recipes, "recipes")
	if err != nil {
		t.Fatal(err)
	}
	s, err := Load(sub, "recipes")
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// placeholders lists the placeholder fields of r, worked out from its values
// rather than by Unpublished.
func placeholders(r *Recipe) []string {
	var out []string
	if r.Model.Revision == strings.Repeat("0", 40) {
		out = append(out, "revision")
	}
	if r.Model.SHA256 == strings.Repeat("0", 64) {
		out = append(out, "sha256")
	}
	if strings.HasSuffix(r.Engine.Image, "@sha256:"+strings.Repeat("0", 64)) {
		out = append(out, "image digest")
	}
	if r.Model.SizeBytes == 0 {
		out = append(out, "size_bytes")
	}
	return out
}

// TestEmbeddedRecipes checks every built-in recipe, whatever they are.
func TestEmbeddedRecipes(t *testing.T) {
	s := embedded(t)
	for _, r := range s.Recipes {
		if want := "recipes/" + r.Repo() + "/" + r.Card.ID + ".toml"; r.Source != want {
			t.Errorf("source = %q, want %q", r.Source, want)
		}
		if r.Run.Alias != r.Repo() {
			t.Errorf("%s: the served alias %q is not the repo id", r.Source, r.Run.Alias)
		}
		if !contains(s.Models(), r.Repo()) {
			t.Errorf("Models() = %v lacks %s", s.Models(), r.Repo())
		}
		got, want := r.Unpublished(), placeholders(r)
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("%s: Unpublished() = %v, want %v", r.Source, got, want)
		}
		if r.Published() != (len(want) == 0) {
			t.Errorf("%s: Published() = %v", r.Source, r.Published())
		}
		if r.Run.Context <= 0 || r.Run.ThreadsMax <= 0 {
			t.Errorf("%s: run = %+v", r.Source, r.Run)
		}
	}
}

func TestResolveByRepoIDOrShortName(t *testing.T) {
	s := embedded(t)
	for _, r := range s.Recipes {
		repo := r.Repo()
		names := []string{repo, strings.ToLower(repo), strings.ToUpper(repo)}
		if r.Model.Name != "" {
			names = append(names, r.Model.Name, strings.ToUpper(r.Model.Name))
		}
		for _, m := range names {
			rs, err := s.ForModel(m)
			if err != nil || len(rs) == 0 {
				t.Errorf("ForModel(%q) = %v, %v", m, rs, err)
				continue
			}
			for _, got := range rs {
				if got.Repo() != repo {
					t.Errorf("ForModel(%q) returned %s", m, got.Repo())
				}
			}
		}
		org, name, _ := strings.Cut(repo, "/")
		for _, m := range []string{repo[:len(repo)-1], name, "not-" + org + "/" + name} {
			if _, err := s.ForModel(m); err == nil {
				t.Errorf("ForModel(%q) should fail", m)
			}
		}
	}
}

func TestSplitModel(t *testing.T) {
	const p100Repo = "iamMrDuncan/Qwen3.8-27B-PN3-GGUF" // any repo id
	rev := "0123456789abcdef0123456789abcdef01234567"
	tests := []struct{ in, model, rev string }{
		{p100Repo, p100Repo, ""},
		{p100Repo + "@" + rev, p100Repo, rev},
		{p100Repo + "@" + strings.ToUpper(rev), p100Repo, rev},
		{"pn3m19", "pn3m19", ""},
	}
	for _, tc := range tests {
		m, r, err := SplitModel(tc.in)
		if err != nil || m != tc.model || r != tc.rev {
			t.Errorf("SplitModel(%q) = %q, %q, %v", tc.in, m, r, err)
		}
	}
	for _, bad := range []string{p100Repo + "@main", p100Repo + "@", "@" + rev, p100Repo + "@abc"} {
		if _, _, err := SplitModel(bad); err == nil {
			t.Errorf("SplitModel(%q) should fail", bad)
		}
	}
}

func TestLoadIgnoresOtherFilesAndChecksNames(t *testing.T) {
	one := strings.Replace(good, `file       = "dir/m1.gguf"`, `file = "m1.gguf"`, 1)
	fsys := fstest.MapFS{
		"README.md":              {Data: []byte("# recipes")},
		"org/repo/p100.toml":     {Data: []byte(one)},
		"org/repo/notes.md":      {Data: []byte("notes")},
		"org/repo/old/x.toml":    {Data: []byte("not a recipe")},
		"org/stray.toml":         {Data: []byte("not a recipe")},
		"stray.toml":             {Data: []byte("not a recipe")},
		".hidden/repo/x.toml":    {Data: []byte("not a recipe")},
		"org/repo/.draft.toml":   {Data: []byte("not a recipe")},
		"empty/README.md":        {Data: []byte("")},
		"empty/also/README.md":   {Data: []byte("")},
		"pn3m19/p100.toml/x.txt": {Data: []byte("")},
	}
	s, err := Load(fsys, "dir")
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Recipes) != 1 || s.Recipes[0].Source != "dir/org/repo/p100.toml" {
		t.Fatalf("recipes = %+v", s.Recipes)
	}

	fsys = fstest.MapFS{"org/other/p100.toml": {Data: []byte(good)}}
	if _, err := Load(fsys, "dir"); err == nil || !strings.Contains(err.Error(), "must match its directory") {
		t.Errorf("repo/dir mismatch: %v", err)
	}
	fsys = fstest.MapFS{"Org/Repo/p100.toml": {Data: []byte(good)}}
	if _, err := Load(fsys, "dir"); err == nil || !strings.Contains(err.Error(), "must match its directory") {
		t.Errorf("a directory must keep the repo id's case: %v", err)
	}
	fsys = fstest.MapFS{"org/repo/v100.toml": {Data: []byte(good)}}
	if _, err := Load(fsys, "dir"); err == nil || !strings.Contains(err.Error(), "must match its file name") {
		t.Errorf("card/file mismatch: %v", err)
	}
	other := strings.ReplaceAll(good, "org/repo", "org/repo2")
	fsys = fstest.MapFS{"org/repo/p100.toml": {Data: []byte(good)}, "org/repo2/p100.toml": {Data: []byte(other)}}
	if _, err := Load(fsys, "dir"); err == nil || !strings.Contains(err.Error(), "already the short name") {
		t.Errorf("a short name shared by two repos: %v", err)
	}
	if _, err := Load(fstest.MapFS{"README.md": {}}, "dir"); err == nil || !strings.Contains(err.Error(), "no recipes found") {
		t.Errorf("empty: %v", err)
	}
}

func TestLookupErrorsListWhatExists(t *testing.T) {
	s := embedded(t)
	_, err := s.Lookup("nobody/nothing", "p100")
	if err == nil || !strings.Contains(err.Error(), "Known models: "+strings.Join(s.Models(), ", ")) {
		t.Errorf("unknown model: %v", err)
	}
	r := s.Recipes[0]
	rs, _ := s.ForModel(r.Repo())
	var cards []string
	for _, x := range rs {
		cards = append(cards, x.Card.ID)
	}
	_, err = s.Lookup(strings.ToLower(r.Repo()), "no-such-card")
	if err == nil || !strings.Contains(err.Error(), r.Repo()+" has recipes for: "+strings.Join(cards, ", ")) ||
		!strings.Contains(err.Error(), "Known cards: "+strings.Join(s.Cards(), ", ")) {
		t.Errorf("unknown card: %v", err)
	}
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}
