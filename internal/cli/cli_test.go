package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/iammrduncan/tensorless"
	"github.com/iammrduncan/tensorless/internal/container"
	"github.com/iammrduncan/tensorless/internal/execx"
	"github.com/iammrduncan/tensorless/internal/hfcache"
	"github.com/iammrduncan/tensorless/internal/recipe"
)

const p100GPU = "0, Tesla P100-PCIE-16GB, 6.0, 16384, 580.95.05, GPU-1a2b3c4d\n"

// syncBuffer is a bytes.Buffer safe for the log-following goroutine.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// host fakes nvidia-smi, podman and docker.
type host struct {
	mu           sync.Mutex
	gpus         string // nvidia-smi output
	imagePresent bool
	running      map[string]bool
	logs         string
	exitAtStart  bool // the container dies right after run
	dockerInfo   string
}

func (h *host) fn(ctx context.Context, name string, args []string, stdout, stderr io.Writer) error {
	h.mu.Lock()
	if name == "nvidia-smi" {
		defer h.mu.Unlock()
		_, err := io.WriteString(stdout, h.gpus)
		return err
	}
	if name != "podman" && name != "docker" {
		h.mu.Unlock()
		return nil
	}
	switch args[0] {
	case "image":
		defer h.mu.Unlock()
		if h.imagePresent {
			return nil
		}
		io.WriteString(stderr, "no such image")
		return errors.New("exit status 125")
	case "pull":
		h.imagePresent = true
	case "container":
		defer h.mu.Unlock()
		if h.running[args[len(args)-1]] {
			io.WriteString(stdout, "true\n")
			return nil
		}
		return errors.New("no such container")
	case "run":
		for i, a := range args {
			if a == "--name" {
				h.running[args[i+1]] = !h.exitAtStart
			}
		}
		io.WriteString(stdout, "c0ffee\n")
	case "logs":
		logs, exit := h.logs, h.exitAtStart
		h.mu.Unlock()
		io.WriteString(stderr, logs)
		if exit {
			return nil
		}
		<-ctx.Done()
		return ctx.Err()
	case "stop":
		h.running[args[1]] = false
	case "info":
		io.WriteString(stdout, h.dockerInfo)
	}
	h.mu.Unlock()
	return nil
}

func (h *host) calls(f *execx.Fake, prefix string) []string {
	var out []string
	for _, c := range f.Calls() {
		if strings.HasPrefix(c, prefix) {
			out = append(out, c)
		}
	}
	return out
}

type testApp struct {
	*App
	fake   *execx.Fake
	host   *host
	stdout *syncBuffer
	stderr *syncBuffer
	env    map[string]string
	files  map[string]bool
	dir    string
}

// newApp returns an App on a fake host that has the given executables.
func newApp(t *testing.T, found ...string) *testApp {
	t.Helper()
	recipes, err := fs.Sub(tensorless.Recipes, "recipes")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	h := &host{running: map[string]bool{}}
	f := execx.NewFake(found...)
	f.Func = h.fn
	ta := &testApp{fake: f, host: h, stdout: &syncBuffer{}, stderr: &syncBuffer{}, dir: dir, files: map[string]bool{},
		env: map[string]string{"HF_HUB_CACHE": filepath.Join(dir, "hub"), "XDG_STATE_HOME": filepath.Join(dir, "state")}}
	ta.App = &App{
		Stdout:         ta.stdout,
		Stderr:         ta.stderr,
		Runner:         f,
		Getenv:         func(k string) string { return ta.env[k] },
		Home:           func() (string, error) { return dir, nil },
		HTTP:           http.DefaultClient,
		Recipes:        recipes,
		License:        tensorless.License,
		NumCPU:         func() int { return 6 },
		FileExists:     func(p string) bool { return ta.files[p] },
		FreeSpace:      func(string) (uint64, error) { return 1 << 40, nil },
		PortFree:       func(string, int) error { return nil },
		Version:        "v9.9.9",
		Commit:         "abc1234",
		HealthInterval: 10 * time.Millisecond,
		HealthTimeout:  5 * time.Second,
	}
	return ta
}

func (ta *testApp) run(args ...string) int {
	return ta.Run(context.Background(), args)
}

// published writes a released recipe for org/model on p100, whose model
// file is data, and returns the recipes directory.
func (ta *testApp) published(t *testing.T, data []byte) string {
	t.Helper()
	sum := sha256.Sum256(data)
	rec := fmt.Sprintf(`schema = 1
status = "released"
[model]
name = "m1"
hf_repo = "Org/Model-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "m.gguf"
sha256 = "%s"
size_bytes = %d
license = "apache-2.0"
[card]
id = "p100"
title = "NVIDIA Tesla P100 16 GB"
compute_capability = "6.0"
min_memory_mib = 16000
min_driver = "525.60.13"
[engine]
release = "engine-v0.1.0"
arch = "sm60"
image = "ghcr.io/x/engine@sha256:%s"
[run]
alias = "Org/Model-GGUF"
context = 4096
threads_max = 8
args = ["-ngl", "99"]
env = { GGML_X = "1" }
[receipt]
decode_tps = { short = 30.0 }
`, hex.EncodeToString(sum[:]), len(data), strings.Repeat("a", 64))
	dir := filepath.Join(ta.dir, "recipes")
	os.MkdirAll(filepath.Join(dir, "Org", "Model-GGUF"), 0o755)
	if err := os.WriteFile(filepath.Join(dir, "Org", "Model-GGUF", "p100.toml"), []byte(rec), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// cache puts data in the model cache as the published recipe's blob.
func (ta *testApp) cache(t *testing.T, data []byte) string {
	sum := sha256.Sum256(data)
	blob := filepath.Join(ta.env["HF_HUB_CACHE"], "models--Org--Model-GGUF", "blobs", hex.EncodeToString(sum[:]))
	os.MkdirAll(filepath.Dir(blob), 0o755)
	if err := os.WriteFile(blob, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return blob
}

// embedded returns every built-in recipe. Tests derive their expectations
// from these, so they keep passing as recipes are published or added.
func embedded(t *testing.T) []*recipe.Recipe {
	t.Helper()
	sub, err := fs.Sub(tensorless.Recipes, "recipes")
	if err != nil {
		t.Fatal(err)
	}
	set, err := recipe.Load(sub, "recipes")
	if err != nil {
		t.Fatal(err)
	}
	return set.Recipes
}

// placeholders lists r's placeholder fields, worked out from its values.
func placeholders(r *recipe.Recipe) []string {
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

// cardFlag pins r's card when its model has recipes for several cards, since
// a host without a GPU cannot choose between them.
func cardFlag(all []*recipe.Recipe, r *recipe.Recipe) []string {
	n := 0
	for _, x := range all {
		if x.Repo() == r.Repo() {
			n++
		}
	}
	if n > 1 {
		return []string{"--card", r.Card.ID}
	}
	return nil
}

// gpuLines is nvidia-smi output with one GPU per card the recipes name, each
// just meeting its card's requirements, on a new driver.
func gpuLines(rs []*recipe.Recipe) string {
	var b strings.Builder
	seen := map[string]bool{}
	for _, r := range rs {
		if seen[r.Card.ID] {
			continue
		}
		seen[r.Card.ID] = true
		fmt.Fprintf(&b, "%d, Fake %s, %s, %d, 999.0, GPU-%d\n", len(seen)-1, r.Card.ID, r.Card.ComputeCapability, r.Card.MinMemoryMiB, len(seen)-1)
	}
	return b.String()
}

// names returns every MODEL argument that must resolve to r.
func names(r *recipe.Recipe) []string {
	n := []string{r.Repo(), strings.ToLower(r.Repo()), strings.ToUpper(r.Repo())}
	if r.Model.Name != "" {
		n = append(n, r.Model.Name, strings.ToUpper(r.Model.Name))
	}
	return n
}

// flatten undoes shellWrap's line breaks.
func flatten(s string) string { return strings.ReplaceAll(s, " \\\n    ", " ") }

func TestCapabilitiesJSONWithNothingInstalled(t *testing.T) {
	all := embedded(t)
	ta := newApp(t) // no nvidia-smi, no runtime
	if code := ta.run("capabilities", "--json"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	var caps capsOut
	if err := json.Unmarshal([]byte(ta.stdout.String()), &caps); err != nil {
		t.Fatalf("%v\n%s", err, ta.stdout)
	}
	if caps.Version != "v9.9.9" || caps.Commit != "abc1234" || caps.RecipeSchema != 1 {
		t.Errorf("caps = %+v", caps)
	}
	if strings.Join(caps.Runtimes, ",") != "podman,docker" {
		t.Errorf("runtimes = %v", caps.Runtimes)
	}
	if len(caps.Recipes) != len(all) {
		t.Fatalf("%d recipes, want %d", len(caps.Recipes), len(all))
	}
	got := map[string]capRecipe{}
	for _, c := range caps.Recipes {
		got[c.Model+"/"+c.Card] = c
	}
	cards := map[string]bool{}
	for _, r := range all {
		c, ok := got[r.ID()]
		want := capRecipe{Model: r.Repo(), Name: r.Model.Name, Card: r.Card.ID, Status: r.Status,
			Published: len(placeholders(r)) == 0, Release: r.Engine.Release, Arch: r.Engine.Arch}
		if !ok || c != want {
			t.Errorf("recipe %s = %+v, want %+v", r.ID(), c, want)
		}
		cards[r.Card.ID] = true
	}
	if len(caps.Cards) != len(cards) {
		t.Errorf("cards = %+v", caps.Cards)
	}
	for _, c := range caps.Cards {
		if !cards[c.ID] {
			t.Errorf("unexpected card %s", c.ID)
		}
	}
	if calls := ta.fake.Calls(); len(calls) != 0 {
		t.Errorf("capabilities ran commands: %v", calls)
	}
}

func TestInfoWithNoGPU(t *testing.T) {
	all := embedded(t)
	for _, r := range all {
		ph := placeholders(r)
		ta := newApp(t)
		if code := ta.run(append([]string{"info", strings.ToLower(r.Repo()), "--json"}, cardFlag(all, r)...)...); code != 0 {
			t.Fatalf("%s: exit %d: %s", r.ID(), code, ta.stderr)
		}
		var info infoOut
		if err := json.Unmarshal([]byte(ta.stdout.String()), &info); err != nil {
			t.Fatal(err)
		}
		if info.Recipe.Model.HFRepo != r.Repo() || info.Recipe.Card.ID != r.Card.ID || info.Fits != "unknown" || info.Published != (len(ph) == 0) {
			t.Errorf("%s: info = %+v", r.ID(), info)
		}
		if strings.Join(info.Unpublished, ",") != strings.Join(ph, ",") {
			t.Errorf("%s: unpublished = %v, want %v", r.ID(), info.Unpublished, ph)
		}
		// The cache keeps the recipe's case, whatever case was typed.
		if want := filepath.Join("models--"+strings.ReplaceAll(r.Repo(), "/", "--"), "blobs", r.Model.SHA256); !strings.HasSuffix(info.Download.Blob, want) {
			t.Errorf("%s: blob = %s, want .../%s", r.ID(), info.Download.Blob, want)
		}
		cmd := strings.Join(info.Command, " ")
		for _, want := range []string{"--name " + container.Name(r.Repo()), "--alias " + r.Run.Alias, " " + r.Engine.Image + " "} {
			if !strings.Contains(cmd, want) {
				t.Errorf("%s: command lacks %q: %s", r.ID(), want, cmd)
			}
		}

		ta = newApp(t)
		if code := ta.run(append([]string{"info", r.Repo()}, cardFlag(all, r)...)...); code != 0 {
			t.Fatalf("%s: exit %d: %s", r.ID(), code, ta.stderr)
		}
		out := ta.stdout.String()
		wants := []string{r.Repo() + " on " + r.Card.ID, "unknown (nvidia-smi not found)", r.Engine.Image, r.Model.SHA256}
		if r.Receipt.Summary != "" {
			wants = append(wants, r.Receipt.Summary)
		}
		for _, k := range recipe.MeasurementKeys(r.Receipt.DecodeTPS) {
			wants = append(wants, fmt.Sprintf("%s %.2f tok/s", k, r.Receipt.DecodeTPS[k]))
		}
		wants = append(wants, r.Receipt.Notes...)
		for _, want := range wants {
			if !strings.Contains(out, want) {
				t.Errorf("%s: info lacks %q:\n%s", r.ID(), want, out)
			}
		}
		if strings.Contains(out, "NOT PUBLISHED") != (len(ph) > 0) {
			t.Errorf("%s: NOT PUBLISHED shown wrongly (placeholders %v):\n%s", r.ID(), ph, out)
		}
	}
}

func TestRunDryRunOnEmbeddedRecipes(t *testing.T) {
	all := embedded(t)
	for _, r := range all {
		ph := placeholders(r)
		ta := newApp(t, "podman") // no GPU here
		if code := ta.run(append([]string{"run", r.Repo(), "--dry-run", "--allow-draft"}, cardFlag(all, r)...)...); code != 0 {
			t.Fatalf("%s: exit %d: %s", r.ID(), code, ta.stderr)
		}
		out := flatten(ta.stdout.String())
		f := modelFile(r)
		wants := []string{
			"dry run of 'tensorless run " + r.Repo() + "'",
			"podman pull " + r.Engine.Image,
			"GET " + hfcache.URL("", f),
			filepath.Join("models--"+strings.ReplaceAll(r.Repo(), "/", "--"), "blobs", r.Model.SHA256),
			filepath.Join("snapshots", r.Model.Revision, r.Model.File) + " -> " + hfcache.LinkTarget(f),
			"podman run -d --rm --name " + container.Name(r.Repo()) + " --device nvidia.com/gpu=all",
			"--alias " + r.Run.Alias,
			fmt.Sprintf("-t %d -tb %d", min(6, r.Run.ThreadsMax), min(6, r.Run.ThreadsMax)),
			"podman logs -f " + container.Name(r.Repo()),
		}
		if len(ph) > 0 {
			wants = append(wants, "NOT PUBLISHED: the recipe's "+joinAnd(ph))
		}
		for _, want := range wants {
			if !strings.Contains(out, want) {
				t.Errorf("%s: plan lacks %q:\n%s", r.ID(), want, out)
			}
		}
		if len(ph) == 0 && strings.Contains(out, "NOT PUBLISHED") {
			t.Errorf("%s: a published recipe is shown as unpublished", r.ID())
		}
		if strings.Contains(out, "UNTUNED") {
			t.Errorf("%s: a plain run is not untuned", r.ID())
		}
		for _, c := range ta.fake.Calls() {
			if !strings.HasPrefix(c, "nvidia-smi") {
				t.Errorf("%s: a dry run ran %q", r.ID(), c)
			}
		}

		// Without --allow-draft, a draft is refused even for a dry run.
		ta = newApp(t, "podman")
		code := ta.run(append([]string{"run", r.Repo(), "--dry-run"}, cardFlag(all, r)...)...)
		if r.Draft() && (code != 1 || !strings.Contains(ta.stderr.String(), "is a draft")) {
			t.Errorf("%s: exit %d: %s", r.ID(), code, ta.stderr)
		}
		if !r.Draft() && code != 0 {
			t.Errorf("%s: a released recipe was refused: %s", r.ID(), ta.stderr)
		}
	}
}

// The exact wording, on fixtures that keep their placeholders.
func TestPlaceholderFixtures(t *testing.T) {
	ta := newApp(t, "podman")
	if code := ta.run("run", "org/placeholder-gguf", "--recipes", "testdata/recipes", "--dry-run", "--allow-draft"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	for _, want := range []string{
		"NOT PUBLISHED: the recipe's revision, image digest and size_bytes are placeholders. A real pull, run or serve refuses it until they are filled in.",
		"download the model (size unknown: size_bytes is 0)",
		"/Org/Placeholder-GGUF/resolve/0000000000000000000000000000000000000000/m.gguf",
	} {
		if !strings.Contains(ta.stdout.String(), want) {
			t.Errorf("plan lacks %q:\n%s", want, ta.stdout)
		}
	}

	ta = newApp(t, "podman")
	if code := ta.run("pull", "partial-gguf", "--recipes", "testdata/recipes", "--dry-run", "--allow-draft"); code == 0 {
		t.Errorf("Partial-GGUF has no short name; %q must not resolve", "partial-gguf")
	}
	if code := ta.run("pull", "org/partial-gguf", "--recipes", "testdata/recipes", "--dry-run", "--allow-draft"); code != 0 ||
		!strings.Contains(ta.stdout.String(), "NOT PUBLISHED: the recipe's revision is a placeholder. A real pull") {
		t.Errorf("exit %d: %s%s", code, ta.stdout, ta.stderr)
	}

	tests := []struct{ model, want string }{
		{"placeholder", "the recipe for Org/Placeholder-GGUF on p100 is not published yet: its revision, image digest and size_bytes are placeholders. It cannot be pulled or run until they are filled in, even with --allow-draft"},
		{"ORG/PARTIAL-GGUF", "the recipe for Org/Partial-GGUF on p100 is not published yet: its revision is a placeholder."},
	}
	for _, tc := range tests {
		for _, cmd := range []string{"pull", "run", "serve"} {
			for _, found := range [][]string{{"podman"}, {"podman", "nvidia-smi"}} {
				ta := newApp(t, found...)
				ta.host.gpus = p100GPU
				if code := ta.run(cmd, tc.model, "--recipes", "testdata/recipes", "--allow-draft"); code != 1 || !strings.Contains(ta.stderr.String(), tc.want) {
					t.Errorf("%s %s (%v): exit %d: %s", cmd, tc.model, found, code, ta.stderr)
				}
				if c := ta.host.calls(ta.fake, "podman"); len(c) != 0 {
					t.Errorf("an unpublished recipe ran %v", c)
				}
			}
		}
	}
}

func TestEmbeddedRecipesResolve(t *testing.T) {
	all := embedded(t)
	for _, r := range all {
		ph := placeholders(r)
		for _, arg := range names(r) {
			for _, gpuHost := range []bool{false, true} {
				found := []string{"podman"}
				if gpuHost {
					found = append(found, "nvidia-smi")
				}
				ta := newApp(t, found...)
				ta.host.gpus = gpuLines([]*recipe.Recipe{r})
				if code := ta.run(append([]string{"info", arg, "--json"}, cardFlag(all, r)...)...); code != 0 {
					t.Errorf("info %s: exit %d: %s", arg, code, ta.stderr)
					continue
				}
				var info infoOut
				if err := json.Unmarshal([]byte(ta.stdout.String()), &info); err != nil || info.Recipe.Model.HFRepo != r.Repo() {
					t.Errorf("info %s resolved to %s (%v)", arg, info.Recipe.Model.HFRepo, err)
				}

				if r.Draft() {
					ta = newApp(t, found...)
					ta.host.gpus = gpuLines([]*recipe.Recipe{r})
					want := "the recipe for " + r.Repo() + " on " + r.Card.ID + " is a draft"
					if code := ta.run(append([]string{"run", arg}, cardFlag(all, r)...)...); code != 1 || !strings.Contains(ta.stderr.String(), want) {
						t.Errorf("run %s (gpu %v): exit %d: %s", arg, gpuHost, code, ta.stderr)
					}
				}
				if len(ph) == 0 && gpuHost {
					// Published: --allow-draft (if a draft) is all it needs. It
					// gets past the gate and stops at the missing image.
					ta = newApp(t, found...)
					ta.host.gpus = gpuLines([]*recipe.Recipe{r})
					if code := ta.run(append([]string{"serve", arg, "--allow-draft", "--detach"}, cardFlag(all, r)...)...); code != 1 ||
						!strings.Contains(ta.stderr.String(), "the engine image is not pulled. Run: tensorless pull "+r.Repo()) {
						t.Errorf("serve %s --allow-draft: exit %d: %s", arg, code, ta.stderr)
					}
				}
				if len(ph) > 0 {
					for _, cmd := range []string{"run", "serve", "pull"} {
						ta = newApp(t, found...)
						ta.host.gpus = gpuLines([]*recipe.Recipe{r})
						want := "the recipe for " + r.Repo() + " on " + r.Card.ID + " is not published yet: its " + joinAnd(ph)
						if code := ta.run(append([]string{cmd, arg, "--allow-draft"}, cardFlag(all, r)...)...); code != 1 || !strings.Contains(ta.stderr.String(), want) {
							t.Errorf("%s %s --allow-draft (gpu %v): exit %d: %s", cmd, arg, gpuHost, code, ta.stderr)
						}
						if c := ta.host.calls(ta.fake, "podman"); len(c) != 0 {
							t.Errorf("an unpublished recipe ran %v", c)
						}
					}
				}
			}
		}
	}
}

func TestModelsListsEveryRecipe(t *testing.T) {
	all := embedded(t)
	ta := newApp(t, "nvidia-smi")
	ta.host.gpus = gpuLines(all)
	if code := ta.run("models", "--json"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	var rows []modelRow
	if err := json.Unmarshal([]byte(ta.stdout.String()), &rows); err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(all) {
		t.Fatalf("%d rows, want %d", len(rows), len(all))
	}
	for i, r := range all {
		want := modelRow{Model: r.Repo(), Name: r.Model.Name, Card: r.Card.ID, Title: r.Model.Title, Status: r.Status,
			Published: len(placeholders(r)) == 0, Release: r.Engine.Release, Arch: r.Engine.Arch, Fits: "yes", Cached: "no"}
		if rows[i] != want {
			t.Errorf("row %d = %+v, want %+v", i, rows[i], want)
		}
	}

	ta = newApp(t)
	if code := ta.run("models"); code != 0 {
		t.Fatalf("exit %d", code)
	}
	out := ta.stdout.String()
	for _, r := range all {
		if !strings.Contains(out, r.Repo()) {
			t.Errorf("models lacks %s:\n%s", r.Repo(), out)
		}
	}
	if !strings.Contains(out, "unknown") || !strings.Contains(ta.stderr.String(), "nvidia-smi not found") {
		t.Errorf("models without a GPU:\n%s%s", out, ta.stderr)
	}
}

func TestUnknownModelListsRepoIDs(t *testing.T) {
	ta := newApp(t)
	if code := ta.run("info", "nobody/nothing"); code != 1 {
		t.Fatalf("exit %d", code)
	}
	e := ta.stderr.String()
	if !strings.Contains(e, "Known models: ") {
		t.Errorf("stderr = %s", e)
	}
	for _, r := range embedded(t) {
		if !strings.Contains(e, r.Repo()) {
			t.Errorf("stderr lacks %s: %s", r.Repo(), e)
		}
	}
}

func TestUsageErrors(t *testing.T) {
	for _, args := range [][]string{
		{"run"},
		{"run", "a", "b"},
		{"run", "org/repo", "--", "-c", "4096"},
		{"run", "org/repo", "--", "--port=9000"},
		{"pull", "org/repo", "--", "--temp", "1"},
		{"run", "org/repo@main"},
		{"run", "org/repo", "--gpu", "x"},
		{"frobnicate"},
	} {
		ta := newApp(t)
		if code := ta.run(args...); code != 2 {
			t.Errorf("%v: exit %d, want 2: %s", args, code, ta.stderr)
		}
	}
	ta := newApp(t)
	if code := ta.run("version"); code != 0 || !strings.HasPrefix(ta.stdout.String(), "tensorless v9.9.9 (commit abc1234)") {
		t.Errorf("version: exit %d, %q", code, ta.stdout)
	}
}

func TestGPUSelection(t *testing.T) {
	data := []byte("model weights")
	mixed := "0, NVIDIA GeForce RTX 3090, 8.6, 24576, 580.95.05, GPU-a\n1, Tesla P100-PCIE-16GB, 6.0, 16280, 580.95.05, GPU-b\n"

	// Only the P100 matches: the container gets just that GPU.
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = mixed
	dir := ta.published(t, data)
	if code := ta.run("serve", "org/model-gguf", "--recipes", dir, "--dry-run"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	if out := ta.stdout.String(); !strings.Contains(out, "--device nvidia.com/gpu=1") || !strings.Contains(out, "GPU 1 (Tesla P100-PCIE-16GB) matches card p100") {
		t.Errorf("plan:\n%s", out)
	}

	// --gpu 0 picks the 3090, which fits no recipe.
	ta = newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = mixed
	if code := ta.run("serve", "m1", "--recipes", dir, "--gpu", "0", "--dry-run"); code != 1 || !strings.Contains(ta.stderr.String(), "fits GPU 0") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}

	// A 12 GB P100 does not fit.
	ta = newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = "0, Tesla P100-PCIE-12GB, 6.0, 12288, 580.95.05, GPU-c\n"
	if code := ta.run("pull", "m1", "--recipes", dir); code != 1 || !strings.Contains(ta.stderr.String(), "at least 16000 MiB") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}

	// An old driver is refused.
	ta = newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = "0, Tesla P100-PCIE-16GB, 6.0, 16384, 470.256.02, GPU-d\n"
	if code := ta.run("pull", "m1", "--recipes", dir); code != 1 || !strings.Contains(ta.stderr.String(), "470.256.02 is older than 525.60.13") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}

	// Two P100s: every GPU matches, so the container gets all of them; docker.
	ta = newApp(t, "nvidia-smi", "docker")
	ta.host.gpus = p100GPU + "1, Tesla P100-PCIE-16GB, 6.0, 16384, 580.95.05, GPU-e\n"
	if code := ta.run("serve", "m1", "--recipes", dir, "--dry-run"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	if out := ta.stdout.String(); !strings.Contains(out, "docker run -d --rm --name tensorless-org-model-gguf --gpus all -p") {
		t.Errorf("plan:\n%s", out)
	}
}

func TestRevisionOverride(t *testing.T) {
	data := []byte("model weights")
	rev := strings.Repeat("b", 40)
	ta := newApp(t, "podman")
	dir := ta.published(t, data)
	if code := ta.run("run", "Org/Model-GGUF@"+rev, "--recipes", dir, "--card", "p100", "--dry-run"); code != 1 ||
		!strings.Contains(ta.stderr.String(), "is not the one the recipe for Org/Model-GGUF on p100 pins") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}
	ta = newApp(t, "podman")
	if code := ta.run("run", "org/model-gguf@"+rev, "--recipes", dir, "--card", "p100", "--dry-run", "--allow-draft"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	out := ta.stdout.String()
	for _, want := range []string{"UNTUNED: revision " + rev, "/Org/Model-GGUF/resolve/" + rev + "/m.gguf", "snapshots/" + rev} {
		if !strings.Contains(out, want) {
			t.Errorf("plan lacks %q:\n%s", want, out)
		}
	}
	// The same revision as the recipe's is not an override.
	ta = newApp(t, "podman")
	if code := ta.run("run", "m1@0123456789abcdef0123456789abcdef01234567", "--recipes", dir, "--card", "p100", "--dry-run"); code != 0 || strings.Contains(ta.stdout.String(), "UNTUNED") {
		t.Errorf("exit %d: %s%s", code, ta.stdout, ta.stderr)
	}
}

func healthServer(t *testing.T, status int) (string, int) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(status) }))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	return u.Hostname(), port
}

func TestServeDetach(t *testing.T) {
	data := []byte("model weights")
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	ta.host.imagePresent = true
	dir := ta.published(t, data)
	blob := ta.cache(t, data)
	h, port := healthServer(t, http.StatusOK)

	code := ta.run("serve", "org/model-gguf", "--recipes", dir, "--detach", "--host", h, "--port", strconv.Itoa(port), "--context", "2048", "--", "--temp", "0.5")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	runs := ta.host.calls(ta.fake, "podman run")
	if len(runs) != 1 {
		t.Fatalf("podman run calls: %v", ta.fake.Calls())
	}
	want := "podman run -d --rm --name tensorless-org-model-gguf --device nvidia.com/gpu=all --security-opt label=disable " +
		"-p 127.0.0.1:" + strconv.Itoa(port) + ":8080 -v " + blob + ":/models/m.gguf:ro -e OMP_NUM_THREADS=6 -e GGML_X=1 " +
		"ghcr.io/x/engine@sha256:" + strings.Repeat("a", 64) +
		" -m /models/m.gguf --host 0.0.0.0 --port 8080 -c 2048 --alias Org/Model-GGUF -t 6 -tb 6 -ngl 99 --temp 0.5"
	if runs[0] != want {
		t.Errorf("run =\n%s\nwant\n%s", runs[0], want)
	}
	if e := ta.stderr.String(); strings.Count(e, "UNTUNED") != 1 || !strings.Contains(e, "--context 2048 instead of 4096") {
		t.Errorf("stderr = %s", e)
	}
	if out := ta.stdout.String(); !strings.Contains(out, "Org/Model-GGUF is ready at http://127.0.0.1:") || !strings.Contains(out, "tensorless stop Org/Model-GGUF") {
		t.Errorf("stdout = %s", out)
	}

	// Serving again while it runs is refused.
	ta.stderr.b.Reset()
	if code := ta.run("serve", "m1", "--recipes", dir, "--detach", "--port", strconv.Itoa(port)); code != 1 ||
		!strings.Contains(ta.stderr.String(), "already running") || !strings.Contains(ta.stderr.String(), "tensorless stop Org/Model-GGUF") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}

	// stop finds it.
	ta.stdout.b.Reset()
	if code := ta.run("stop", "ORG/MODEL-GGUF", "--recipes", dir); code != 0 || !strings.Contains(ta.stdout.String(), "stopped tensorless-org-model-gguf") {
		t.Errorf("stop: exit %d: %s %s", code, ta.stdout, ta.stderr)
	}
}

func TestServeNeedsPull(t *testing.T) {
	data := []byte("model weights")
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	dir := ta.published(t, data)
	if code := ta.run("serve", "m1", "--recipes", dir, "--detach"); code != 1 || !strings.Contains(ta.stderr.String(), "Run: tensorless pull Org/Model-GGUF") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}
	ta.host.imagePresent = true
	ta.stderr.b.Reset()
	if code := ta.run("serve", "m1", "--recipes", dir, "--detach"); code != 1 || !strings.Contains(ta.stderr.String(), "the model is not downloaded") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}
}

func TestServeContainerExits(t *testing.T) {
	data := []byte("model weights")
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	ta.host.imagePresent = true
	ta.host.exitAtStart = true
	ta.host.logs = "loading model\nCUDA error: no kernel image is available for execution on the device\n"
	dir := ta.published(t, data)
	ta.cache(t, data)
	h, port := healthServer(t, http.StatusServiceUnavailable)
	code := ta.run("serve", "m1", "--recipes", dir, "--detach", "--host", h, "--port", strconv.Itoa(port))
	if code != 1 {
		t.Fatalf("exit %d", code)
	}
	e := ta.stderr.String()
	if !strings.Contains(e, "last 2 log lines") || !strings.Contains(e, "no kernel image") || !strings.Contains(e, "exited before it was ready") {
		t.Errorf("stderr = %s", e)
	}
}

func TestServeForegroundStopsOnInterrupt(t *testing.T) {
	data := []byte("model weights")
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	ta.host.imagePresent = true
	ta.host.logs = "server listening\n"
	dir := ta.published(t, data)
	ta.cache(t, data)
	h, port := healthServer(t, http.StatusOK)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan int)
	go func() {
		done <- ta.Run(ctx, []string{"serve", "m1", "--recipes", dir, "--host", h, "--port", strconv.Itoa(port)})
	}()
	deadline := time.Now().Add(5 * time.Second)
	for !strings.Contains(ta.stdout.String(), "is ready") {
		if time.Now().After(deadline) {
			t.Fatalf("never ready: %s %s", ta.stdout, ta.stderr)
		}
		time.Sleep(5 * time.Millisecond)
	}
	cancel() // what Ctrl-C does
	select {
	case code := <-done:
		if code != 0 {
			t.Errorf("exit %d: %s", code, ta.stderr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("serve did not return")
	}
	if stops := ta.host.calls(ta.fake, "podman stop tensorless-org-model-gguf"); len(stops) != 1 {
		t.Errorf("calls: %v", ta.fake.Calls())
	}
	if !strings.Contains(ta.stderr.String(), "server listening") || !strings.Contains(ta.stdout.String(), "Ctrl-C stops the server") {
		t.Errorf("the logs were not followed: %s", ta.stderr)
	}
}

func TestPullDownloadsFromHFEndpoint(t *testing.T) {
	data := []byte(strings.Repeat("weights ", 1000))
	var auth string
	hf := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/Org/Model-GGUF/resolve/0123456789abcdef0123456789abcdef01234567/m.gguf" {
			http.NotFound(w, r)
			return
		}
		auth = r.Header.Get("Authorization")
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(data))
	}))
	defer hf.Close()

	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	ta.env["HF_ENDPOINT"] = hf.URL
	ta.env["HF_TOKEN"] = "hf_secret_token"
	ta.HTTP = hf.Client()
	dir := ta.published(t, data)
	if code := ta.run("pull", "m1", "--recipes", dir); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	if len(ta.host.calls(ta.fake, "podman pull ghcr.io/x/engine@sha256:")) != 1 {
		t.Errorf("calls: %v", ta.fake.Calls())
	}
	if auth != "Bearer hf_secret_token" {
		t.Errorf("Authorization = %q", auth)
	}
	if strings.Contains(ta.stdout.String()+ta.stderr.String(), "hf_secret_token") {
		t.Errorf("the token was printed")
	}
	link := filepath.Join(ta.env["HF_HUB_CACHE"], "models--Org--Model-GGUF", "snapshots", "0123456789abcdef0123456789abcdef01234567", "m.gguf")
	if got, err := os.ReadFile(link); err != nil || !bytes.Equal(got, data) {
		t.Errorf("snapshot: %v", err)
	}

	// The second pull finds both in place.
	ta.stderr.b.Reset()
	if code := ta.run("pull", "m1", "--recipes", dir); code != 0 || !strings.Contains(ta.stderr.String(), "already pulled") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}
	if len(ta.host.calls(ta.fake, "podman pull")) != 1 {
		t.Errorf("the image was pulled twice")
	}

	// The dry run shows the token is used without showing it.
	ta.stdout.b.Reset()
	os.RemoveAll(ta.env["HF_HUB_CACHE"])
	if code := ta.run("pull", "m1", "--recipes", dir, "--dry-run"); code != 0 ||
		!strings.Contains(ta.stdout.String(), "Authorization: Bearer (HF_TOKEN, not shown)") || strings.Contains(ta.stdout.String(), "hf_secret") {
		t.Errorf("exit %d: %s", code, ta.stdout)
	}
}

func TestDoctor(t *testing.T) {
	all := embedded(t)
	ta := newApp(t, "nvidia-smi", "podman", "nvidia-ctk")
	ta.host.gpus = gpuLines(all)
	ta.files["/etc/cdi/nvidia.yaml"] = true
	if code := ta.run("doctor"); code != 0 {
		t.Errorf("exit %d:\n%s", code, ta.stdout)
	}
	out := ta.stdout.String()
	wants := []string{"PASS  GPU 0: Fake ", "PASS  driver 999.0 is at least", "PASS  NVIDIA CDI spec for podman: /etc/cdi/nvidia.yaml", "all checks passed"}
	for _, r := range all {
		wants = append(wants, describe(r))
	}
	for _, want := range wants {
		if !strings.Contains(out, want) {
			t.Errorf("doctor lacks %q:\n%s", want, out)
		}
	}

	// An old driver, no CDI spec and a full disk, on a fixture recipe.
	ta = newApp(t, "nvidia-smi", "podman", "nvidia-ctk")
	dir := ta.published(t, []byte("model weights"))
	ta.host.gpus = "0, Tesla P100-PCIE-16GB, 6.0, 16384, 470.256.02, GPU-x\n"
	ta.FreeSpace = func(string) (uint64, error) { return 1, nil }
	if code := ta.run("doctor", "--recipes", dir); code != 1 {
		t.Errorf("exit %d", code)
	}
	out = ta.stdout.String()
	for _, want := range []string{
		"PASS  GPU 0: Tesla P100-PCIE-16GB, compute 6.0, 16384 MiB, driver 470.256.02; fits Org/Model-GGUF on p100",
		"FAIL  driver 470.256.02 is older than 525.60.13 (needed by Org/Model-GGUF on p100)",
		"FAIL  no NVIDIA CDI spec for podman",
		"fix: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml",
		"FAIL  model cache", "3 checks failed",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("doctor lacks %q:\n%s", want, out)
		}
	}

	// A GPU that fits no recipe.
	ta = newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = "0, Tesla P100-PCIE-12GB, 6.0, 12288, 580.95.05, GPU-y\n"
	ta.files["/etc/cdi/nvidia.yaml"] = true
	if code := ta.run("doctor", "--recipes", dir); code != 1 || !strings.Contains(ta.stdout.String(), "FAIL  no GPU here fits a recipe") {
		t.Errorf("exit %d:\n%s", code, ta.stdout)
	}

	ta = newApp(t, "nvidia-smi", "docker")
	ta.host.gpus = gpuLines(all)
	ta.host.dockerInfo = `{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}`
	if code := ta.run("doctor"); code != 0 || !strings.Contains(ta.stdout.String(), "docker has the nvidia runtime") {
		t.Errorf("exit %d:\n%s", code, ta.stdout)
	}
	ta = newApp(t)
	if code := ta.run("doctor"); code != 1 || !strings.Contains(ta.stdout.String(), "FAIL  nvidia-smi not found") || !strings.Contains(ta.stdout.String(), "neither podman nor docker") {
		t.Errorf("exit %d:\n%s", code, ta.stdout)
	}
}

func TestTail(t *testing.T) {
	tl := newTail(3)
	io.WriteString(tl, "a\nb\nc")
	io.WriteString(tl, "d\ne\n")
	io.WriteString(tl, "f")
	if got := strings.Join(tl.Lines(), ","); got != "cd,e,f" {
		t.Errorf("Lines = %q", got)
	}
}

func TestShellWrap(t *testing.T) {
	got := shellWrap([]string{"podman", "run", "-e", "A=1", "--x", "it's", "--flag", "img"}, 20)
	want := "podman run -e A=1 \\\n    --x 'it'\\''s' \\\n    --flag img"
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}
