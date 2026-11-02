package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/iammrduncan/tensorless/internal/container"
	"github.com/iammrduncan/tensorless/internal/gpu"
	"github.com/iammrduncan/tensorless/internal/hfcache"
	"github.com/iammrduncan/tensorless/internal/recipe"
)

func (a *App) loadRecipes(o *options) (*recipe.Set, error) {
	if o.recipes != "" {
		fi, err := os.Stat(o.recipes)
		if err != nil || !fi.IsDir() {
			return nil, fmt.Errorf("--recipes %s is not a directory", o.recipes)
		}
		return recipe.Load(os.DirFS(o.recipes), o.recipes)
	}
	return recipe.Load(a.Recipes, "recipes")
}

// target is the recipe a command uses, and the GPU it was matched to.
type target struct {
	r      *recipe.Recipe
	gpu    *gpu.GPU // nil when detection was skipped or failed
	device int      // the container's GPU: container.AllGPUs or an index
	how    string   // how the recipe was chosen
	pinned string   // the recipe's own revision, when MODEL@REVISION overrides it
}

// withRevision applies a MODEL@REVISION that differs from the recipe's to a
// copy of the recipe, and remembers the pinned one.
func withRevision(t *target, o *options) *target {
	if o.revision == "" || o.revision == t.r.Model.Revision {
		return t
	}
	r := *t.r
	r.Model.Revision = o.revision
	t.pinned = t.r.Model.Revision
	t.r = &r
	return t
}

// modelArg is the MODEL argument to suggest in messages: the repo id, with
// the revision when one was given.
func modelArg(t *target, o *options) string {
	if o.revision != "" {
		return t.r.Repo() + "@" + o.revision
	}
	return t.r.Repo()
}

func describe(r *recipe.Recipe) string { return r.Repo() + " on " + r.Card.ID }

func detectReason(err error) string {
	if errors.Is(err, gpu.ErrNoNvidiaSMI) {
		return "nvidia-smi not found"
	}
	return err.Error()
}

func cardIDs(rs []*recipe.Recipe) string {
	ids := make([]string, len(rs))
	for i, r := range rs {
		ids[i] = r.Card.ID
	}
	return strings.Join(ids, "|")
}

func requirement(r *recipe.Recipe) string {
	return fmt.Sprintf("%s (compute %s, at least %d MiB)", r.Card.ID, r.Card.ComputeCapability, r.Card.MinMemoryMiB)
}

// pick returns the first GPU among gpus (only index want, if want >= 0) that
// matches one of rs, and that recipe.
func pick(gpus []gpu.GPU, rs []*recipe.Recipe, want int) (*gpu.GPU, *recipe.Recipe) {
	for i := range gpus {
		g := &gpus[i]
		if want >= 0 && g.Index != want {
			continue
		}
		for _, r := range rs {
			if g.Matches(r.Card.ComputeCapability, r.Card.MinMemoryMiB) {
				return g, r
			}
		}
	}
	return nil, nil
}

// device is the container's GPU: the one asked for, else every GPU when they
// all match the recipe, else the matched one, so that llama-server cannot
// land on a card the image was not built for.
func device(gpus []gpu.GPU, r *recipe.Recipe, g *gpu.GPU, want int) int {
	if want >= 0 {
		return want
	}
	if g == nil {
		return container.AllGPUs
	}
	for _, o := range gpus {
		if !o.Matches(r.Card.ComputeCapability, r.Card.MinMemoryMiB) {
			return g.Index
		}
	}
	return container.AllGPUs
}

// fits says whether r can run on this host: yes, no, or unknown without a GPU.
func fits(gpus []gpu.GPU, detectErr error, r *recipe.Recipe) string {
	if detectErr != nil {
		return "unknown"
	}
	g, _ := pick(gpus, []*recipe.Recipe{r}, -1)
	if g == nil {
		return "no"
	}
	if ok, err := gpu.DriverAtLeast(g.Driver, r.Card.MinDriver); err != nil || !ok {
		return "no"
	}
	return "yes"
}

// resolve picks the model's recipe: the --card one, or the one that matches
// the detected GPU. With lenient (dry runs and info), a host without a GPU
// falls back to the model's only recipe. A MODEL@REVISION is applied.
func (a *App) resolve(ctx context.Context, set *recipe.Set, o *options, lenient bool) (*target, error) {
	t, err := a.pickRecipe(ctx, set, o, lenient)
	if err != nil {
		return nil, err
	}
	return withRevision(t, o), nil
}

func (a *App) pickRecipe(ctx context.Context, set *recipe.Set, o *options, lenient bool) (*target, error) {
	rs, err := set.ForModel(o.model)
	if err != nil {
		return nil, err
	}
	if o.card != "" {
		r, err := set.Lookup(o.model, o.card)
		if err != nil {
			return nil, err
		}
		dev := container.AllGPUs
		if o.gpu >= 0 {
			dev = o.gpu
		}
		return &target{r: r, device: dev, how: "chosen with --card; GPU detection skipped"}, nil
	}

	gpus, err := gpu.Detect(ctx, a.Runner)
	if err != nil {
		// A refusal that holds for every recipe of the model matters more
		// than the missing GPU.
		if !lenient {
			var gateErr error
			for _, r := range rs {
				if gateErr = gate(withRevision(&target{r: r}, o), o); gateErr == nil {
					break
				}
			}
			if gateErr != nil {
				return nil, gateErr
			}
		}
		if lenient && len(rs) == 1 {
			dev := container.AllGPUs
			if o.gpu >= 0 {
				dev = o.gpu
			}
			return &target{r: rs[0], device: dev,
				how: fmt.Sprintf("no GPU detected (%s); showing %s, the only recipe for %s", detectReason(err), rs[0].Card.ID, rs[0].Repo())}, nil
		}
		return nil, fmt.Errorf("cannot detect the GPU: %s. Run 'tensorless doctor', or pass --card %s to choose the recipe without detection",
			detectReason(err), cardIDs(rs))
	}
	if o.gpu >= 0 {
		found := false
		for _, g := range gpus {
			found = found || g.Index == o.gpu
		}
		if !found {
			return nil, fmt.Errorf("there is no GPU %d; nvidia-smi lists:\n%s", o.gpu, listGPUs(gpus))
		}
	}
	g, r := pick(gpus, rs, o.gpu)
	if g == nil {
		var need []string
		for _, r := range rs {
			need = append(need, requirement(r))
		}
		which := "this host's GPUs"
		if o.gpu >= 0 {
			which = fmt.Sprintf("GPU %d", o.gpu)
		}
		return nil, fmt.Errorf("no recipe for %s fits %s:\n%s\n%s needs: %s. See: tensorless models",
			rs[0].Repo(), which, listGPUs(gpus), rs[0].Repo(), strings.Join(need, "; "))
	}
	ok, err := gpu.DriverAtLeast(g.Driver, r.Card.MinDriver)
	if err != nil {
		return nil, fmt.Errorf("cannot read the driver version %q of GPU %d: %v", g.Driver, g.Index, err)
	}
	if !ok {
		return nil, fmt.Errorf("the NVIDIA driver %s is older than %s, the minimum for %s. Install driver %s or newer, then run: tensorless doctor",
			g.Driver, r.Card.MinDriver, describe(r), r.Card.MinDriver)
	}
	return &target{r: r, gpu: g, device: device(gpus, r, g, o.gpu),
		how: fmt.Sprintf("GPU %d (%s) matches card %s", g.Index, g.Name, r.Card.ID)}, nil
}

func listGPUs(gpus []gpu.GPU) string {
	lines := make([]string, len(gpus))
	for i, g := range gpus {
		lines[i] = "  " + g.String()
	}
	return strings.Join(lines, "\n")
}

// gate refuses drafts and other revisions without --allow-draft, and recipes
// with placeholders unless this is a dry run.
func gate(t *target, o *options) error {
	r := t.r
	if t.pinned != "" && !o.allowDraft {
		return fmt.Errorf("revision %s is not the one the recipe for %s pins (%s). Pass --allow-draft to use it anyway; the run is then untuned",
			r.Model.Revision, describe(r), t.pinned)
	}
	if r.Draft() && !o.allowDraft {
		return fmt.Errorf("the recipe for %s is a draft: it has not passed the release gate. Pass --allow-draft to use it anyway; 'tensorless info %s' shows it",
			describe(r), r.Repo())
	}
	if u := r.Unpublished(); len(u) > 0 && !o.dryRun {
		return fmt.Errorf("the recipe for %s is not published yet: its %s %s. It cannot be pulled or run until they are filled in, even with --allow-draft. 'tensorless info %s' shows it, and --dry-run shows the plan",
			describe(r), joinAnd(u), plural(len(u), "is a placeholder", "are placeholders"), r.Repo())
	}
	return nil
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

// prepare loads the recipes, resolves the model's recipe and applies the gate.
func (a *App) prepare(ctx context.Context, o *options) (*target, error) {
	set, err := a.loadRecipes(o)
	if err != nil {
		return nil, err
	}
	t, err := a.resolve(ctx, set, o, o.dryRun)
	if err != nil {
		return nil, err
	}
	if err := gate(t, o); err != nil {
		return nil, err
	}
	if t.r.Draft() && !o.dryRun {
		fmt.Fprintf(a.Stderr, "note: using the draft recipe for %s (--allow-draft)\n", describe(t.r))
	}
	return t, nil
}

// runtimeFor selects the container runtime. When none is installed, lenient
// callers get a name to show and a note instead of an error.
func (a *App) runtimeFor(o *options, lenient bool) (string, string, error) {
	rt, err := container.Select(a.Runner, o.runtime)
	if err == nil {
		return rt, "", nil
	}
	if !lenient {
		return "", "", err
	}
	rt = o.runtime
	if rt != "podman" && rt != "docker" {
		rt = "podman"
	}
	return rt, fmt.Sprintf("%s is not installed here; the commands show %s", rt, rt), nil
}

func (a *App) cacheDir(o *options) (string, error) {
	dir := o.cache
	if dir == "" {
		home := a.home()
		if home == "" && a.getenv("HF_HUB_CACHE") == "" && a.getenv("HF_HOME") == "" {
			return "", errors.New("cannot find your home directory for the model cache; pass --cache DIR or set HF_HUB_CACHE")
		}
		dir = hfcache.DefaultCacheDir(a.getenv, home)
	}
	return filepath.Abs(dir) // the container mount needs an absolute path
}

func (a *App) stateDir() string {
	home := a.home()
	if home == "" && a.getenv("XDG_STATE_HOME") == "" {
		return ""
	}
	return hfcache.DefaultStateDir(a.getenv, home)
}

func (a *App) downloader() *hfcache.Downloader {
	return &hfcache.Downloader{
		Client:    a.HTTP,
		Endpoint:  a.getenv("HF_ENDPOINT"),
		Token:     a.getenv("HF_TOKEN"),
		UserAgent: "tensorless/" + a.Version,
		StateDir:  a.stateDir(),
		Log:       a.Stderr,
		TTY:       a.StderrTTY,
		FreeSpace: a.FreeSpace,
	}
}

func modelFile(r *recipe.Recipe) hfcache.File {
	return hfcache.File{
		Repo:     r.Model.HFRepo,
		Revision: r.Model.Revision,
		Path:     r.Model.File,
		SHA256:   r.Model.SHA256,
		Size:     r.Model.SizeBytes,
	}
}

func (a *App) threads(r *recipe.Recipe) int {
	n := a.NumCPU()
	if n < 1 {
		n = 1
	}
	return min(n, r.Run.ThreadsMax)
}

// spec is the container to start for t.
func (a *App) spec(t *target, o *options, rt, blob string) (container.Spec, error) {
	if strings.Contains(blob, ":") {
		return container.Spec{}, fmt.Errorf("the model path %s contains ':', which the container mount cannot express; pass --cache DIR with a path without ':'", blob)
	}
	ctxLen := t.r.Run.Context
	if o.context > 0 {
		ctxLen = o.context
	}
	return container.Spec{
		Runtime:   rt,
		Name:      container.Name(t.r.Repo()),
		Image:     t.r.Engine.Image,
		Host:      o.host,
		Port:      o.port,
		ModelPath: blob,
		ModelFile: filepath.Base(filepath.FromSlash(t.r.Model.File)),
		Threads:   a.threads(t.r),
		Env:       t.r.Run.Env,
		Context:   ctxLen,
		Alias:     t.r.Run.Alias,
		Args:      t.r.Run.Args,
		Extra:     o.extra,
		GPU:       t.device,
	}, nil
}

// untuned describes how the command line changes the measured configuration,
// or returns "".
func untuned(t *target, o *options) string {
	r := t.r
	var why []string
	if t.pinned != "" {
		why = append(why, fmt.Sprintf("revision %s instead of the recipe's %s", r.Model.Revision, t.pinned))
	}
	if o.context > 0 && o.context != r.Run.Context {
		why = append(why, fmt.Sprintf("--context %d instead of %d", o.context, r.Run.Context))
	}
	if len(o.extra) > 0 {
		why = append(why, "extra arguments "+shellJoin(o.extra))
	}
	return strings.Join(why, " and ")
}

func deviceName(d int) string {
	if d == container.AllGPUs {
		return "all"
	}
	return strconv.Itoa(d)
}
