package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"strings"
	"text/tabwriter"

	"github.com/iammrduncan/tensorless/internal/container"
	"github.com/iammrduncan/tensorless/internal/gpu"
	"github.com/iammrduncan/tensorless/internal/hfcache"
	"github.com/iammrduncan/tensorless/internal/recipe"
)

func writeJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func statusLabel(r *recipe.Recipe) string {
	if !r.Published() {
		return r.Status + ", unpublished"
	}
	return r.Status
}

type modelRow struct {
	Model     string `json:"model"` // the repo id
	Name      string `json:"name,omitempty"`
	Card      string `json:"card"`
	Title     string `json:"title"`
	Status    string `json:"status"`
	Published bool   `json:"published"`
	Release   string `json:"release"`
	Arch      string `json:"arch"`
	Fits      string `json:"fits"`   // yes, no or unknown
	Cached    string `json:"cached"` // yes, partial or no
}

func (a *App) cmdModels(ctx context.Context, o *options) error {
	set, err := a.loadRecipes(o)
	if err != nil {
		return err
	}
	gpus, derr := gpu.Detect(ctx, a.Runner)
	cache, cerr := a.cacheDir(o)
	rows := []modelRow{}
	for _, r := range set.Recipes {
		cached := "unknown"
		if cerr == nil {
			cached = hfcache.Status(cache, modelFile(r))
		}
		rows = append(rows, modelRow{
			Model: r.Repo(), Name: r.Model.Name, Card: r.Card.ID, Title: r.Model.Title, Status: r.Status, Published: r.Published(),
			Release: r.Engine.Release, Arch: r.Engine.Arch, Fits: fits(gpus, derr, r), Cached: cached,
		})
	}
	if o.json {
		return writeJSON(a.Stdout, rows)
	}
	tw := tabwriter.NewWriter(a.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tCARD\tSTATUS\tENGINE\tFITS\tCACHED")
	for i, r := range set.Recipes {
		row := rows[i]
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n", row.Model, row.Card, statusLabel(r), row.Release, row.Fits, row.Cached)
	}
	tw.Flush()
	if derr != nil {
		fmt.Fprintf(a.Stderr, "fits is unknown: %s\n", detectReason(derr))
	}
	return nil
}

type infoDownload struct {
	URL      string `json:"url"`
	Blob     string `json:"blob"`
	Snapshot string `json:"snapshot"`
	Size     int64  `json:"size_bytes"`
	Cached   string `json:"cached"`
}

type infoOut struct {
	Recipe      *recipe.Recipe `json:"recipe"`
	Published   bool           `json:"published"`
	Unpublished []string       `json:"unpublished"`
	Fits        string         `json:"fits"`
	Untuned     string         `json:"untuned,omitempty"`
	Download    infoDownload   `json:"download"`
	Runtime     string         `json:"runtime"`
	GPU         string         `json:"gpu"`
	Threads     int            `json:"threads"`
	Command     []string       `json:"command"`
	Endpoint    string         `json:"endpoint"`
}

func (a *App) cmdInfo(ctx context.Context, o *options) error {
	set, err := a.loadRecipes(o)
	if err != nil {
		return err
	}
	rs, err := set.ForModel(o.model)
	if err != nil {
		return err
	}
	gpus, derr := gpu.Detect(ctx, a.Runner)
	var t *target
	switch {
	case o.card != "" || len(rs) == 1:
		r := rs[0]
		if o.card != "" {
			if r, err = set.Lookup(o.model, o.card); err != nil {
				return err
			}
		}
		t = &target{r: r, device: container.AllGPUs}
		if derr == nil {
			g, _ := pick(gpus, []*recipe.Recipe{r}, o.gpu)
			t.device = device(gpus, r, g, o.gpu)
		} else if o.gpu >= 0 {
			t.device = o.gpu
		}
	default:
		if t, err = a.resolve(ctx, set, o, true); err != nil {
			return err
		}
	}
	t = withRevision(t, o)
	r := t.r
	rt, _, _ := a.runtimeFor(o, true)
	cache, err := a.cacheDir(o)
	if err != nil {
		return err
	}
	f := modelFile(r)
	spec, err := a.spec(t, o, rt, hfcache.BlobPath(cache, f))
	if err != nil {
		return err
	}
	out := infoOut{
		Recipe:      r,
		Published:   r.Published(),
		Unpublished: append([]string{}, r.Unpublished()...),
		Fits:        fits(gpus, derr, r),
		Untuned:     untuned(t, o),
		Download: infoDownload{
			URL:      hfcache.URL(a.getenv("HF_ENDPOINT"), f),
			Blob:     hfcache.BlobPath(cache, f),
			Snapshot: hfcache.SnapshotPath(cache, f),
			Size:     f.Size,
			Cached:   hfcache.Status(cache, f),
		},
		Runtime:  rt,
		GPU:      deviceName(t.device),
		Threads:  spec.Threads,
		Command:  append([]string{rt}, spec.RunArgs()...),
		Endpoint: container.BaseURL(o.host, o.port),
	}
	if o.json {
		return writeJSON(a.Stdout, out)
	}
	a.printInfo(out, derr)
	return nil
}

func (a *App) printInfo(out infoOut, derr error) {
	r := out.Recipe
	w := tabwriter.NewWriter(a.Stdout, 0, 0, 2, ' ', 0)
	kv := func(k, v string) { fmt.Fprintf(w, "  %s\t%s\n", k, v) }
	section := func(name string) { w.Flush(); fmt.Fprintf(a.Stdout, "\n%s\n", name) }

	fmt.Fprintf(a.Stdout, "%s on %s: %s\n", r.Repo(), r.Card.ID, r.Model.Title)
	if r.Model.Name != "" {
		kv("short name", r.Model.Name)
	}
	kv("status", r.Status)
	kv("recipe", r.Source)
	fit := out.Fits
	if derr != nil {
		fit += " (" + detectReason(derr) + ")"
	}
	kv("fits this host", fit)
	if !out.Published {
		kv("NOT PUBLISHED", joinAnd(out.Unpublished)+" "+plural(len(out.Unpublished), "is a placeholder", "are placeholders")+"; pull, run and serve refuse this recipe")
	}
	if out.Untuned != "" {
		kv("UNTUNED", out.Untuned)
	}

	section("Model")
	kv("repository", r.Model.HFRepo)
	kv("revision", r.Model.Revision)
	kv("file", r.Model.File)
	kv("sha256", r.Model.SHA256)
	size := "unknown (size_bytes is 0)"
	if r.Model.SizeBytes > 0 {
		size = fmt.Sprintf("%s (%d bytes)", hfcache.FormatBytes(r.Model.SizeBytes), r.Model.SizeBytes)
	}
	kv("size", size)
	kv("license", r.Model.License)
	kv("url", out.Download.URL)
	kv("cache", out.Download.Blob)
	kv("cached", out.Download.Cached)

	section("Card")
	kv("title", r.Card.Title)
	kv("compute", r.Card.ComputeCapability)
	kv("memory", fmt.Sprintf("at least %d MiB", r.Card.MinMemoryMiB))
	kv("driver", r.Card.MinDriver+" or newer")

	section("Engine")
	kv("release", r.Engine.Release)
	kv("arch", r.Engine.Arch)
	kv("image", r.Engine.Image)

	section("Run")
	kv("alias", r.Run.Alias)
	kv("context", fmt.Sprint(r.Run.Context))
	kv("threads", fmt.Sprintf("%d (threads_max %d, %d CPUs here)", out.Threads, r.Run.ThreadsMax, a.NumCPU()))
	kv("args", shellJoin(r.Run.Args))
	for i, k := range r.EnvKeys() {
		label := ""
		if i == 0 {
			label = "env"
		}
		kv(label, k+"="+r.Run.Env[k])
	}
	w.Flush()
	fmt.Fprintf(a.Stdout, "  command (%s, GPU %s, endpoint %s):\n", out.Runtime, out.GPU, out.Endpoint)
	fmt.Fprintf(a.Stdout, "    %s\n", strings.ReplaceAll(shellWrap(out.Command, 96), "\n", "\n  "))

	rc := r.Receipt
	section("Receipt")
	if rc.Source == "" && rc.Summary == "" && len(rc.DecodeTPS) == 0 && len(rc.PrefillTPS) == 0 && len(rc.Notes) == 0 {
		kv("", "none yet")
	}
	if rc.Summary != "" {
		kv("summary", rc.Summary)
	}
	if rc.Source != "" {
		kv("source", rc.Source)
	}
	if len(rc.DecodeTPS) > 0 {
		kv("decode", measurements(rc.DecodeTPS))
	}
	if len(rc.PrefillTPS) > 0 {
		kv("prefill", measurements(rc.PrefillTPS))
	}
	w.Flush()
	if len(rc.Notes) > 0 {
		fmt.Fprintln(a.Stdout, "  notes:")
		for _, n := range rc.Notes {
			fmt.Fprintf(a.Stdout, "    - %s\n", n)
		}
	}
}

func measurements(m map[string]float64) string {
	var parts []string
	for _, k := range recipe.MeasurementKeys(m) {
		parts = append(parts, fmt.Sprintf("%s %.2f tok/s", k, m[k]))
	}
	return strings.Join(parts, ", ")
}

type capCard struct {
	ID                string `json:"id"`
	Title             string `json:"title"`
	ComputeCapability string `json:"compute_capability"`
	MinMemoryMiB      int    `json:"min_memory_mib"`
	MinDriver         string `json:"min_driver"`
	Arch              string `json:"arch"`
}

type capRecipe struct {
	Model     string `json:"model"` // the repo id
	Name      string `json:"name,omitempty"`
	Card      string `json:"card"`
	Status    string `json:"status"`
	Published bool   `json:"published"`
	Release   string `json:"release"`
	Arch      string `json:"arch"`
}

type capsOut struct {
	Name         string      `json:"name"`
	Version      string      `json:"version"`
	Commit       string      `json:"commit"`
	Platform     string      `json:"platform"`
	RecipeSchema int         `json:"recipe_schema"`
	Runtimes     []string    `json:"runtimes"`
	Cards        []capCard   `json:"cards"`
	Recipes      []capRecipe `json:"recipes"`
}

// cmdCapabilities runs nothing external: it must work with no GPU, driver,
// runtime or network.
func (a *App) cmdCapabilities(_ context.Context, o *options) error {
	set, err := a.loadRecipes(o)
	if err != nil {
		return err
	}
	out := capsOut{
		Name: "tensorless", Version: a.Version, Commit: a.Commit,
		Platform:     runtime.GOOS + "/" + runtime.GOARCH,
		RecipeSchema: recipe.Schema,
		Runtimes:     container.Runtimes,
		Cards:        []capCard{},
		Recipes:      []capRecipe{},
	}
	seen := map[string]bool{}
	for _, r := range set.Recipes {
		out.Recipes = append(out.Recipes, capRecipe{
			Model: r.Repo(), Name: r.Model.Name, Card: r.Card.ID, Status: r.Status, Published: r.Published(),
			Release: r.Engine.Release, Arch: r.Engine.Arch,
		})
		if !seen[r.Card.ID] {
			seen[r.Card.ID] = true
			out.Cards = append(out.Cards, capCard{
				ID: r.Card.ID, Title: r.Card.Title, ComputeCapability: r.Card.ComputeCapability,
				MinMemoryMiB: r.Card.MinMemoryMiB, MinDriver: r.Card.MinDriver, Arch: r.Engine.Arch,
			})
		}
	}
	if o.json {
		return writeJSON(a.Stdout, out)
	}
	fmt.Fprintln(a.Stdout, a.versionString())
	fmt.Fprintf(a.Stdout, "recipe schema: %d\nruntimes: %s\n\ncards:\n", out.RecipeSchema, strings.Join(out.Runtimes, ", "))
	tw := tabwriter.NewWriter(a.Stdout, 0, 0, 2, ' ', 0)
	for _, c := range out.Cards {
		fmt.Fprintf(tw, "  %s\t%s\tcompute %s, at least %d MiB, driver %s+\t%s\n", c.ID, c.Title, c.ComputeCapability, c.MinMemoryMiB, c.MinDriver, c.Arch)
	}
	tw.Flush()
	fmt.Fprintln(a.Stdout, "\nrecipes:")
	for _, r := range set.Recipes {
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%s\n", r.Repo(), r.Card.ID, statusLabel(r), r.Engine.Release, r.Engine.Arch)
	}
	return tw.Flush()
}
