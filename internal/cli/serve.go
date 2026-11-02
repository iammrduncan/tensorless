package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/iammrduncan/tensorless/internal/container"
	"github.com/iammrduncan/tensorless/internal/hfcache"
	"github.com/iammrduncan/tensorless/internal/recipe"
)

// plan prints a dry run.
type plan struct {
	w io.Writer
	n int
}

func (p *plan) step(format string, args ...any) {
	p.n++
	fmt.Fprintf(p.w, "\n# %d. %s\n", p.n, fmt.Sprintf(format, args...))
}

func (p *plan) cmd(argv ...string) { fmt.Fprintln(p.w, shellWrap(argv, 100)) }

func (p *plan) line(format string, args ...any) { fmt.Fprintf(p.w, format+"\n", args...) }

func (a *App) newPlan(cmd string, t *target, o *options, rt, rtNote string) *plan {
	p := &plan{w: a.Stdout}
	r := t.r
	p.line("dry run of 'tensorless %s %s': nothing is pulled, downloaded or started.", cmd, modelArg(t, o))
	p.line("recipe:   %s on %s (%s), from %s", r.Repo(), r.Card.ID, r.Status, r.Source)
	p.line("gpu:      %s; the container gets GPU %s", t.how, deviceName(t.device))
	if rtNote != "" {
		p.line("runtime:  %s (%s)", rt, rtNote)
	} else {
		p.line("runtime:  %s", rt)
	}
	if u := r.Unpublished(); len(u) > 0 {
		p.line("NOT PUBLISHED: the recipe's %s %s. A real pull, run or serve refuses it until they are filled in.",
			joinAnd(u), plural(len(u), "is a placeholder", "are placeholders"))
	}
	if why := untuned(t, o); why != "" && (cmd != "pull" || t.pinned != "") {
		p.line("UNTUNED: %s; the receipt's numbers do not apply to this run.", why)
	}
	return p
}

func (a *App) planPull(p *plan, rt string, r *recipe.Recipe, cache string) {
	p.step("pull the engine image, unless '%s image inspect' finds it", rt)
	p.cmd(rt, "pull", r.Engine.Image)

	f := modelFile(r)
	blob := hfcache.BlobPath(cache, f)
	link := hfcache.SnapshotPath(cache, f)
	if hfcache.Status(cache, f) == "yes" {
		p.step("the model is in the cache: verify its sha256 (once) and link it into the snapshot")
		p.line("verify %s", blob)
	} else {
		size := "size unknown: size_bytes is 0"
		if f.Size > 0 {
			size = hfcache.FormatBytes(f.Size)
		}
		p.step("download the model (%s), resuming a partial download, and check its sha256 %s", size, f.SHA256)
		p.line("GET %s", hfcache.URL(a.getenv("HF_ENDPOINT"), f))
		if a.getenv("HF_TOKEN") != "" {
			p.line("    with Authorization: Bearer (HF_TOKEN, not shown)")
		}
		p.line("    into %s", blob)
	}
	p.line("link %s -> %s", link, hfcache.LinkTarget(f))
}

func (a *App) planServe(p *plan, rt string, spec container.Spec, t *target, o *options, pulled bool) {
	if !pulled {
		p.step("check that the image is pulled and the model is cached and verified (else: tensorless pull %s)", modelArg(t, o))
		p.cmd(rt, "image", "inspect", spec.Image)
		p.line("verify %s", spec.ModelPath)
	}
	p.step("start the server")
	p.cmd(append([]string{rt}, spec.RunArgs()...)...)
	base := container.BaseURL(o.host, o.port)
	if o.detach {
		p.step("wait for %s/health (up to %s), then return; stop it with: tensorless stop %s", base, duration(a.HealthTimeout), t.r.Repo())
		return
	}
	p.step("wait for %s/health (up to %s) and follow the logs; Ctrl-C stops the server", base, duration(a.HealthTimeout))
	p.cmd(append([]string{rt}, container.LogsArgs(spec.Name)...)...)
	p.cmd(append([]string{rt}, container.StopArgs(spec.Name)...)...)
}

func (a *App) cmdPull(ctx context.Context, o *options) error {
	t, err := a.prepare(ctx, o)
	if err != nil {
		return err
	}
	rt, rtNote, err := a.runtimeFor(o, o.dryRun)
	if err != nil {
		return err
	}
	cache, err := a.cacheDir(o)
	if err != nil {
		return err
	}
	if o.dryRun {
		a.planPull(a.newPlan("pull", t, o, rt, rtNote), rt, t.r, cache)
		return nil
	}
	if err := a.pullImage(ctx, rt, t.r.Engine.Image); err != nil {
		return err
	}
	blob, err := a.pullModel(ctx, t, cache)
	if err != nil {
		return err
	}
	fmt.Fprintf(a.Stdout, "pulled %s for %s\n  image: %s\n  model: %s\nstart it with: tensorless serve %s\n",
		t.r.Repo(), t.r.Card.ID, t.r.Engine.Image, blob, modelArg(t, o))
	return nil
}

func (a *App) pullImage(ctx context.Context, rt, image string) error {
	if container.ImagePresent(ctx, a.Runner, rt, image) {
		fmt.Fprintf(a.Stderr, "the engine image is already pulled: %s\n", image)
		return nil
	}
	fmt.Fprintf(a.Stderr, "pulling the engine image %s\n", image)
	if err := a.Runner.Run(ctx, a.Stderr, a.Stderr, rt, "pull", image); err != nil {
		if ctx.Err() != nil {
			return errInterrupted
		}
		return fmt.Errorf("could not pull the engine image: %v. Check that this host can reach the registry, then try again", err)
	}
	return nil
}

func (a *App) pullModel(ctx context.Context, t *target, cache string) (string, error) {
	r := t.r
	fmt.Fprintf(a.Stderr, "model %s from %s at %s\n", r.Model.File, r.Model.HFRepo, r.Model.Revision[:12])
	blob, err := a.downloader().Fetch(ctx, cache, modelFile(r))
	if err != nil && t.pinned != "" && strings.Contains(err.Error(), "sha256 mismatch") {
		err = fmt.Errorf("%w. The file at revision %s is not the one the recipe pins at %s", err, r.Model.Revision, t.pinned)
	}
	return blob, err
}

func (a *App) cmdServe(ctx context.Context, o *options) error { return a.serve(ctx, o, false) }

func (a *App) cmdRun(ctx context.Context, o *options) error { return a.serve(ctx, o, true) }

// serve starts the server; with pull, it pulls first (tensorless run).
func (a *App) serve(ctx context.Context, o *options, pull bool) error {
	cmd := "serve"
	if pull {
		cmd = "run"
	}
	t, err := a.prepare(ctx, o)
	if err != nil {
		return err
	}
	rt, rtNote, err := a.runtimeFor(o, o.dryRun)
	if err != nil {
		return err
	}
	cache, err := a.cacheDir(o)
	if err != nil {
		return err
	}
	spec, err := a.spec(t, o, rt, hfcache.BlobPath(cache, modelFile(t.r)))
	if err != nil {
		return err
	}
	if o.dryRun {
		p := a.newPlan(cmd, t, o, rt, rtNote)
		if pull {
			a.planPull(p, rt, t.r, cache)
		}
		a.planServe(p, rt, spec, t, o, pull)
		return nil
	}
	if why := untuned(t, o); why != "" {
		fmt.Fprintf(a.Stderr, "UNTUNED: %s. This is not the configuration the receipt measured; its numbers do not apply.\n", why)
	}
	switch container.State(ctx, a.Runner, rt, spec.Name) {
	case container.Running:
		return fmt.Errorf("%s is already running in %s. Stop it with: tensorless stop %s", spec.Name, rt, t.r.Repo())
	case container.Stopped:
		fmt.Fprintf(a.Stderr, "removing the stopped container %s\n", spec.Name)
		if _, err := a.Runner.Output(ctx, rt, "rm", spec.Name); err != nil {
			return fmt.Errorf("could not remove the stopped container %s: %v", spec.Name, err)
		}
	}
	if a.PortFree != nil {
		if err := a.PortFree(o.host, o.port); err != nil {
			return fmt.Errorf("cannot use port %d on %s: %v. Pass --port to choose another one", o.port, o.host, err)
		}
	}

	var blob string
	if pull {
		if err := a.pullImage(ctx, rt, t.r.Engine.Image); err != nil {
			return err
		}
		if blob, err = a.pullModel(ctx, t, cache); err != nil {
			return err
		}
	} else {
		if !container.ImagePresent(ctx, a.Runner, rt, t.r.Engine.Image) {
			return fmt.Errorf("the engine image is not pulled. Run: tensorless pull %s", modelArg(t, o))
		}
		blob, err = a.downloader().Verify(ctx, cache, modelFile(t.r))
		if errors.Is(err, hfcache.ErrNotCached) {
			return fmt.Errorf("the model is not downloaded. Run: tensorless pull %s", modelArg(t, o))
		}
		if err != nil {
			return err
		}
	}
	spec.ModelPath = blob
	return a.startAndWait(ctx, rt, spec, t, o)
}

var (
	errExited  = errors.New("the container exited")
	errTimeout = errors.New("timed out")
)

func (a *App) startAndWait(ctx context.Context, rt string, spec container.Spec, t *target, o *options) error {
	name := spec.Name
	fmt.Fprintf(a.Stderr, "starting %s (%s, GPU %s)\n", name, rt, deviceName(spec.GPU))
	if _, err := a.Runner.Output(ctx, rt, spec.RunArgs()...); err != nil {
		if ctx.Err() != nil {
			if container.State(context.Background(), a.Runner, rt, name) == container.Running {
				a.stopContainer(rt, name)
			}
			return errInterrupted
		}
		return fmt.Errorf("could not start the container: %v. Run 'tensorless doctor' to check GPU support in %s", err, rt)
	}

	logs := newTail(60)
	var stdout, stderr io.Writer = logs, logs
	if !o.detach {
		stdout, stderr = io.MultiWriter(a.Stdout, logs), io.MultiWriter(a.Stderr, logs)
	}
	logsCtx, cancelLogs := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelLogs()
	logsDone := make(chan error, 1)
	go func() { logsDone <- a.Runner.Run(logsCtx, stdout, stderr, rt, container.LogsArgs(name)...) }()
	waitLogs := func() {
		select {
		case <-logsDone:
		case <-time.After(3 * time.Second):
		}
	}

	base := container.BaseURL(o.host, o.port)
	if o.detach {
		fmt.Fprintf(a.Stderr, "waiting for %s/health while the model loads (up to %s)\n", base, duration(a.HealthTimeout))
	}
	running := func() bool {
		return container.State(context.WithoutCancel(ctx), a.Runner, rt, name) == container.Running
	}
	err := a.waitHealthy(ctx, base+"/health", running, o.detach)
	switch {
	case err == nil:
	case ctx.Err() != nil:
		a.stopContainer(rt, name)
		return errInterrupted
	case errors.Is(err, errExited):
		waitLogs()
		a.printTail(logs, name, o.detach)
		return fmt.Errorf("the server exited before it was ready%s. Run 'tensorless doctor', then try again", tailHint(o.detach))
	default:
		a.stopContainer(rt, name)
		waitLogs()
		a.printTail(logs, name, o.detach)
		return fmt.Errorf("the server was not ready after %s, so it was stopped%s", duration(a.HealthTimeout), tailHint(o.detach))
	}

	a.printReady(t, o, base)
	if o.detach {
		return nil
	}
	select {
	case <-ctx.Done():
	case <-logsDone:
		// A Ctrl-C also ends the log stream; give the signal a moment.
		select {
		case <-ctx.Done():
		case <-time.After(500 * time.Millisecond):
			if running() {
				fmt.Fprintf(a.Stderr, "the log stream ended, but %s is still running. Follow it with: %s logs -f %s; stop it with: tensorless stop %s\n",
					name, rt, name, t.r.Repo())
				return nil
			}
			return fmt.Errorf("the server stopped; its last log lines are above")
		}
	}
	a.stopContainer(rt, name)
	waitLogs()
	return nil
}

// duration formats d as 20m rather than 20m0s.
func duration(d time.Duration) string {
	if d >= time.Minute && d%time.Minute == 0 {
		return fmt.Sprintf("%dm", d/time.Minute)
	}
	return d.String()
}

func tailHint(detach bool) string {
	if detach {
		return "; the last log lines are above"
	}
	return ""
}

// printTail prints the last log lines; in the foreground they were already
// shown as they came.
func (a *App) printTail(t *tail, name string, detach bool) {
	if !detach {
		return
	}
	lines := t.Lines()
	if len(lines) == 0 {
		fmt.Fprintf(a.Stderr, "%s left no log lines\n", name)
		return
	}
	fmt.Fprintf(a.Stderr, "last %d log lines of %s:\n", len(lines), name)
	for _, l := range lines {
		fmt.Fprintln(a.Stderr, "  "+l)
	}
}

func (a *App) stopContainer(rt, name string) {
	fmt.Fprintf(a.Stderr, "stopping %s\n", name)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	if _, err := a.Runner.Output(ctx, rt, container.StopArgs(name)...); err != nil {
		fmt.Fprintf(a.Stderr, "could not stop %s: %v. Stop it with: %s stop %s\n", name, err, rt, name)
	}
}

func localClient(timeout time.Duration) *http.Client {
	return &http.Client{Timeout: timeout, Transport: &http.Transport{}} // no proxy for a local server
}

func healthy(ctx context.Context, c *http.Client, url string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := c.Do(req)
	if err != nil {
		return false
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// waitHealthy polls url until it answers 200, the container stops running,
// the timeout passes, or ctx is done.
func (a *App) waitHealthy(ctx context.Context, url string, running func() bool, chatty bool) error {
	c := localClient(5 * time.Second)
	start := time.Now()
	next := start.Add(30 * time.Second)
	for {
		if healthy(ctx, c, url) {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if !running() {
			return errExited
		}
		if time.Since(start) > a.HealthTimeout {
			return errTimeout
		}
		if chatty && time.Now().After(next) {
			fmt.Fprintf(a.Stderr, "still loading (%s)\n", time.Since(start).Round(time.Second))
			next = next.Add(30 * time.Second)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(a.HealthInterval):
		}
	}
}

func (a *App) printReady(t *target, o *options, base string) {
	alias := t.r.Run.Alias
	w := a.Stdout
	fmt.Fprintf(w, "\n%s is ready at %s (OpenAI-compatible API under /v1)\n", alias, base)
	if o.host == "0.0.0.0" || o.host == "::" {
		fmt.Fprintf(w, "it listens on every interface of this host (--host %s)\n", o.host)
	}
	body := fmt.Sprintf(`{"model": %q, "messages": [{"role": "user", "content": "Hello"}]}`, alias)
	fmt.Fprintf(w, "try it:\n  curl %s/v1/chat/completions -H 'Content-Type: application/json' \\\n    -d %s\n", base, quote(body))
	if o.detach {
		fmt.Fprintf(w, "stop it with: tensorless stop %s\n", t.r.Repo())
	} else {
		fmt.Fprintln(w, "Ctrl-C stops the server.")
	}
}

func (a *App) cmdStop(ctx context.Context, o *options) error {
	set, err := a.loadRecipes(o)
	if err != nil {
		return err
	}
	rs, err := set.ForModel(o.model)
	if err != nil {
		return err
	}
	if o.card != "" {
		r, err := set.Lookup(o.model, o.card)
		if err != nil {
			return err
		}
		rs = []*recipe.Recipe{r}
	}
	var names []string
	for _, r := range rs {
		n := container.Name(r.Repo())
		if !contains(names, n) {
			names = append(names, n)
		}
	}
	// With auto, look in every installed runtime: the server may have been
	// started with either.
	var rts []string
	if o.runtime == "" || o.runtime == "auto" {
		for _, rt := range container.Runtimes {
			if _, err := a.Runner.LookPath(rt); err == nil {
				rts = append(rts, rt)
			}
		}
	}
	if len(rts) == 0 {
		rt, _, err := a.runtimeFor(o, o.dryRun)
		if err != nil {
			return err
		}
		rts = []string{rt}
	}
	stopped := 0
	for _, rt := range rts {
		for _, n := range names {
			if o.dryRun {
				fmt.Fprintln(a.Stdout, shellJoin(append([]string{rt}, container.StopArgs(n)...)))
				continue
			}
			if container.State(ctx, a.Runner, rt, n) != container.Running {
				continue
			}
			if _, err := a.Runner.Output(ctx, rt, container.StopArgs(n)...); err != nil {
				return fmt.Errorf("could not stop %s: %v", n, err)
			}
			fmt.Fprintf(a.Stdout, "stopped %s\n", n)
			stopped++
		}
	}
	if stopped == 0 && !o.dryRun {
		fmt.Fprintf(a.Stdout, "%s is not running\n", strings.Join(names, ", "))
	}
	return nil
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}
