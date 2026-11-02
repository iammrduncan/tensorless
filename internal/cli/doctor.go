package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/iammrduncan/tensorless/internal/container"
	"github.com/iammrduncan/tensorless/internal/gpu"
	"github.com/iammrduncan/tensorless/internal/hfcache"
	"github.com/iammrduncan/tensorless/internal/recipe"
)

// CDISpecs are where podman and docker look for the NVIDIA CDI spec.
var CDISpecs = []string{"/etc/cdi/nvidia.yaml", "/var/run/cdi/nvidia.yaml"}

const cdiFix = "sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"

type doctor struct {
	a     *App
	fails int
}

func (d *doctor) report(status, what, fix string) {
	fmt.Fprintf(d.a.Stdout, "%-4s  %s\n", status, what)
	if fix != "" {
		fmt.Fprintf(d.a.Stdout, "      fix: %s\n", fix)
	}
	if status == "FAIL" {
		d.fails++
	}
}

func (a *App) cmdDoctor(ctx context.Context, o *options) error {
	set, err := a.loadRecipes(o)
	if err != nil {
		return err
	}
	d := &doctor{a: a}

	// The driver and the GPUs.
	minDriver := ""
	for _, r := range set.Recipes {
		if c, err := gpu.CompareVersions(r.Card.MinDriver, minDriver); minDriver == "" || (err == nil && c < 0) {
			minDriver = r.Card.MinDriver
		}
	}
	gpus, derr := gpu.Detect(ctx, a.Runner)
	var fitting []*recipe.Recipe
	switch {
	case errors.Is(derr, gpu.ErrNoNvidiaSMI):
		d.report("FAIL", "nvidia-smi not found: no NVIDIA driver is installed", "install the NVIDIA driver, "+minDriver+" or newer, and reboot")
	case derr != nil:
		d.report("FAIL", "nvidia-smi failed: "+derr.Error(), "make sure the NVIDIA driver is installed and loaded (reboot after installing it), then run: nvidia-smi")
	default:
		d.report("PASS", fmt.Sprintf("nvidia-smi lists %d GPU%s", len(gpus), plural(len(gpus), "", "s")), "")
		for _, g := range gpus {
			var names []string
			for _, r := range set.Recipes {
				if g.Matches(r.Card.ComputeCapability, r.Card.MinMemoryMiB) {
					names = append(names, describe(r))
					if !containsRecipe(fitting, r) {
						fitting = append(fitting, r)
					}
				}
			}
			if len(names) > 0 {
				d.report("PASS", fmt.Sprintf("%s; fits %s", g, strings.Join(names, ", ")), "")
			} else {
				d.report("WARN", fmt.Sprintf("%s; fits no recipe", g), "")
			}
		}
		if len(fitting) == 0 {
			var need []string
			for _, c := range cardsOf(set) {
				need = append(need, requirement(c))
			}
			d.report("FAIL", "no GPU here fits a recipe; known cards: "+strings.Join(need, "; "), "none: there is no recipe for this card yet")
		}
		// One driver serves every GPU.
		driver := gpus[0].Driver
		need, needBy := "", ""
		for _, r := range fitting {
			if c, err := gpu.CompareVersions(r.Card.MinDriver, need); need == "" || (err == nil && c > 0) {
				need, needBy = r.Card.MinDriver, describe(r)
			}
		}
		if need != "" {
			ok, err := gpu.DriverAtLeast(driver, need)
			switch {
			case err != nil:
				d.report("FAIL", fmt.Sprintf("cannot read the driver version %q: %v", driver, err), "reinstall the NVIDIA driver")
			case ok:
				d.report("PASS", fmt.Sprintf("driver %s is at least %s (needed by %s)", driver, need, needBy), "")
			default:
				d.report("FAIL", fmt.Sprintf("driver %s is older than %s (needed by %s)", driver, need, needBy), "install NVIDIA driver "+need+" or newer, and reboot")
			}
		}
	}

	// The container runtime and its NVIDIA support.
	rt, rerr := container.Select(a.Runner, o.runtime)
	if rerr != nil {
		d.report("FAIL", rerr.Error(), "install podman: sudo apt install podman (Debian, Ubuntu) or sudo dnf install podman (Fedora, RHEL)")
	} else {
		path, _ := a.Runner.LookPath(rt)
		d.report("PASS", fmt.Sprintf("container runtime: %s (%s)", rt, path), "")
		a.checkNvidiaContainer(ctx, d, rt)
	}

	// Disk space in the model cache.
	a.checkDisk(d, o, set, fitting, derr == nil)

	if d.fails > 0 {
		fmt.Fprintf(a.Stdout, "\n%d check%s failed.\n", d.fails, plural(d.fails, "", "s"))
		return errFailed
	}
	fmt.Fprintln(a.Stdout, "\nall checks passed.")
	return nil
}

func containsRecipe(rs []*recipe.Recipe, r *recipe.Recipe) bool {
	for _, x := range rs {
		if x == r {
			return true
		}
	}
	return false
}

// cardsOf returns one recipe per card, for describing card requirements.
func cardsOf(set *recipe.Set) []*recipe.Recipe {
	seen := map[string]bool{}
	var out []*recipe.Recipe
	for _, r := range set.Recipes {
		if !seen[r.Card.ID] {
			seen[r.Card.ID] = true
			out = append(out, r)
		}
	}
	return out
}

func (a *App) cdiSpec() string {
	for _, p := range CDISpecs {
		if a.FileExists(p) {
			return p
		}
	}
	return ""
}

func (a *App) checkNvidiaContainer(ctx context.Context, d *doctor, rt string) {
	cdi := a.cdiSpec()
	_, ctkErr := a.Runner.LookPath("nvidia-ctk")
	if rt == "podman" {
		if cdi != "" {
			d.report("PASS", "NVIDIA CDI spec for podman: "+cdi, "")
			return
		}
		fix := cdiFix
		if ctkErr != nil {
			fix = "install nvidia-container-toolkit, then: " + cdiFix
		}
		d.report("FAIL", "no NVIDIA CDI spec for podman in "+strings.Join(CDISpecs, " or "), fix)
		return
	}
	out, err := a.Runner.Output(ctx, "docker", "info", "--format", "{{json .Runtimes}}")
	if err != nil {
		d.report("FAIL", "docker info failed: "+err.Error(), "start docker (sudo systemctl start docker) and make sure you may use it (the docker group)")
		return
	}
	_, hookErr := a.Runner.LookPath("nvidia-container-runtime-hook")
	switch {
	case strings.Contains(string(out), "nvidia"):
		d.report("PASS", "docker has the nvidia runtime", "")
	case cdi != "":
		d.report("PASS", "NVIDIA CDI spec for docker: "+cdi, "")
	case hookErr == nil:
		d.report("PASS", "the NVIDIA container toolkit hook is installed, which docker --gpus uses", "")
	default:
		d.report("FAIL", "docker has no NVIDIA GPU support (no nvidia runtime, no CDI spec)",
			"install nvidia-container-toolkit, then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker")
	}
}

func (a *App) checkDisk(d *doctor, o *options, set *recipe.Set, fitting []*recipe.Recipe, detected bool) {
	cache, err := a.cacheDir(o)
	if err != nil {
		d.report("WARN", err.Error(), "")
		return
	}
	candidates := fitting
	if !detected {
		candidates = set.Recipes
	}
	var need int64
	for _, r := range candidates {
		if hfcache.Status(cache, modelFile(r)) != "yes" && r.Model.SizeBytes > need {
			need = r.Model.SizeBytes
		}
	}
	dir := existingAncestor(cache)
	free, err := a.FreeSpace(dir)
	if err != nil {
		d.report("WARN", fmt.Sprintf("cannot read the free space in %s: %v", dir, err), "")
		return
	}
	switch {
	case need > 0 && free < uint64(need):
		d.report("FAIL", fmt.Sprintf("model cache %s: %s free, the model needs %s", cache, hfcache.FormatBytes(int64(free)), hfcache.FormatBytes(need)),
			"free some space, or pass --cache DIR (or set HF_HUB_CACHE) on a larger disk")
	case need > 0:
		d.report("PASS", fmt.Sprintf("model cache %s: %s free, the model needs %s", cache, hfcache.FormatBytes(int64(free)), hfcache.FormatBytes(need)), "")
	default:
		d.report("PASS", fmt.Sprintf("model cache %s: %s free", cache, hfcache.FormatBytes(int64(free))), "")
	}
}

func existingAncestor(dir string) string {
	for {
		if _, err := os.Stat(dir); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return dir
		}
		dir = parent
	}
}
