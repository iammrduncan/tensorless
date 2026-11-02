// Package gpu detects NVIDIA GPUs with nvidia-smi and matches them against
// recipe requirements.
package gpu

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/iammrduncan/tensorless/internal/execx"
)

// GPU is one line of nvidia-smi's query output.
type GPU struct {
	Index      int    `json:"index"`
	Name       string `json:"name"`
	ComputeCap string `json:"compute_cap"`
	MemoryMiB  int    `json:"memory_mib"`
	Driver     string `json:"driver"`
	UUID       string `json:"uuid"`
}

func (g GPU) String() string {
	return fmt.Sprintf("GPU %d: %s, compute %s, %d MiB, driver %s", g.Index, g.Name, g.ComputeCap, g.MemoryMiB, g.Driver)
}

// QueryArgs are the nvidia-smi arguments Detect uses.
var QueryArgs = []string{
	"--query-gpu=index,name,compute_cap,memory.total,driver_version,uuid",
	"--format=csv,noheader,nounits",
}

// ErrNoNvidiaSMI means nvidia-smi is not installed or not on PATH.
var ErrNoNvidiaSMI = errors.New("nvidia-smi not found")

// Detect lists the GPUs that nvidia-smi reports.
func Detect(ctx context.Context, r execx.Runner) ([]GPU, error) {
	if _, err := r.LookPath("nvidia-smi"); err != nil {
		return nil, ErrNoNvidiaSMI
	}
	out, err := r.Output(ctx, "nvidia-smi", QueryArgs...)
	if err != nil {
		if strings.Contains(err.Error(), "compute_cap") {
			return nil, fmt.Errorf("this driver's nvidia-smi cannot report the compute capability; install a newer driver: %w", err)
		}
		return nil, err
	}
	gpus, err := Parse(out)
	if err != nil {
		return nil, err
	}
	if len(gpus) == 0 {
		return nil, errors.New("nvidia-smi lists no GPUs")
	}
	return gpus, nil
}

// Parse reads nvidia-smi's csv,noheader,nounits output for QueryArgs. Fields
// that nvidia-smi cannot report ("[N/A]") are left empty or zero, so a GPU with
// them never matches a recipe.
func Parse(out []byte) ([]GPU, error) {
	var gpus []GPU
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		raw := strings.Split(line, ",")
		f := make([]string, len(raw))
		for i := range raw {
			f[i] = strings.TrimSpace(raw[i])
		}
		n := len(f)
		if n < 6 {
			return nil, fmt.Errorf("unexpected nvidia-smi output: %q", line)
		}
		idx, err := strconv.Atoi(f[0])
		if err != nil {
			return nil, fmt.Errorf("unexpected nvidia-smi output: %q", line)
		}
		g := GPU{
			Index:      idx,
			Name:       strings.TrimSpace(strings.Join(raw[1:n-4], ",")), // a name could contain a comma
			ComputeCap: known(f[n-4]),
			Driver:     known(f[n-2]),
			UUID:       known(f[n-1]),
		}
		if mem, err := strconv.Atoi(f[n-3]); err == nil {
			g.MemoryMiB = mem
		}
		gpus = append(gpus, g)
	}
	return gpus, nil
}

func known(s string) string {
	if strings.HasPrefix(s, "[") { // [N/A], [Not Supported]
		return ""
	}
	return s
}

// Matches reports whether g has the compute capability and at least the
// memory that a card needs.
func (g GPU) Matches(computeCap string, minMemoryMiB int) bool {
	return g.ComputeCap != "" && g.ComputeCap == computeCap && g.MemoryMiB >= minMemoryMiB
}

// CompareVersions compares dotted numeric versions such as 580.95.05. Missing
// parts count as 0. It returns -1, 0 or 1.
func CompareVersions(a, b string) (int, error) {
	pa, err := versionParts(a)
	if err != nil {
		return 0, err
	}
	pb, err := versionParts(b)
	if err != nil {
		return 0, err
	}
	for i := 0; i < max(len(pa), len(pb)); i++ {
		var x, y int
		if i < len(pa) {
			x = pa[i]
		}
		if i < len(pb) {
			y = pb[i]
		}
		switch {
		case x < y:
			return -1, nil
		case x > y:
			return 1, nil
		}
	}
	return 0, nil
}

func versionParts(v string) ([]int, error) {
	if v == "" {
		return nil, errors.New("empty version")
	}
	parts := strings.Split(v, ".")
	out := make([]int, len(parts))
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("version %q is not dotted numbers", v)
		}
		out[i] = n
	}
	return out, nil
}

// DriverAtLeast reports whether driver version have is min or newer.
func DriverAtLeast(have, min string) (bool, error) {
	c, err := CompareVersions(have, min)
	if err != nil {
		return false, err
	}
	return c >= 0, nil
}
