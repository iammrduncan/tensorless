// Package container builds and runs the podman or docker commands that serve
// a recipe.
package container

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"

	"github.com/iammrduncan/tensorless/internal/execx"
)

// Runtimes are the supported container runtimes, in the order auto tries them.
var Runtimes = []string{"podman", "docker"}

// AllGPUs gives the container every GPU.
const AllGPUs = -1

// ServerPort is the port llama-server listens on inside the container.
const ServerPort = 8080

// Select picks the runtime: "auto" (or "") prefers podman, then docker.
func Select(r execx.Runner, want string) (string, error) {
	switch want {
	case "", "auto":
		for _, name := range Runtimes {
			if _, err := r.LookPath(name); err == nil {
				return name, nil
			}
		}
		return "", errors.New("neither podman nor docker is installed. Install podman (or docker) and the NVIDIA container toolkit, then run: tensorless doctor")
	case "podman", "docker":
		if _, err := r.LookPath(want); err != nil {
			return "", fmt.Errorf("%s is not installed. Install it, or pass --runtime auto to use whichever is installed", want)
		}
		return want, nil
	}
	return "", fmt.Errorf("unknown runtime %q: use auto, podman or docker", want)
}

// Name is the container name for a model: "tensorless-" and the lowercased
// repo id, with every character outside [a-z0-9_.-] replaced by "-", so that
// podman and docker accept it.
func Name(repo string) string {
	b := []byte(strings.ToLower(repo))
	for i, c := range b {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '.' || c == '-') {
			b[i] = '-'
		}
	}
	return "tensorless-" + string(b)
}

// Spec is everything needed to start the server container.
type Spec struct {
	Runtime   string // podman or docker
	Name      string
	Image     string
	Host      string // host address the port is published on
	Port      int    // host port
	ModelPath string // the model file on the host (the blob)
	ModelFile string // its base name inside the container, under /models
	Threads   int
	Env       map[string]string
	Context   int
	Alias     string
	Args      []string // the recipe's llama-server arguments
	Extra     []string // extra arguments from the command line
	GPU       int      // AllGPUs, or one GPU index
}

// ContainerModelPath is where the model is mounted inside the container.
func (s Spec) ContainerModelPath() string { return "/models/" + s.ModelFile }

// RunArgs is the argv after the runtime name: run -d --rm ... IMAGE <server args>.
// The image's entrypoint is llama-server.
func (s Spec) RunArgs() []string {
	a := []string{"run", "-d", "--rm", "--name", s.Name}
	gpu := "all"
	if s.GPU != AllGPUs {
		gpu = strconv.Itoa(s.GPU)
	}
	if s.Runtime == "docker" {
		if s.GPU == AllGPUs {
			a = append(a, "--gpus", "all")
		} else {
			a = append(a, "--gpus", "device="+gpu)
		}
	} else {
		a = append(a, "--device", "nvidia.com/gpu="+gpu, "--security-opt", "label=disable")
	}
	a = append(a,
		"-p", PublishArg(s.Host, s.Port),
		"-v", s.ModelPath+":"+s.ContainerModelPath()+":ro",
		"-e", "OMP_NUM_THREADS="+strconv.Itoa(s.Threads),
	)
	keys := make([]string, 0, len(s.Env))
	for k := range s.Env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		a = append(a, "-e", k+"="+s.Env[k])
	}
	a = append(a, s.Image,
		"-m", s.ContainerModelPath(),
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(ServerPort),
		"-c", strconv.Itoa(s.Context),
		"--alias", s.Alias,
		"-t", strconv.Itoa(s.Threads),
		"-tb", strconv.Itoa(s.Threads),
	)
	a = append(a, s.Args...)
	return append(a, s.Extra...)
}

// PublishArg is the -p value HOST:PORT:8080, with brackets for IPv6.
func PublishArg(host string, port int) string {
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	return fmt.Sprintf("%s:%d:%d", host, port, ServerPort)
}

// ClientHost is the address a local client uses to reach a server bound to
// host: a wildcard address becomes loopback.
func ClientHost(host string) string {
	switch host {
	case "", "0.0.0.0":
		return "127.0.0.1"
	case "::", "[::]":
		return "::1"
	}
	return host
}

// BaseURL is http://HOST:PORT for a local client.
func BaseURL(host string, port int) string {
	return "http://" + net.JoinHostPort(ClientHost(host), strconv.Itoa(port))
}

// Container states.
const (
	Absent  = "absent"
	Running = "running"
	Stopped = "stopped"
)

// State reports whether the named container is running, stopped, or absent.
func State(ctx context.Context, r execx.Runner, rt, name string) string {
	out, err := r.Output(ctx, rt, "container", "inspect", "--format", "{{.State.Running}}", name)
	if err != nil {
		return Absent
	}
	if strings.TrimSpace(string(out)) == "true" {
		return Running
	}
	return Stopped
}

// ImagePresent reports whether the image is already in local storage.
func ImagePresent(ctx context.Context, r execx.Runner, rt, image string) bool {
	_, err := r.Output(ctx, rt, "image", "inspect", image)
	return err == nil
}

// StopArgs is the argv after the runtime name that stops the container.
func StopArgs(name string) []string { return []string{"stop", name} }

// LogsArgs is the argv after the runtime name that follows the logs.
func LogsArgs(name string) []string { return []string{"logs", "-f", name} }
