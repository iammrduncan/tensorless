package container

import (
	"context"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/iammrduncan/tensorless/internal/execx"
)

func spec(rt string) Spec {
	return Spec{
		Runtime:   rt,
		Name:      "tensorless-iammrduncan-qwen3.8-27b-pn3-gguf",
		Image:     "ghcr.io/x/engine@sha256:" + strings.Repeat("a", 64),
		Host:      "127.0.0.1",
		Port:      8080,
		ModelPath: "/hub/models--o--r/blobs/" + strings.Repeat("b", 64),
		ModelFile: "m.gguf",
		Threads:   6,
		Env:       map[string]string{"GGML_B": "2", "GGML_A": "1"},
		Context:   65536,
		Alias:     "iamMrDuncan/Qwen3.8-27B-PN3-GGUF",
		Args:      []string{"-ngl", "99", "-fa", "on"},
		GPU:       AllGPUs,
	}
}

func TestRunArgs(t *testing.T) {
	image := "ghcr.io/x/engine@sha256:" + strings.Repeat("a", 64)
	mount := "/hub/models--o--r/blobs/" + strings.Repeat("b", 64) + ":/models/m.gguf:ro"
	tail := func(ctx string, extra ...string) []string {
		a := []string{"-p", "127.0.0.1:8080:8080", "-v", mount, "-e", "OMP_NUM_THREADS=6", "-e", "GGML_A=1", "-e", "GGML_B=2",
			image, "-m", "/models/m.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", ctx, "--alias", "iamMrDuncan/Qwen3.8-27B-PN3-GGUF",
			"-t", "6", "-tb", "6", "-ngl", "99", "-fa", "on"}
		return append(a, extra...)
	}
	head := []string{"run", "-d", "--rm", "--name", "tensorless-iammrduncan-qwen3.8-27b-pn3-gguf"}
	join := func(parts ...[]string) []string {
		var out []string
		for _, p := range parts {
			out = append(out, p...)
		}
		return out
	}

	tests := []struct {
		name string
		edit func(*Spec)
		want []string
	}{
		{"podman all", func(s *Spec) {}, join(head, []string{"--device", "nvidia.com/gpu=all", "--security-opt", "label=disable"}, tail("65536"))},
		{"podman gpu 1", func(s *Spec) { s.GPU = 1 }, join(head, []string{"--device", "nvidia.com/gpu=1", "--security-opt", "label=disable"}, tail("65536"))},
		{"docker all", func(s *Spec) { s.Runtime = "docker" }, join(head, []string{"--gpus", "all"}, tail("65536"))},
		{"docker gpu 0", func(s *Spec) { s.Runtime = "docker"; s.GPU = 0 }, join(head, []string{"--gpus", "device=0"}, tail("65536"))},
		{"podman context and extra", func(s *Spec) { s.Context = 8192; s.Extra = []string{"--temp", "0.7"} },
			join(head, []string{"--device", "nvidia.com/gpu=all", "--security-opt", "label=disable"}, tail("8192", "--temp", "0.7"))},
		{"docker gpu, context and extra", func(s *Spec) { s.Runtime = "docker"; s.GPU = 2; s.Context = 4096; s.Extra = []string{"-v"} },
			join(head, []string{"--gpus", "device=2"}, tail("4096", "-v"))},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := spec("podman")
			tc.edit(&s)
			if got := s.RunArgs(); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got\n%q\nwant\n%q", got, tc.want)
			}
		})
	}
}

func TestName(t *testing.T) {
	tests := map[string]string{
		"iamMrDuncan/Qwen3.8-27B-PN3-GGUF": "tensorless-iammrduncan-qwen3.8-27b-pn3-gguf",
		"Org/Some Model+v2":                "tensorless-org-some-model-v2",
		"a_b/c.d-e":                        "tensorless-a_b-c.d-e",
	}
	for in, want := range tests {
		if got := Name(in); got != want {
			t.Errorf("Name(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDockerHasNoSecurityOpt(t *testing.T) {
	s := spec("docker")
	for _, a := range s.RunArgs() {
		if a == "--security-opt" || a == "--device" {
			t.Errorf("docker argv has %s", a)
		}
	}
}

func TestAddresses(t *testing.T) {
	if got := PublishArg("::1", 9000); got != "[::1]:9000:8080" {
		t.Errorf("PublishArg = %q", got)
	}
	if got := BaseURL("0.0.0.0", 8080); got != "http://127.0.0.1:8080" {
		t.Errorf("BaseURL = %q", got)
	}
	if got := BaseURL("::", 8080); got != "http://[::1]:8080" {
		t.Errorf("BaseURL = %q", got)
	}
}

func TestSelect(t *testing.T) {
	both := execx.NewFake("podman", "docker")
	if rt, err := Select(both, "auto"); rt != "podman" || err != nil {
		t.Errorf("auto with both = %q, %v", rt, err)
	}
	if rt, err := Select(both, "docker"); rt != "docker" || err != nil {
		t.Errorf("docker = %q, %v", rt, err)
	}
	if rt, err := Select(execx.NewFake("docker"), "auto"); rt != "docker" || err != nil {
		t.Errorf("auto with docker = %q, %v", rt, err)
	}
	if _, err := Select(execx.NewFake(), "auto"); err == nil {
		t.Errorf("auto with neither should fail")
	}
	if _, err := Select(execx.NewFake("docker"), "podman"); err == nil {
		t.Errorf("missing podman should fail")
	}
	if _, err := Select(both, "lxc"); err == nil {
		t.Errorf("unknown runtime should fail")
	}
}

func TestState(t *testing.T) {
	f := execx.NewFake("podman")
	f.Func = func(_ context.Context, _ string, args []string, stdout, _ io.Writer) error {
		if args[len(args)-1] == "tensorless-up" {
			io.WriteString(stdout, "true\n")
			return nil
		}
		return io.ErrUnexpectedEOF
	}
	if got := State(context.Background(), f, "podman", "tensorless-up"); got != Running {
		t.Errorf("State = %q", got)
	}
	if got := State(context.Background(), f, "podman", "tensorless-gone"); got != Absent {
		t.Errorf("State = %q", got)
	}
}
