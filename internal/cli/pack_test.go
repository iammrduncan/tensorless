package cli

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

const packRev = "89abcdef0123456789abcdef0123456789abcdef"

var packPrepare = []string{"python3", "-E", "-s", "/opt/tensorless/inkling-prepare/prepare.py"}

// packData is a three-file pack, in the order the recipe names it.
func packData() ([]string, map[string][]byte) {
	return []string{"config.json", "layer_00_packed.npz", ".gitattributes"}, map[string][]byte{
		"config.json":         []byte(`{"text_config": {}}`),
		"layer_00_packed.npz": bytes.Repeat([]byte("weights "), 500),
		".gitattributes":      []byte("*.npz filter=lfs\n"),
	}
}

// publishedPack writes a released recipe for Org/Pack on p100 whose model
// is a pack of the given files, and returns the recipes directory.
func (ta *testApp) publishedPack(t *testing.T, names []string, data map[string][]byte) string {
	t.Helper()
	var files []string
	for _, n := range names {
		sum := sha256.Sum256(data[n])
		files = append(files, fmt.Sprintf(`  { name = %q, sha256 = %q, size_bytes = %d },`, n, hex.EncodeToString(sum[:]), len(data[n])))
	}
	q := make([]string, len(packPrepare))
	for i, a := range packPrepare {
		q[i] = strconv.Quote(a)
	}
	rec := fmt.Sprintf(`schema = 1
status = "released"
[model]
name = "pk"
hf_repo = "Org/Pack"
revision = %q
license = "apache-2.0"
[model.pack]
prepare = [%s]
files = [
%s
]
[card]
id = "p100"
title = "NVIDIA Tesla P100 16 GB"
compute_capability = "6.0"
min_memory_mib = 16000
min_driver = "525.60.13"
[engine]
release = "engine-v0.4.0"
arch = "sm60"
image = "ghcr.io/x/engine@sha256:%s"
[run]
alias = "Org/Pack"
context = 4096
threads_max = 8
args = ["-ngl", "99"]
env = { GGML_X = "1" }
`, packRev, strings.Join(q, ", "), strings.Join(files, "\n"), strings.Repeat("a", 64))
	dir := filepath.Join(ta.dir, "recipes")
	os.MkdirAll(filepath.Join(dir, "Org", "Pack"), 0o755)
	if err := os.WriteFile(filepath.Join(dir, "Org", "Pack", "p100.toml"), []byte(rec), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// hfServer serves the pack's files at the revision and counts the requests.
func hfServer(t *testing.T, data map[string][]byte) (*httptest.Server, *int) {
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name, ok := strings.CutPrefix(r.URL.Path, "/Org/Pack/resolve/"+packRev+"/")
		if !ok || data[name] == nil {
			http.NotFound(w, r)
			return
		}
		n++
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(data[name]))
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

func (ta *testApp) repoDir() string {
	return filepath.Join(ta.env["HF_HUB_CACHE"], "models--Org--Pack")
}

func (ta *testApp) preparedDir() string {
	return filepath.Join(ta.env["XDG_STATE_HOME"], "tensorless", "prepared", "Org--Pack", "p100", packRev[:12]+"-"+strings.Repeat("a", 12))
}

// mountedOut is the host folder a prepare argv mounts at /models/prepared.
func mountedOut(args []string) string {
	for i, a := range args {
		if a == "-v" && strings.HasSuffix(args[i+1], ":/models/prepared") {
			return strings.TrimSuffix(args[i+1], ":/models/prepared")
		}
	}
	return ""
}

// fakePrepare writes inkling2.gguf into the mounted output folder and prints
// its path last, as prepare.py does. It fails if the folder is not empty.
func fakePrepare(args []string, stdout, stderr io.Writer) error {
	out := mountedOut(args)
	if entries, err := os.ReadDir(out); err != nil || len(entries) != 0 {
		return fmt.Errorf("the output folder %s is not there and empty: %v %v", out, entries, err)
	}
	if err := os.WriteFile(filepath.Join(out, "inkling2.gguf"), []byte("GGUF"), 0o644); err != nil {
		return err
	}
	io.WriteString(stderr, "tensors 3 matrices 1 vocab 8 merges 2\n")
	io.WriteString(stdout, "/models/prepared/inkling2.gguf\n")
	return nil
}

func TestPackPullAndServe(t *testing.T) {
	names, data := packData()
	srv, served := hfServer(t, data)
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	ta.host.prepare = fakePrepare
	ta.env["HF_ENDPOINT"] = srv.URL
	ta.HTTP = srv.Client()
	var spaceAsked []string
	ta.FreeSpace = func(dir string) (uint64, error) { spaceAsked = append(spaceAsked, dir); return 1 << 40, nil }
	dir := ta.publishedPack(t, names, data)

	if code := ta.run("pull", "org/pack", "--recipes", dir); code != 0 {
		t.Fatalf("pull: exit %d: %s", code, ta.stderr)
	}
	if *served != 3 {
		t.Errorf("%d downloads, want 3", *served)
	}
	snap := filepath.Join(ta.repoDir(), "snapshots", packRev)
	for _, n := range names {
		if got, err := os.ReadFile(filepath.Join(snap, n)); err != nil || !bytes.Equal(got, data[n]) {
			t.Errorf("%s through the snapshot: %v", n, err)
		}
	}
	if len(spaceAsked) == 0 || spaceAsked[0] != filepath.Join(ta.repoDir(), "blobs") {
		t.Errorf("disk space asked for %v", spaceAsked)
	}
	for _, want := range []string{"model pack Org/Pack at " + packRev[:12] + ": 3 files, 4.0 kB", "file 2 of 3: layer_00_packed.npz"} {
		if !strings.Contains(ta.stderr.String(), want) {
			t.Errorf("pull stderr lacks %q:\n%s", want, ta.stderr)
		}
	}
	if !strings.Contains(ta.stdout.String(), "model: 3 files (4.0 kB) in "+snap) {
		t.Errorf("pull stdout = %s", ta.stdout)
	}

	// A file left by an earlier prepare step is removed first.
	os.MkdirAll(ta.preparedDir(), 0o755)
	os.WriteFile(filepath.Join(ta.preparedDir(), "stale.gguf"), []byte("old"), 0o644)

	h, port := healthServer(t, http.StatusOK)
	ta.stderr.b.Reset()
	if code := ta.run("serve", "pk", "--recipes", dir, "--detach", "--host", h, "--port", strconv.Itoa(port)); code != 0 {
		t.Fatalf("serve: exit %d: %s", code, ta.stderr)
	}
	runs := ta.host.calls(ta.fake, "podman run")
	if len(runs) != 2 {
		t.Fatalf("podman run calls: %v", ta.fake.Calls())
	}
	image := "ghcr.io/x/engine@sha256:" + strings.Repeat("a", 64)
	wantPrepare := "podman run --rm --network none --security-opt label=disable " +
		"-v " + ta.repoDir() + ":/models/repo:ro -v " + ta.preparedDir() + ":/models/prepared " +
		"--entrypoint python3 " + image + " -E -s /opt/tensorless/inkling-prepare/prepare.py " +
		"/models/repo/snapshots/" + packRev + " /models/prepared"
	if runs[0] != wantPrepare {
		t.Errorf("prepare =\n%s\nwant\n%s", runs[0], wantPrepare)
	}
	wantServe := "podman run -d --rm --name tensorless-org-pack --device nvidia.com/gpu=all --security-opt label=disable " +
		"-p 127.0.0.1:" + strconv.Itoa(port) + ":8080 -v " + ta.repoDir() + ":/models/repo:ro -v " + ta.preparedDir() + ":/models/prepared:ro " +
		"-e OMP_NUM_THREADS=6 -e GGML_X=1 " + image +
		" -m /models/prepared/inkling2.gguf --host 0.0.0.0 --port 8080 -c 4096 --alias Org/Pack -t 6 -tb 6 -ngl 99"
	if runs[1] != wantServe {
		t.Errorf("serve =\n%s\nwant\n%s", runs[1], wantServe)
	}
	if _, err := os.Stat(filepath.Join(ta.preparedDir(), "stale.gguf")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("the stale file is still there: %v", err)
	}
	for _, want := range []string{"preparing the pack in " + ta.preparedDir() + ": python3 -E -s /opt/tensorless/inkling-prepare/prepare.py", "tensors 3 matrices 1"} {
		if !strings.Contains(ta.stderr.String(), want) {
			t.Errorf("serve stderr lacks %q:\n%s", want, ta.stderr)
		}
	}
	if !strings.Contains(ta.stdout.String(), "Org/Pack is ready at") {
		t.Errorf("stdout = %s", ta.stdout)
	}
}

func TestPackServeNeedsEveryFile(t *testing.T) {
	names, data := packData()
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	ta.host.imagePresent = true
	ta.host.prepare = fakePrepare
	dir := ta.publishedPack(t, names, data)
	// Two of the three files are cached.
	for _, n := range names[:2] {
		sum := sha256.Sum256(data[n])
		blob := filepath.Join(ta.repoDir(), "blobs", hex.EncodeToString(sum[:]))
		os.MkdirAll(filepath.Dir(blob), 0o755)
		os.WriteFile(blob, data[n], 0o644)
	}
	if code := ta.run("serve", "pk", "--recipes", dir, "--detach"); code != 1 ||
		!strings.Contains(ta.stderr.String(), "the model pack is not complete or not downloaded. Run: tensorless pull Org/Pack") {
		t.Errorf("exit %d: %s", code, ta.stderr)
	}
	if runs := ta.host.calls(ta.fake, "podman run"); len(runs) != 0 {
		t.Errorf("ran %v", runs)
	}

	ta.stdout.b.Reset()
	if code := ta.run("models", "--recipes", dir, "--json"); code != 0 {
		t.Fatalf("models: exit %d", code)
	}
	var rows []modelRow
	if err := json.Unmarshal([]byte(ta.stdout.String()), &rows); err != nil || len(rows) != 1 || rows[0].Cached != "partial" {
		t.Errorf("models = %+v, %v", rows, err)
	}
}

func TestPackPrepareFailures(t *testing.T) {
	names, data := packData()
	tests := []struct {
		name    string
		prepare func(args []string, stdout, stderr io.Writer) error
		want    string
	}{
		{"exits", func(_ []string, _, stderr io.Writer) error {
			io.WriteString(stderr, "Traceback: no module named numpy\n")
			return errors.New("exit status 1")
		}, "the pack's prepare step failed: exit status 1. Its output is above"},
		{"prints nothing", func([]string, io.Writer, io.Writer) error { return nil }, `printed "" last, not a file in /models/prepared`},
		{"outside the folder", func(_ []string, stdout, _ io.Writer) error {
			io.WriteString(stdout, "/models/prepared/../repo/x.gguf\n")
			return nil
		}, `printed "/models/prepared/../repo/x.gguf" last, not a file in /models/prepared`},
		{"relative", func(_ []string, stdout, _ io.Writer) error {
			io.WriteString(stdout, "models/prepared/inkling2.gguf\n")
			return nil
		}, "not a file in /models/prepared"},
		{"no such file", func(_ []string, stdout, _ io.Writer) error {
			io.WriteString(stdout, "/models/prepared/inkling2.gguf\n")
			return nil
		}, "printed /models/prepared/inkling2.gguf, but there is no such file in "},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ta := newApp(t, "nvidia-smi", "podman")
			ta.host.gpus = p100GPU
			ta.host.imagePresent = true
			ta.host.prepare = tc.prepare
			dir := ta.publishedPack(t, names, data)
			for _, n := range names {
				sum := sha256.Sum256(data[n])
				blob := filepath.Join(ta.repoDir(), "blobs", hex.EncodeToString(sum[:]))
				os.MkdirAll(filepath.Dir(blob), 0o755)
				os.WriteFile(blob, data[n], 0o644)
			}
			if code := ta.run("serve", "pk", "--recipes", dir, "--detach"); code != 1 || !strings.Contains(ta.stderr.String(), tc.want) {
				t.Errorf("exit %d: %s", code, ta.stderr)
			}
			if runs := ta.host.calls(ta.fake, "podman run -d"); len(runs) != 0 {
				t.Errorf("the server was started: %v", runs)
			}
		})
	}
}

func TestPackDryRunAndInfo(t *testing.T) {
	names, data := packData()
	ta := newApp(t, "podman") // no GPU
	dir := ta.publishedPack(t, names, data)
	if code := ta.run("run", "pk", "--recipes", dir, "--dry-run"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	out := flatten(ta.stdout.String())
	wants := []string{
		"# 2. download the model pack, 3 files and 4.0 kB (0 of them cached, 4.0 kB to download), resuming partial downloads, and check each file's sha256",
		"link each file into " + filepath.Join(ta.repoDir(), "snapshots", packRev) + ", as <name> -> ../../blobs/<sha256>",
		"# 3. prepare the pack: empty " + ta.preparedDir() + ", then run the recipe's prepare step in the engine image, with no GPU and no network",
		"podman run --rm --network none --security-opt label=disable -v " + ta.repoDir() + ":/models/repo:ro",
		"# 4. start the server",
		"-v " + ta.repoDir() + ":/models/repo:ro -v " + ta.preparedDir() + ":/models/prepared:ro",
		"-m '<the file the prepare step prints>'",
	}
	for _, n := range names {
		wants = append(wants, "GET https://huggingface.co/Org/Pack/resolve/"+packRev+"/"+n+"\n")
	}
	for _, want := range wants {
		if !strings.Contains(out, want) {
			t.Errorf("plan lacks %q:\n%s", want, out)
		}
	}
	for _, c := range ta.fake.Calls() {
		if !strings.HasPrefix(c, "nvidia-smi") {
			t.Errorf("a dry run ran %q", c)
		}
	}
	if _, err := os.Stat(ta.preparedDir()); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a dry run made %s", ta.preparedDir())
	}

	// serve's plan verifies every file before the prepare step.
	ta.stdout.b.Reset()
	if code := ta.run("serve", "pk", "--recipes", dir, "--dry-run"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	if out := ta.stdout.String(); strings.Count(out, "\nverify ") != 3 || !strings.Contains(out, "the model pack's 3 files are cached and verified") {
		t.Errorf("serve plan:\n%s", out)
	}

	ta.stdout.b.Reset()
	if code := ta.run("info", "pk", "--recipes", dir, "--json"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	var info infoOut
	if err := json.Unmarshal([]byte(ta.stdout.String()), &info); err != nil {
		t.Fatal(err)
	}
	if info.Recipe.Model.Pack == nil || len(info.Download.Files) != 3 || info.Download.Size != 4036 || info.Download.Cached != "no" ||
		info.Download.URL != "https://huggingface.co/Org/Pack/resolve/"+packRev+"/" || len(info.Prepare) == 0 {
		t.Errorf("info = %+v", info.Download)
	}
	if strings.Contains(ta.stdout.String(), `"file"`) || strings.Contains(ta.stdout.String(), `"sha256": ""`) {
		t.Errorf("a pack's JSON has an empty file or sha256:\n%s", ta.stdout)
	}

	ta.stdout.b.Reset()
	if code := ta.run("info", "pk", "--recipes", dir); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	for _, want := range []string{"pack", "3 files, read in place after a prepare step", "4.0 kB (4036 bytes)", "layer_00_packed.npz", "prepare step (podman, before each serve; no GPU, no network):"} {
		if !strings.Contains(ta.stdout.String(), want) {
			t.Errorf("info lacks %q:\n%s", want, ta.stdout)
		}
	}
}

func TestPackDoctorCountsTheSum(t *testing.T) {
	names, data := packData()
	ta := newApp(t, "nvidia-smi", "podman")
	ta.host.gpus = p100GPU
	ta.files["/etc/cdi/nvidia.yaml"] = true
	dir := ta.publishedPack(t, names, data)
	ta.FreeSpace = func(string) (uint64, error) { return 4000, nil } // more than any one file, less than the sum
	if code := ta.run("doctor", "--recipes", dir); code != 1 || !strings.Contains(ta.stdout.String(), "FAIL  model cache") ||
		!strings.Contains(ta.stdout.String(), "the model needs 4.0 kB") {
		t.Errorf("exit %d:\n%s", code, ta.stdout)
	}
}
