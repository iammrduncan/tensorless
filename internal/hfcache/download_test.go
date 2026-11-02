package hfcache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	repo = "iamMrDuncan/Qwen3.8-27B-PN3-GGUF" // mixed case: kept as is in the cache
	rev  = "0123456789abcdef0123456789abcdef01234567"
)

func testData(n int) ([]byte, string) {
	b := make([]byte, n)
	rand.New(rand.NewSource(1)).Read(b)
	sum := sha256.Sum256(b)
	return b, hex.EncodeToString(sum[:])
}

type server struct {
	*httptest.Server
	mu       sync.Mutex
	requests []*http.Request
	served   int64
}

func (s *server) record(r *http.Request) {
	s.mu.Lock()
	s.requests = append(s.requests, r.Clone(context.Background()))
	s.mu.Unlock()
}

func (s *server) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

type countingWriter struct {
	http.ResponseWriter
	n *int64
}

func (c countingWriter) Write(b []byte) (int, error) {
	n, err := c.ResponseWriter.Write(b)
	*c.n += int64(n)
	return n, err
}

// newServer serves data at the resolve URL. handler, if set, replaces the
// default (http.ServeContent, which honours Range).
func newServer(t *testing.T, data []byte, handler func(s *server, w http.ResponseWriter, r *http.Request)) *server {
	s := &server{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.record(r)
		if handler != nil {
			handler(s, w, r)
			return
		}
		http.ServeContent(countingWriter{w, &s.served}, r, "", time.Time{}, bytes.NewReader(data))
	}))
	t.Cleanup(s.Close)
	return s
}

func newDownloader(t *testing.T, s *server) (*Downloader, string) {
	t.Helper()
	dir := t.TempDir()
	return &Downloader{
		Client:    s.Client(),
		Endpoint:  s.URL,
		StateDir:  filepath.Join(dir, "state"),
		Log:       io.Discard,
		FreeSpace: func(string) (uint64, error) { return 1 << 40, nil },
	}, filepath.Join(dir, "hub")
}

func checkCached(t *testing.T, cache string, f File, data []byte) {
	t.Helper()
	blob := BlobPath(cache, f)
	got, err := os.ReadFile(blob)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("blob content wrong (err %v, %d bytes)", err, len(got))
	}
	if _, err := os.Stat(blob + ".incomplete"); !os.IsNotExist(err) {
		t.Errorf("the .incomplete file is still there")
	}
	link := SnapshotPath(cache, f)
	target, err := os.Readlink(link)
	if err != nil || target != LinkTarget(f) {
		t.Errorf("snapshot link = %q, %v", target, err)
	}
	if got, err := os.ReadFile(link); err != nil || !bytes.Equal(got, data) {
		t.Errorf("the snapshot link does not resolve to the blob: %v", err)
	}
}

func TestPaths(t *testing.T) {
	f := File{Repo: repo, Revision: rev, Path: "Qwen3.8-27B-PN3-P0k19.gguf", SHA256: strings.Repeat("a", 64)}
	if got, want := RepoDir("/c", f.Repo), "/c/models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF"; got != want {
		t.Errorf("RepoDir = %q, want %q", got, want)
	}
	if got, want := URL("https://hf.example/", f), "https://hf.example/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/resolve/"+rev+"/Qwen3.8-27B-PN3-P0k19.gguf"; got != want {
		t.Errorf("URL = %q, want %q", got, want)
	}
	if got := URL("", File{Repo: "o/r", Revision: rev, Path: "a b/c.gguf"}); got != "https://huggingface.co/o/r/resolve/"+rev+"/a%20b/c.gguf" {
		t.Errorf("URL with default endpoint = %q", got)
	}
	if got := LinkTarget(f); got != "../../blobs/"+f.SHA256 {
		t.Errorf("LinkTarget = %q", got)
	}
	f.Path = "sub/dir/m.gguf"
	if got := LinkTarget(f); got != "../../../../blobs/"+f.SHA256 {
		t.Errorf("LinkTarget with subdirectories = %q", got)
	}
	env := map[string]string{}
	getenv := func(k string) string { return env[k] }
	if got := DefaultCacheDir(getenv, "/home/u"); got != "/home/u/.cache/huggingface/hub" {
		t.Errorf("default cache = %q", got)
	}
	env["HF_HOME"] = "/hf"
	if got := DefaultCacheDir(getenv, "/home/u"); got != "/hf/hub" {
		t.Errorf("HF_HOME cache = %q", got)
	}
	env["HF_HUB_CACHE"] = "/hub"
	if got := DefaultCacheDir(getenv, "/home/u"); got != "/hub" {
		t.Errorf("HF_HUB_CACHE cache = %q", got)
	}
	if got := DefaultStateDir(getenv, "/home/u"); got != "/home/u/.local/state/tensorless" {
		t.Errorf("state dir = %q", got)
	}
}

func TestFreshDownloadThenMarker(t *testing.T) {
	data, sum := testData(300_000)
	s := newServer(t, data, nil)
	d, cache := newDownloader(t, s)
	d.Token = "hf_secret"
	var log bytes.Buffer
	d.Log = &log
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: int64(len(data))}

	blob, err := d.Fetch(context.Background(), cache, f)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(blob, "models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF") {
		t.Errorf("blob path lost the repo's case: %s", blob)
	}
	checkCached(t, cache, f, data)
	r := s.requests[0]
	if r.URL.Path != "/"+repo+"/resolve/"+rev+"/m.gguf" || r.Header.Get("Range") != "" {
		t.Errorf("request %s, Range %q", r.URL.Path, r.Header.Get("Range"))
	}
	if r.Header.Get("Authorization") != "Bearer hf_secret" {
		t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
	}
	if strings.Contains(log.String(), "hf_secret") {
		t.Errorf("the token was printed")
	}
	if Status(cache, f) != "yes" {
		t.Errorf("Status = %q", Status(cache, f))
	}

	// The second fetch trusts the verified marker: no request, no hashing.
	log.Reset()
	if _, err := d.Fetch(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	if s.count() != 1 || strings.Contains(log.String(), "verifying") {
		t.Errorf("second fetch: %d requests, log %q", s.count(), log.String())
	}

	// A changed mtime invalidates the marker: the blob is hashed again.
	os.Chtimes(blob, time.Now(), time.Now().Add(-time.Hour))
	if _, err := d.Verify(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(log.String(), "verifying") {
		t.Errorf("expected a re-verification, log %q", log.String())
	}
}

func TestResume(t *testing.T) {
	data, sum := testData(300_000)
	s := newServer(t, data, nil)
	d, cache := newDownloader(t, s)
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: int64(len(data))}
	blob := BlobPath(cache, f)
	os.MkdirAll(filepath.Dir(blob), 0o755)
	os.WriteFile(blob+".incomplete", data[:100_000], 0o644)

	if _, err := d.Fetch(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	checkCached(t, cache, f, data)
	if got := s.requests[0].Header.Get("Range"); got != "bytes=100000-" {
		t.Errorf("Range = %q", got)
	}
	if s.served != 200_000 {
		t.Errorf("served %d bytes, want 200000", s.served)
	}
}

func TestInterruptedThenResumed(t *testing.T) {
	data, sum := testData(300_000)
	fail := true
	s := newServer(t, data, func(s *server, w http.ResponseWriter, r *http.Request) {
		if fail {
			w.Header().Set("Content-Length", "300000")
			w.WriteHeader(http.StatusOK)
			w.Write(data[:120_000])
			w.(http.Flusher).Flush()
			panic(http.ErrAbortHandler) // drop the connection mid-body
		}
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(data))
	})
	d, cache := newDownloader(t, s)
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: int64(len(data))}
	_, err := d.Fetch(context.Background(), cache, f)
	if err == nil || !strings.Contains(err.Error(), "resume") {
		t.Fatalf("expected an interruption, got %v", err)
	}
	if Status(cache, f) != "partial" {
		t.Errorf("Status = %q, want partial", Status(cache, f))
	}
	fail = false
	if _, err := d.Fetch(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	checkCached(t, cache, f, data)
	if got := s.requests[1].Header.Get("Range"); got != "bytes=120000-" {
		t.Errorf("Range = %q", got)
	}
}

func TestSHAMismatch(t *testing.T) {
	data, _ := testData(50_000)
	_, otherSum := testData(10)
	s := newServer(t, data, nil)
	d, cache := newDownloader(t, s)
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: otherSum, Size: int64(len(data))}
	_, err := d.Fetch(context.Background(), cache, f)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("expected a sha256 mismatch, got %v", err)
	}
	blob := BlobPath(cache, f)
	for _, p := range []string{blob, blob + ".incomplete", SnapshotPath(cache, f)} {
		if _, err := os.Lstat(p); !os.IsNotExist(err) {
			t.Errorf("%s should not exist", p)
		}
	}
}

func TestRangeAnsweredWith200(t *testing.T) {
	data, sum := testData(300_000)
	s := newServer(t, data, func(s *server, w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "300000")
		w.WriteHeader(http.StatusOK) // ignores Range
		w.Write(data)
	})
	d, cache := newDownloader(t, s)
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: int64(len(data))}
	blob := BlobPath(cache, f)
	os.MkdirAll(filepath.Dir(blob), 0o755)
	os.WriteFile(blob+".incomplete", bytes.Repeat([]byte("x"), 100_000), 0o644) // stale bytes

	if _, err := d.Fetch(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	checkCached(t, cache, f, data)
	if s.requests[0].Header.Get("Range") != "bytes=100000-" {
		t.Errorf("expected a Range request first")
	}
}

func TestRedirect(t *testing.T) {
	data, sum := testData(100_000)
	s := newServer(t, data, func(s *server, w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/resolve/") {
			http.Redirect(w, r, "/cdn/blob?sig=abc", http.StatusFound)
			return
		}
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(data))
	})
	d, cache := newDownloader(t, s)
	d.Token = "hf_secret"
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: int64(len(data))}
	if _, err := d.Fetch(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	checkCached(t, cache, f, data)
	if s.count() != 2 || s.requests[1].URL.Path != "/cdn/blob" {
		t.Errorf("requests: %d", s.count())
	}
}

func TestSnapshotLinkInSubdirectory(t *testing.T) {
	data, sum := testData(1000)
	s := newServer(t, data, nil)
	d, cache := newDownloader(t, s)
	f := File{Repo: repo, Revision: rev, Path: "sub/dir/m.gguf", SHA256: sum, Size: int64(len(data))}
	if _, err := d.Fetch(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	checkCached(t, cache, f, data)
	if got, _ := os.Readlink(SnapshotPath(cache, f)); got != "../../../../blobs/"+sum {
		t.Errorf("link target = %q", got)
	}
	if !strings.HasSuffix(SnapshotPath(cache, f), filepath.Join("snapshots", rev, "sub", "dir", "m.gguf")) {
		t.Errorf("snapshot path = %q", SnapshotPath(cache, f))
	}

	// A wrong link is replaced.
	link := SnapshotPath(cache, f)
	os.Remove(link)
	os.Symlink("../elsewhere", link)
	if err := Link(cache, f); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.Readlink(link); got != LinkTarget(f) {
		t.Errorf("link not repaired: %q", got)
	}
}

func TestCorruptCachedBlobIsReplaced(t *testing.T) {
	data, sum := testData(10_000)
	s := newServer(t, data, nil)
	d, cache := newDownloader(t, s)
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: int64(len(data))}
	blob := BlobPath(cache, f)
	os.MkdirAll(filepath.Dir(blob), 0o755)
	os.WriteFile(blob, bytes.Repeat([]byte("y"), len(data)), 0o644)

	if _, err := d.Verify(context.Background(), cache, f); err != ErrNotCached {
		t.Fatalf("Verify of a corrupt blob = %v, want ErrNotCached", err)
	}
	if _, err := d.Fetch(context.Background(), cache, f); err != nil {
		t.Fatal(err)
	}
	checkCached(t, cache, f, data)
}

func TestDiskSpaceAndSizeChecks(t *testing.T) {
	data, sum := testData(10_000)
	s := newServer(t, data, nil)
	d, cache := newDownloader(t, s)
	d.FreeSpace = func(string) (uint64, error) { return 5000, nil }
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: int64(len(data))}
	if _, err := d.Fetch(context.Background(), cache, f); err == nil || !strings.Contains(err.Error(), "not enough disk space") {
		t.Errorf("disk check: %v", err)
	}
	if s.count() != 0 {
		t.Errorf("no request should be made without disk space")
	}

	d.FreeSpace = nil
	f.Size = 20_000
	if _, err := d.Fetch(context.Background(), cache, f); err == nil || !strings.Contains(err.Error(), "recipe expects 20000") {
		t.Errorf("size check: %v", err)
	}
}

func TestHTTPErrors(t *testing.T) {
	s := newServer(t, nil, func(s *server, w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	})
	d, cache := newDownloader(t, s)
	_, sum := testData(1)
	f := File{Repo: repo, Revision: rev, Path: "m.gguf", SHA256: sum, Size: 1}
	if _, err := d.Fetch(context.Background(), cache, f); err == nil || !strings.Contains(err.Error(), "HF_TOKEN") {
		t.Errorf("401: %v", err)
	}
}
