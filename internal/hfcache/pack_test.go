package hfcache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// packServer serves several files of one revision by name.
func packServer(t *testing.T, files map[string][]byte) (*httptest.Server, *[]string) {
	var mu sync.Mutex
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		prefix := "/" + repo + "/resolve/" + rev + "/"
		name, ok := strings.CutPrefix(r.URL.Path, prefix)
		data, found := files[name]
		if !ok || !found {
			http.NotFound(w, r)
			return
		}
		mu.Lock()
		got = append(got, name)
		mu.Unlock()
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(data))
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

func packFiles(files map[string][]byte, names ...string) []File {
	var out []File
	for _, n := range names {
		sum := sha256.Sum256(files[n])
		out = append(out, File{Repo: repo, Revision: rev, Path: n, SHA256: hex.EncodeToString(sum[:]), Size: int64(len(files[n]))})
	}
	return out
}

func TestFetchAllPack(t *testing.T) {
	big, _ := testData(5000)
	data := map[string][]byte{"config.json": []byte(`{"a": 1}`), "layer_00.npz": big, ".gitattributes": []byte("*.npz lfs\n")}
	srv, served := packServer(t, data)
	dir := t.TempDir()
	var free []string
	d := &Downloader{Client: srv.Client(), Endpoint: srv.URL, StateDir: filepath.Join(dir, "state"),
		FreeSpace: func(p string) (uint64, error) { free = append(free, p); return 1 << 40, nil }}
	cache := filepath.Join(dir, "hub")
	files := packFiles(data, "config.json", "layer_00.npz", ".gitattributes")

	if got := Missing(cache, files); got != int64(5000+8+10) {
		t.Errorf("Missing = %d", got)
	}
	// Half of one file is already there: only the rest is missing.
	part := BlobPath(cache, files[1]) + ".incomplete"
	os.MkdirAll(filepath.Dir(part), 0o755)
	os.WriteFile(part, big[:2000], 0o644)
	if got := Missing(cache, files); got != int64(3000+8+10) {
		t.Errorf("Missing with a partial file = %d", got)
	}

	if _, err := d.VerifyAll(context.Background(), cache, files); !errors.Is(err, ErrNotCached) {
		t.Errorf("VerifyAll before the download = %v", err)
	}
	blobs, err := d.FetchAll(context.Background(), cache, files)
	if err != nil {
		t.Fatal(err)
	}
	if len(blobs) != 3 || len(*served) != 3 {
		t.Errorf("blobs %v, served %v", blobs, *served)
	}
	// One check for the whole pack, before the first download.
	if len(free) == 0 || free[0] != filepath.Join(RepoDir(cache, repo), "blobs") {
		t.Errorf("free space checked in %v", free)
	}
	snap := SnapshotDir(cache, repo, rev)
	for _, f := range files {
		got, err := os.ReadFile(filepath.Join(snap, f.Path))
		if err != nil || !bytes.Equal(got, data[path.Base(f.Path)]) {
			t.Errorf("%s through the snapshot: %v", f.Path, err)
		}
	}
	if Missing(cache, files) != 0 {
		t.Errorf("Missing after the download = %d", Missing(cache, files))
	}
	if _, err := d.VerifyAll(context.Background(), cache, files); err != nil {
		t.Errorf("VerifyAll = %v", err)
	}
}

func TestFetchAllChecksTheSum(t *testing.T) {
	data := map[string][]byte{"a.bin": bytes.Repeat([]byte("a"), 600), "b.bin": bytes.Repeat([]byte("b"), 600)}
	srv, served := packServer(t, data)
	dir := t.TempDir()
	// Room for either file, not for both.
	d := &Downloader{Client: srv.Client(), Endpoint: srv.URL, FreeSpace: func(string) (uint64, error) { return 1000, nil }}
	_, err := d.FetchAll(context.Background(), filepath.Join(dir, "hub"), packFiles(data, "a.bin", "b.bin"))
	if err == nil || !strings.Contains(err.Error(), "the download needs 1.2 kB") {
		t.Errorf("err = %v", err)
	}
	if len(*served) != 0 {
		t.Errorf("downloaded %v before refusing", *served)
	}
}

func TestFetchAllRefusesABadPath(t *testing.T) {
	d := &Downloader{}
	_, err := d.FetchAll(context.Background(), t.TempDir(), []File{{Repo: repo, Revision: rev, Path: "../x", SHA256: strings.Repeat("1", 64)}})
	if err == nil || !strings.Contains(err.Error(), "bad file path") {
		t.Errorf("err = %v", err)
	}
}
