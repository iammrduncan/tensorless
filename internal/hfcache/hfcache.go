// Package hfcache downloads a pinned model file into the Hugging Face cache
// layout ({cache}/models--{org}--{name}/blobs and snapshots), resumably, and
// verifies it by sha256.
package hfcache

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

// DefaultEndpoint is used when HF_ENDPOINT is not set.
const DefaultEndpoint = "https://huggingface.co"

// File is one file in a Hugging Face model repository, pinned by revision and
// sha256.
type File struct {
	Repo     string // org/name
	Revision string // the repository commit
	Path     string // the file's path inside the repository
	SHA256   string
	Size     int64 // 0 if unknown
}

// ErrNotCached means the file is not in the cache.
var ErrNotCached = errors.New("the model is not in the cache")

// DefaultCacheDir is $HF_HUB_CACHE, else $HF_HOME/hub, else
// ~/.cache/huggingface/hub.
func DefaultCacheDir(getenv func(string) string, home string) string {
	if v := getenv("HF_HUB_CACHE"); v != "" {
		return v
	}
	if v := getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// DefaultStateDir is $XDG_STATE_HOME/tensorless, else
// ~/.local/state/tensorless.
func DefaultStateDir(getenv func(string) string, home string) string {
	if v := getenv("XDG_STATE_HOME"); v != "" {
		return filepath.Join(v, "tensorless")
	}
	return filepath.Join(home, ".local", "state", "tensorless")
}

// RepoDir is {cache}/models--{org}--{name}.
func RepoDir(cache, repo string) string {
	return filepath.Join(cache, "models--"+strings.ReplaceAll(repo, "/", "--"))
}

// BlobPath is where the file's content lives: {repo dir}/blobs/{sha256}.
func BlobPath(cache string, f File) string {
	return filepath.Join(RepoDir(cache, f.Repo), "blobs", f.SHA256)
}

// SnapshotPath is the symlink {repo dir}/snapshots/{revision}/{file}.
func SnapshotPath(cache string, f File) string {
	return filepath.Join(RepoDir(cache, f.Repo), "snapshots", f.Revision, filepath.FromSlash(f.Path))
}

// LinkTarget is the relative target of the snapshot symlink:
// ../../blobs/{sha256}, with one more ../ per subdirectory in the file path.
func LinkTarget(f File) string {
	depth := strings.Count(f.Path, "/") + 2
	return strings.Repeat("../", depth) + "blobs/" + f.SHA256
}

// URL is {endpoint}/{repo}/resolve/{revision}/{file}.
func URL(endpoint string, f File) string {
	endpoint = strings.TrimRight(endpoint, "/")
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	return endpoint + "/" + escapePath(f.Repo) + "/resolve/" + url.PathEscape(f.Revision) + "/" + escapePath(f.Path)
}

func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}

// Status reports whether the file is in the cache: "yes" when the blob exists
// with the expected size, "partial" when a download was interrupted, else
// "no". It does not hash anything.
func Status(cache string, f File) string {
	blob := BlobPath(cache, f)
	if fi, err := os.Stat(blob); err == nil && fi.Mode().IsRegular() && (f.Size == 0 || fi.Size() == f.Size) {
		return "yes"
	}
	if _, err := os.Stat(blob + ".incomplete"); err == nil {
		return "partial"
	}
	return "no"
}

// Link creates the snapshot symlink, replacing a wrong one.
func Link(cache string, f File) error {
	link := SnapshotPath(cache, f)
	target := LinkTarget(f)
	if cur, err := os.Readlink(link); err == nil && cur == target {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
		return err
	}
	if _, err := os.Lstat(link); err == nil {
		if err := os.Remove(link); err != nil {
			return fmt.Errorf("replacing %s: %w", link, err)
		}
	}
	if err := os.Symlink(target, link); err != nil {
		return fmt.Errorf("linking the snapshot: %w", err)
	}
	return nil
}

func checkFile(f File) error {
	switch {
	case !strings.Contains(f.Repo, "/"):
		return fmt.Errorf("bad repository %q", f.Repo)
	case f.Revision == "" || strings.ContainsAny(f.Revision, "/\\"):
		return fmt.Errorf("bad revision %q", f.Revision)
	case len(f.SHA256) != 64:
		return fmt.Errorf("bad sha256 %q", f.SHA256)
	case f.Path == "" || strings.HasPrefix(f.Path, "/") || strings.Contains(f.Path, "..") || strings.Contains(f.Path, `\`):
		return fmt.Errorf("bad file path %q", f.Path)
	}
	return nil
}
