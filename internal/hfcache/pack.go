package hfcache

import (
	"context"
	"fmt"
	"os"
	"path"
	"path/filepath"
)

// SnapshotDir is {repo dir}/snapshots/{revision}: the folder in which every
// file of the revision is linked under its own name.
func SnapshotDir(cache, repo, revision string) string {
	return filepath.Join(RepoDir(cache, repo), "snapshots", revision)
}

// Missing is how many bytes of files are still to download: the size of each
// file whose blob is not in place, less what a partial download already has.
func Missing(cache string, files []File) int64 {
	var need int64
	for _, f := range files {
		if Status(cache, f) == "yes" {
			continue
		}
		need += f.Size
		if fi, err := os.Stat(BlobPath(cache, f) + ".incomplete"); err == nil && fi.Size() <= f.Size {
			need -= fi.Size()
		}
	}
	return need
}

// FetchAll fetches every file, as Fetch does one, after checking that the
// disk has room for all that is missing. It returns the blob paths.
func (d *Downloader) FetchAll(ctx context.Context, cache string, files []File) ([]string, error) {
	for _, f := range files {
		if err := checkFile(f); err != nil {
			return nil, err
		}
	}
	if len(files) > 0 {
		dir := filepath.Dir(BlobPath(cache, files[0]))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
		if err := d.checkSpace(dir, Missing(cache, files)); err != nil {
			return nil, err
		}
	}
	blobs := make([]string, len(files))
	for i, f := range files {
		if len(files) > 1 {
			fmt.Fprintf(d.log(), "file %d of %d: %s (%s)\n", i+1, len(files), path.Base(f.Path), FormatBytes(f.Size))
		}
		blob, err := d.Fetch(ctx, cache, f)
		if err != nil {
			return nil, err
		}
		blobs[i] = blob
	}
	return blobs, nil
}

// VerifyAll verifies every file, as Verify does one. It returns ErrNotCached
// when any of them is missing or fails verification.
func (d *Downloader) VerifyAll(ctx context.Context, cache string, files []File) ([]string, error) {
	blobs := make([]string, len(files))
	for i, f := range files {
		blob, err := d.Verify(ctx, cache, f)
		if err != nil {
			return nil, err
		}
		blobs[i] = blob
	}
	return blobs, nil
}
