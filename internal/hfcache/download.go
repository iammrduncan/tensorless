package hfcache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Downloader fetches files into the cache.
type Downloader struct {
	Client    *http.Client // nil: http.DefaultClient
	Endpoint  string       // "": DefaultEndpoint
	Token     string       // sent as "Authorization: Bearer"; never printed
	UserAgent string
	// StateDir holds the verified markers. "" disables them.
	StateDir string
	// Log receives progress and notes.
	Log io.Writer
	// TTY makes progress rewrite one line instead of printing new ones.
	TTY bool
	// FreeSpace overrides the disk space check, for tests.
	FreeSpace func(dir string) (uint64, error)
	// StallTimeout aborts a download that receives nothing for this long.
	// 0 means two minutes.
	StallTimeout time.Duration
}

var errStalled = errors.New("the download stalled")

// Fetch makes sure the file is in the cache, verified, and linked into its
// snapshot, downloading what is missing. It returns the blob path.
func (d *Downloader) Fetch(ctx context.Context, cache string, f File) (string, error) {
	if err := checkFile(f); err != nil {
		return "", err
	}
	blob := BlobPath(cache, f)
	ok, err := d.checkExisting(ctx, blob, f)
	if err != nil {
		return "", err
	}
	if !ok {
		if err := d.download(ctx, blob, f); err != nil {
			return "", err
		}
	}
	if err := Link(cache, f); err != nil {
		return "", err
	}
	return blob, nil
}

// Verify checks a file that is already in the cache, without downloading. It
// returns ErrNotCached when the file is missing or fails verification.
func (d *Downloader) Verify(ctx context.Context, cache string, f File) (string, error) {
	if err := checkFile(f); err != nil {
		return "", err
	}
	blob := BlobPath(cache, f)
	ok, err := d.checkExisting(ctx, blob, f)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", ErrNotCached
	}
	if err := Link(cache, f); err != nil {
		return "", err
	}
	return blob, nil
}

func (d *Downloader) log() io.Writer {
	if d.Log == nil {
		return io.Discard
	}
	return d.Log
}

// checkExisting reports whether a verified blob is in place. A blob of the
// wrong size or hash is deleted.
func (d *Downloader) checkExisting(ctx context.Context, blob string, f File) (bool, error) {
	fi, err := os.Stat(blob)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if !fi.Mode().IsRegular() {
		return false, fmt.Errorf("%s is not a regular file; remove it and try again", blob)
	}
	name := path.Base(f.Path)
	if f.Size > 0 && fi.Size() != f.Size {
		fmt.Fprintf(d.log(), "the cached %s has %d bytes, not %d; deleting it and downloading again\n", name, fi.Size(), f.Size)
		return false, os.Remove(blob)
	}
	if d.markerOK(blob, fi, f.SHA256) {
		return true, nil
	}
	fmt.Fprintf(d.log(), "verifying the cached %s (once; the result is remembered)\n", name)
	h := sha256.New()
	if err := d.hashFile(ctx, blob, h, "verifying "+name, fi.Size()); err != nil {
		return false, err
	}
	if sum := hex.EncodeToString(h.Sum(nil)); sum != f.SHA256 {
		fmt.Fprintf(d.log(), "the cached %s has sha256 %s, not %s; deleting it and downloading again\n", name, sum, f.SHA256)
		return false, os.Remove(blob)
	}
	d.writeMarker(blob, f.SHA256)
	return true, nil
}

func (d *Downloader) download(ctx context.Context, blob string, f File) error {
	dir := filepath.Dir(blob)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	name := path.Base(f.Path)
	part := blob + ".incomplete"
	h := sha256.New()
	var offset int64
	if fi, err := os.Stat(part); err == nil {
		switch {
		case f.Size > 0 && fi.Size() > f.Size:
			if err := os.Remove(part); err != nil {
				return err
			}
		case fi.Size() > 0:
			fmt.Fprintf(d.log(), "resuming %s: checking the %s already downloaded\n", name, FormatBytes(fi.Size()))
			if err := d.hashFile(ctx, part, h, "checking "+name, fi.Size()); err != nil {
				return err
			}
			offset = fi.Size()
		}
	}
	if err := d.checkSpace(dir, f.Size-offset); err != nil {
		return err
	}
	if f.Size == 0 || offset < f.Size {
		if err := d.fetch(ctx, part, f, &offset, h); err != nil {
			return err
		}
	}

	fi, err := os.Stat(part)
	if err != nil {
		return err
	}
	if f.Size > 0 && fi.Size() != f.Size {
		if fi.Size() < f.Size {
			return fmt.Errorf("the download of %s stopped at %s of %s; run the command again to resume", name, FormatBytes(fi.Size()), FormatBytes(f.Size))
		}
		os.Remove(part)
		return fmt.Errorf("the download of %s is %d bytes, more than the expected %d; deleted it. The recipe's size_bytes or the upload is wrong", name, fi.Size(), f.Size)
	}
	if sum := hex.EncodeToString(h.Sum(nil)); sum != f.SHA256 {
		os.Remove(part)
		return fmt.Errorf("sha256 mismatch for %s: got %s, want %s. The download was deleted. If this happens again, the upload or the recipe is wrong; please report it", name, sum, f.SHA256)
	}
	if err := os.Rename(part, blob); err != nil {
		return err
	}
	d.writeMarker(blob, f.SHA256)
	return nil
}

// fetch downloads from *offset to the end into part, adding to h. It resets
// *offset and h when it has to start over.
func (d *Downloader) fetch(ctx context.Context, part string, f File, offset *int64, h hash.Hash) error {
	u := URL(d.Endpoint, f)
	client := d.Client
	if client == nil {
		client = http.DefaultClient
	}
	restart := func(why string) {
		fmt.Fprintf(d.log(), "%s; starting the download over\n", why)
		*offset = 0
		h.Reset()
	}
	for attempt := 0; attempt < 3; attempt++ {
		reqCtx, cancel := context.WithCancelCause(ctx)
		req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, u, nil)
		if err != nil {
			cancel(nil)
			return err
		}
		if d.Token != "" {
			req.Header.Set("Authorization", "Bearer "+d.Token)
		}
		if d.UserAgent != "" {
			req.Header.Set("User-Agent", d.UserAgent)
		}
		if *offset > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", *offset))
		}
		resp, err := client.Do(req)
		if err != nil {
			cancel(nil)
			if ctx.Err() != nil {
				return fmt.Errorf("download interrupted; run the command again to resume")
			}
			return fmt.Errorf("downloading %s: %w", u, err)
		}
		switch {
		case *offset > 0 && resp.StatusCode == http.StatusPartialContent:
			if start, ok := contentRangeStart(resp.Header.Get("Content-Range")); !ok || start != *offset {
				resp.Body.Close()
				cancel(nil)
				restart("the server answered with a different range")
				continue
			}
		case resp.StatusCode == http.StatusOK:
			if *offset > 0 {
				restart("the server does not resume downloads")
			}
		case *offset > 0 && resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
			resp.Body.Close()
			cancel(nil)
			restart("the server cannot resume from the partial file")
			continue
		default:
			resp.Body.Close()
			cancel(nil)
			return httpError(resp.StatusCode, u)
		}
		if f.Size > 0 && resp.ContentLength >= 0 && *offset+resp.ContentLength != f.Size {
			resp.Body.Close()
			cancel(nil)
			return fmt.Errorf("the server has %d bytes for %s, but the recipe expects %d; the recipe or the upload is wrong", *offset+resp.ContentLength, u, f.Size)
		}
		err = d.write(reqCtx, cancel, part, *offset, resp.Body, h, f)
		resp.Body.Close()
		cancel(nil)
		if err != nil {
			if ctx.Err() != nil {
				return fmt.Errorf("download interrupted; run the command again to resume")
			}
			if errors.Is(context.Cause(reqCtx), errStalled) {
				return fmt.Errorf("the download received nothing for %s; run the command again to resume", d.stallTimeout())
			}
			return fmt.Errorf("download interrupted: %w; run the command again to resume", err)
		}
		return nil
	}
	return fmt.Errorf("could not resume the download; delete %s and try again", part)
}

func (d *Downloader) stallTimeout() time.Duration {
	if d.StallTimeout > 0 {
		return d.StallTimeout
	}
	return 2 * time.Minute
}

func (d *Downloader) write(ctx context.Context, cancel context.CancelCauseFunc, part string, offset int64, body io.Reader, h hash.Hash, f File) error {
	flags := os.O_CREATE | os.O_WRONLY
	if offset == 0 {
		flags |= os.O_TRUNC
	}
	file, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	if offset > 0 {
		if err := file.Truncate(offset); err != nil {
			file.Close()
			return err
		}
		if _, err := file.Seek(offset, io.SeekStart); err != nil {
			file.Close()
			return err
		}
	}
	p := newProgress(d.log(), "downloading "+path.Base(f.Path), f.Size, offset, d.TTY)
	stop := make(chan struct{})
	go func() { // abort a stalled download
		t := time.NewTicker(time.Second)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case <-t.C:
				if time.Since(p.lastWrite()) > d.stallTimeout() {
					cancel(errStalled)
					return
				}
			}
		}
	}()
	_, err = io.Copy(io.MultiWriter(file, h, p), body)
	close(stop)
	p.finish()
	if serr := file.Sync(); err == nil {
		err = serr
	}
	if cerr := file.Close(); err == nil {
		err = cerr
	}
	return err
}

func contentRangeStart(v string) (int64, bool) {
	// bytes 100-199/200
	v, ok := strings.CutPrefix(v, "bytes ")
	if !ok {
		return 0, false
	}
	start, _, ok := strings.Cut(v, "-")
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(start, 10, 64)
	return n, err == nil
}

func httpError(status int, u string) error {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("HTTP %d for %s: the repository does not exist, is private, or is gated. If you have access, accept its terms on the Hub and set HF_TOKEN", status, u)
	case http.StatusNotFound:
		return fmt.Errorf("HTTP 404 for %s: the recipe's repository, revision or file does not exist on the Hub", u)
	}
	return fmt.Errorf("HTTP %d for %s; try again later", status, u)
}

func (d *Downloader) hashFile(ctx context.Context, name string, h hash.Hash, label string, size int64) error {
	file, err := os.Open(name)
	if err != nil {
		return err
	}
	defer file.Close()
	p := newProgress(d.log(), label, size, 0, d.TTY)
	_, err = io.Copy(io.MultiWriter(h, p), ctxReader{ctx, file})
	p.finish()
	if err != nil && ctx.Err() != nil {
		return fmt.Errorf("interrupted")
	}
	return err
}

type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (c ctxReader) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}

func (d *Downloader) checkSpace(dir string, need int64) error {
	if need <= 0 {
		return nil
	}
	free := d.FreeSpace
	if free == nil {
		free = FreeSpace
	}
	avail, err := free(dir)
	if err != nil {
		return nil // unknown; the write fails cleanly if the disk fills
	}
	if avail < uint64(need) {
		return fmt.Errorf("not enough disk space in %s: the download needs %s, %s is free. Free some space, or pass --cache DIR on a larger disk",
			dir, FormatBytes(need), FormatBytes(int64(avail)))
	}
	return nil
}

// marker records that a blob was verified, so that later runs need not hash
// it again while its size and modification time stay the same.
type marker struct {
	SHA256  string `json:"sha256"`
	Size    int64  `json:"size"`
	MTimeNS int64  `json:"mtime_ns"`
	Path    string `json:"path"`
}

func (d *Downloader) markerPath(sha string) string {
	return filepath.Join(d.StateDir, "verified", sha+".json")
}

func (d *Downloader) markerOK(blob string, fi os.FileInfo, sha string) bool {
	if d.StateDir == "" {
		return false
	}
	data, err := os.ReadFile(d.markerPath(sha))
	if err != nil {
		return false
	}
	var m marker
	if json.Unmarshal(data, &m) != nil {
		return false
	}
	abs, _ := filepath.Abs(blob)
	return m.SHA256 == sha && m.Size == fi.Size() && m.MTimeNS == fi.ModTime().UnixNano() && m.Path == abs
}

func (d *Downloader) writeMarker(blob, sha string) {
	if d.StateDir == "" {
		return
	}
	fi, err := os.Stat(blob)
	if err != nil {
		return
	}
	abs, _ := filepath.Abs(blob)
	data, _ := json.Marshal(marker{SHA256: sha, Size: fi.Size(), MTimeNS: fi.ModTime().UnixNano(), Path: abs})
	p := d.markerPath(sha)
	err = os.MkdirAll(filepath.Dir(p), 0o755)
	if err == nil {
		tmp := p + ".tmp"
		if err = os.WriteFile(tmp, data, 0o644); err == nil {
			err = os.Rename(tmp, p)
		}
	}
	if err != nil {
		fmt.Fprintf(d.log(), "note: could not record the verification in %s (%v); the next run verifies again\n", d.StateDir, err)
	}
}
