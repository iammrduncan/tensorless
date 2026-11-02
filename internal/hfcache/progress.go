package hfcache

import (
	"fmt"
	"io"
	"sync/atomic"
	"time"
)

// progress prints percent and MB/s to w, a few times per second on a
// terminal and every few seconds otherwise.
type progress struct {
	w         io.Writer
	label     string
	total     int64 // 0 if unknown
	done      int64
	startDone int64
	start     time.Time
	last      time.Time
	interval  time.Duration
	tty       bool
	printed   bool
	lastNS    atomic.Int64
}

func newProgress(w io.Writer, label string, total, done int64, tty bool) *progress {
	now := time.Now()
	p := &progress{w: w, label: label, total: total, done: done, startDone: done, start: now, last: now, tty: tty}
	p.interval = 5 * time.Second
	if tty {
		p.interval = 250 * time.Millisecond
	}
	p.lastNS.Store(now.UnixNano())
	return p
}

func (p *progress) Write(b []byte) (int, error) {
	now := time.Now()
	p.lastNS.Store(now.UnixNano())
	p.done += int64(len(b))
	if now.Sub(p.last) >= p.interval {
		p.last = now
		p.print(now)
	}
	return len(b), nil
}

func (p *progress) lastWrite() time.Time { return time.Unix(0, p.lastNS.Load()) }

func (p *progress) print(now time.Time) {
	line := p.label + "  "
	if p.total > 0 {
		line += fmt.Sprintf("%5.1f%%  %s / %s", 100*float64(p.done)/float64(p.total), FormatBytes(p.done), FormatBytes(p.total))
	} else {
		line += FormatBytes(p.done)
	}
	if secs := now.Sub(p.start).Seconds(); secs > 0 {
		line += fmt.Sprintf("  %.1f MB/s", float64(p.done-p.startDone)/secs/1e6)
	}
	if p.tty {
		fmt.Fprintf(p.w, "\r%s\x1b[K", line)
	} else {
		fmt.Fprintln(p.w, line)
	}
	p.printed = true
}

// finish prints the final state if anything was printed before.
func (p *progress) finish() {
	if !p.printed {
		return
	}
	p.print(time.Now())
	if p.tty {
		fmt.Fprintln(p.w)
	}
}

// FormatBytes formats n in decimal units: 12.3 GB, 512.0 MB.
func FormatBytes(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.2f GB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.1f MB", float64(n)/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%.1f kB", float64(n)/1e3)
	}
	return fmt.Sprintf("%d B", n)
}
