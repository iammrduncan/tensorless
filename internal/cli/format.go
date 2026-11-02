package cli

import (
	"strings"
	"sync"
)

// quote quotes s for a POSIX shell when it needs it. It is for display only:
// commands always run as an argv.
func quote(s string) string {
	if s == "" {
		return "''"
	}
	safe := true
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.ContainsRune("_-./:=@%+,", c)) {
			safe = false
			break
		}
	}
	if safe {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func shellJoin(argv []string) string {
	q := make([]string, len(argv))
	for i, s := range argv {
		q[i] = quote(s)
	}
	return strings.Join(q, " ")
}

// shellWrap formats argv for display over several lines with backslashes,
// keeping each flag on the same line as its value.
func shellWrap(argv []string, width int) string {
	var groups []string
	for i := 0; i < len(argv); i++ {
		g := quote(argv[i])
		if strings.HasPrefix(argv[i], "-") && i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
			g += " " + quote(argv[i+1])
			i++
		}
		groups = append(groups, g)
	}
	var lines []string
	cur := ""
	for _, g := range groups {
		if cur != "" && len(cur)+1+len(g) > width {
			lines = append(lines, cur)
			cur = "    " + g
			continue
		}
		if cur != "" {
			cur += " "
		}
		cur += g
	}
	if cur != "" {
		lines = append(lines, cur)
	}
	return strings.Join(lines, " \\\n")
}

// tail keeps the last n lines written to it. It is safe for concurrent use.
type tail struct {
	mu      sync.Mutex
	n       int
	lines   []string
	partial string
}

func newTail(n int) *tail { return &tail{n: n} }

func (t *tail) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := t.partial + string(p)
	parts := strings.Split(s, "\n")
	t.partial = parts[len(parts)-1]
	t.lines = append(t.lines, parts[:len(parts)-1]...)
	if len(t.lines) > t.n {
		t.lines = append([]string(nil), t.lines[len(t.lines)-t.n:]...)
	}
	return len(p), nil
}

func (t *tail) Lines() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := append([]string(nil), t.lines...)
	if t.partial != "" {
		out = append(out, t.partial)
	}
	if len(out) > t.n {
		out = out[len(out)-t.n:]
	}
	return out
}
