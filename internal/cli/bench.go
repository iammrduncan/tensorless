package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/iammrduncan/tensorless/internal/container"
)

// benchSeed is the seed the hill-climb judge uses.
const benchSeed = 424242

// BenchPrompt is the prompt tensorless bench sends: the first 2,000
// characters of the license text, and an instruction to continue it.
func BenchPrompt(license string) string {
	r := []rune(license)
	if len(r) > 2000 {
		r = r[:2000]
	}
	return string(r) + "\n\nContinue the text above."
}

type benchRun struct {
	DecodeTPS    float64 `json:"decode_tps"`
	TTFTMillis   float64 `json:"ttft_ms"`
	PromptTokens int     `json:"prompt_tokens,omitempty"`
	Tokens       int     `json:"tokens"`
}

type benchOut struct {
	Model            string     `json:"model"`
	Card             string     `json:"card"`
	Endpoint         string     `json:"endpoint"`
	Runs             []benchRun `json:"runs"`
	MedianDecodeTPS  float64    `json:"median_decode_tps"`
	MedianTTFTMillis float64    `json:"median_ttft_ms"`
	PromptTokens     int        `json:"prompt_tokens,omitempty"`
	ReceiptDecodeTPS float64    `json:"receipt_decode_tps,omitempty"`
	VsReceiptPercent *float64   `json:"vs_receipt_percent,omitempty"`
}

// cmdBench measures decode speed against an already running server, the way
// the hill-climb judge does: one warm-up, then --runs greedy generations of
// --tokens tokens, timed per streamed event.
func (a *App) cmdBench(ctx context.Context, o *options) error {
	set, err := a.loadRecipes(o)
	if err != nil {
		return err
	}
	t, err := a.resolve(ctx, set, o, true)
	if err != nil {
		return err
	}
	r := t.r
	base := container.BaseURL(o.host, o.port)
	if !healthy(ctx, localClient(5*time.Second), base+"/health") {
		return fmt.Errorf("no server answers %s/health. Start one with: tensorless serve %s --detach", base, r.Repo())
	}
	client := localClient(10 * time.Minute)
	a.checkServedModel(ctx, base, r.Run.Alias)

	prompt := BenchPrompt(a.License)
	if !o.json {
		fmt.Fprintf(a.Stderr, "benchmarking %s at %s: warm-up, then %d run%s of %d tokens\n", r.Repo(), base, o.runs, plural(o.runs, "", "s"), o.tokens)
	}
	if _, err := a.completion(ctx, client, base, prompt, 8, false); err != nil {
		return err
	}
	out := benchOut{Model: r.Repo(), Card: r.Card.ID, Endpoint: base, Runs: []benchRun{}}
	for i := 0; i < o.runs; i++ {
		run, err := a.completion(ctx, client, base, prompt, o.tokens, true)
		if err != nil {
			return err
		}
		if !o.json {
			fmt.Fprintf(a.Stderr, "run %d: %.2f tok/s decode, TTFT %.0f ms\n", i+1, run.DecodeTPS, run.TTFTMillis)
		}
		out.Runs = append(out.Runs, run)
	}
	var tps, ttft []float64
	for _, run := range out.Runs {
		tps = append(tps, run.DecodeTPS)
		ttft = append(ttft, run.TTFTMillis)
		if run.PromptTokens > 0 {
			out.PromptTokens = run.PromptTokens
		}
	}
	out.MedianDecodeTPS, out.MedianTTFTMillis = median(tps), median(ttft)
	if rec := r.Receipt.DecodeTPS["short"]; rec > 0 {
		out.ReceiptDecodeTPS = rec
		pct := 100 * (out.MedianDecodeTPS/rec - 1)
		out.VsReceiptPercent = &pct
	}
	if o.json {
		return writeJSON(a.Stdout, out)
	}

	w := a.Stdout
	fmt.Fprintf(w, "%s on %s, %d runs of %d tokens (median):\n", r.Repo(), r.Card.ID, o.runs, o.tokens)
	fmt.Fprintf(w, "  decode         %.2f tok/s\n", out.MedianDecodeTPS)
	fmt.Fprintf(w, "  TTFT           %.0f ms\n", out.MedianTTFTMillis)
	if out.PromptTokens > 0 {
		fmt.Fprintf(w, "  prompt tokens  %d\n", out.PromptTokens)
	}
	if out.VsReceiptPercent != nil {
		fmt.Fprintf(w, "  receipt        %.2f tok/s decode (short); this run is %+.1f%%\n", out.ReceiptDecodeTPS, *out.VsReceiptPercent)
		fmt.Fprintln(w, "The receipt was measured with a different prompt, so expect a few percent of difference.")
	} else {
		fmt.Fprintln(w, "The recipe has no decode receipt to compare with.")
	}
	if why := untuned(t, o); why != "" {
		fmt.Fprintf(w, "note: %s; the server may not run the measured configuration either.\n", why)
	}
	return nil
}

// checkServedModel warns when the server does not list the recipe's alias.
func (a *App) checkServedModel(ctx context.Context, base, alias string) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/v1/models", nil)
	if err != nil {
		return
	}
	resp, err := localClient(5 * time.Second).Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&models) != nil {
		return
	}
	var ids []string
	for _, m := range models.Data {
		if m.ID == alias {
			return
		}
		ids = append(ids, m.ID)
	}
	if len(ids) > 0 {
		fmt.Fprintf(a.Stderr, "warning: the server at %s serves %s, not %s; the receipt comparison may not apply\n", base, strings.Join(ids, ", "), alias)
	}
}

type completionEvent struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`
	Timings *struct {
		PromptN    int `json:"prompt_n"`
		PredictedN int `json:"predicted_n"`
	} `json:"timings"`
}

// completion streams one greedy /completion and, if timed, times its events.
func (a *App) completion(ctx context.Context, client *http.Client, base, prompt string, tokens int, timed bool) (benchRun, error) {
	body, _ := json.Marshal(map[string]any{
		"prompt":       prompt,
		"n_predict":    tokens,
		"temperature":  0,
		"top_k":        1,
		"seed":         benchSeed,
		"ignore_eos":   true,
		"stream":       true,
		"cache_prompt": false,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/completion", bytes.NewReader(body))
	if err != nil {
		return benchRun{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return benchRun{}, errInterrupted
		}
		return benchRun{}, fmt.Errorf("POST %s/completion: %v", base, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return benchRun{}, fmt.Errorf("POST %s/completion: HTTP %d: %s", base, resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	var first, last time.Time
	var events int
	run := benchRun{Tokens: tokens}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for sc.Scan() {
		now := time.Now() // monotonic
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			break
		}
		var ev completionEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return benchRun{}, fmt.Errorf("unexpected event from %s/completion: %.200s", base, data)
		}
		if ev.Content != "" {
			if events == 0 {
				first = now
			}
			last = now
			events++
		}
		if ev.Timings != nil {
			run.PromptTokens = ev.Timings.PromptN
			if ev.Timings.PredictedN > 0 {
				run.Tokens = ev.Timings.PredictedN
			}
		}
		if ev.Stop {
			break
		}
	}
	if err := sc.Err(); err != nil {
		if ctx.Err() != nil {
			return benchRun{}, errInterrupted
		}
		return benchRun{}, fmt.Errorf("reading %s/completion: %v", base, err)
	}
	if !timed {
		return run, nil
	}
	if events < 2 || !last.After(first) {
		return benchRun{}, errors.New("the server streamed fewer than two tokens; is it a llama-server?")
	}
	run.DecodeTPS = float64(run.Tokens-1) / last.Sub(first).Seconds()
	run.TTFTMillis = float64(first.Sub(start).Microseconds()) / 1000
	return run, nil
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}
