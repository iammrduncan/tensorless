package cli

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/iammrduncan/tensorless"
)

type completionRequest struct {
	Prompt      string  `json:"prompt"`
	NPredict    int     `json:"n_predict"`
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
	Seed        int     `json:"seed"`
	IgnoreEOS   bool    `json:"ignore_eos"`
	Stream      bool    `json:"stream"`
	CachePrompt bool    `json:"cache_prompt"`
}

// sseServer fakes llama-server: /health, /v1/models and a streaming
// /completion that emits one event per token, perToken apart.
func sseServer(t *testing.T, alias string, perToken time.Duration) (*httptest.Server, *[]completionRequest) {
	var mu sync.Mutex
	var reqs []completionRequest
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"status":"ok"}`)) })
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"object":"list","data":[{"id":%q}]}`, alias)
	})
	mux.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		var req completionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		mu.Lock()
		reqs = append(reqs, req)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		time.Sleep(5 * time.Millisecond) // prompt processing
		for i := 0; i < req.NPredict; i++ {
			fmt.Fprintf(w, "data: {\"content\":\"t%d \",\"stop\":false}\n\n", i)
			w.(http.Flusher).Flush()
			time.Sleep(perToken)
		}
		fmt.Fprintf(w, "data: {\"content\":\"\",\"stop\":true,\"timings\":{\"prompt_n\":431,\"predicted_n\":%d}}\n\n", req.NPredict)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, &reqs
}

func hostPort(t *testing.T, raw string) (string, string) {
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return u.Hostname(), u.Port()
}

func TestBench(t *testing.T) {
	ta := newApp(t) // no GPU: the model's only recipe is used
	dir := ta.published(t, []byte("x"))
	srv, reqs := sseServer(t, "Org/Model-GGUF", 2*time.Millisecond)
	h, p := hostPort(t, srv.URL)

	code := ta.run("bench", "org/model-gguf", "--recipes", dir, "--host", h, "--port", p, "--runs", "3", "--tokens", "20", "--json")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	var out benchOut
	if err := json.Unmarshal([]byte(ta.stdout.String()), &out); err != nil {
		t.Fatalf("%v: %s", err, ta.stdout)
	}
	if len(out.Runs) != 3 || out.Model != "Org/Model-GGUF" || out.PromptTokens != 431 || out.ReceiptDecodeTPS != 30 {
		t.Errorf("out = %+v", out)
	}
	// 19 intervals of about 2 ms: a few hundred tok/s, well under 1000.
	if out.MedianDecodeTPS < 50 || out.MedianDecodeTPS > 1000 {
		t.Errorf("median decode = %.1f tok/s", out.MedianDecodeTPS)
	}
	if out.MedianTTFTMillis < 5 {
		t.Errorf("median TTFT = %.1f ms, want at least the 5 ms prompt time", out.MedianTTFTMillis)
	}
	if out.VsReceiptPercent == nil || *out.VsReceiptPercent < 0 {
		t.Errorf("vs receipt = %v", out.VsReceiptPercent)
	}

	if len(*reqs) != 4 {
		t.Fatalf("%d requests, want a warm-up and 3 runs", len(*reqs))
	}
	if (*reqs)[0].NPredict != 8 {
		t.Errorf("warm-up n_predict = %d", (*reqs)[0].NPredict)
	}
	for _, r := range (*reqs)[1:] {
		if r.NPredict != 20 || r.Temperature != 0 || r.TopK != 1 || r.Seed != 424242 || !r.IgnoreEOS || !r.Stream || r.CachePrompt {
			t.Errorf("request = %+v", r)
		}
		if !strings.HasPrefix(r.Prompt, tensorless.License[:200]) || !strings.HasSuffix(r.Prompt, "\n\nContinue the text above.") ||
			len(r.Prompt) != 2000+len("\n\nContinue the text above.") {
			t.Errorf("prompt: %d characters", len(r.Prompt))
		}
	}

	// Text output compares with the receipt and says the prompt differs.
	ta.stdout.b.Reset()
	if code := ta.run("bench", "m1", "--recipes", dir, "--host", h, "--port", p, "--runs", "1", "--tokens", "10"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	text := ta.stdout.String()
	for _, want := range []string{"decode ", "TTFT ", "prompt tokens  431", "receipt        30.00 tok/s", "different prompt"} {
		if !strings.Contains(text, want) {
			t.Errorf("bench output lacks %q:\n%s", want, text)
		}
	}
}

func TestBenchWarnsAboutAnotherModel(t *testing.T) {
	ta := newApp(t)
	dir := ta.published(t, []byte("x"))
	srv, _ := sseServer(t, "someone/else", 0)
	h, p := hostPort(t, srv.URL)
	if code := ta.run("bench", "m1", "--recipes", dir, "--host", h, "--port", p, "--runs", "1", "--tokens", "5"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	if !strings.Contains(ta.stderr.String(), "serves someone/else, not Org/Model-GGUF") {
		t.Errorf("stderr = %s", ta.stderr)
	}
}

func TestBenchWithoutServer(t *testing.T) {
	r := embedded(t)[0]
	ta := newApp(t)
	srv := httptest.NewServer(http.NotFoundHandler())
	h, p := hostPort(t, srv.URL)
	srv.Close() // nothing listens there now
	code := ta.run("bench", strings.ToLower(r.Repo()), "--card", r.Card.ID, "--host", h, "--port", p)
	if code != 1 {
		t.Fatalf("exit %d", code)
	}
	if e := ta.stderr.String(); !strings.Contains(e, "no server answers http://127.0.0.1:"+p+"/health") ||
		!strings.Contains(e, "tensorless serve "+r.Repo()+" --detach") {
		t.Errorf("stderr = %s", e)
	}
}

func TestBenchPrompt(t *testing.T) {
	p := BenchPrompt(strings.Repeat("é", 3000))
	if !strings.HasSuffix(p, "\n\nContinue the text above.") || len([]rune(p)) != 2000+len("\n\nContinue the text above.") {
		t.Errorf("BenchPrompt cut by bytes, not characters")
	}
}

// closingServer fakes a llama-server that ends each streamed /completion with a
// complete chunked body and then closes the connection, without sending
// "Connection: close" (seen through a kubectl port-forward on a real P100).
func closingServer(t *testing.T, alias string) (*httptest.Server, *int) {
	var mu sync.Mutex
	n := 0
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"status":"ok"}`)) })
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"object":"list","data":[{"id":%q}]}`, alias)
	})
	mux.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		var req completionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		mu.Lock()
		n++
		mu.Unlock()
		conn, buf, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
		chunk := func(s string) { fmt.Fprintf(buf, "%x\r\n%s\r\n", len(s), s) }
		for i := 0; i < req.NPredict; i++ {
			chunk(fmt.Sprintf("data: {\"content\":\"t%d \",\"stop\":false}\n\n", i))
		}
		chunk(fmt.Sprintf("data: {\"content\":\"\",\"stop\":true,\"timings\":{\"predicted_n\":%d}}\n\n", req.NPredict))
		buf.WriteString("0\r\n\r\n")
		buf.Flush()
		// Close late, as the port-forward did: by then the client has reused the
		// connection for its next request, which gets no answer.
		time.Sleep(100 * time.Millisecond)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, &n
}

func TestBenchServerClosesAfterStream(t *testing.T) {
	ta := newApp(t)
	dir := ta.published(t, []byte("x"))
	srv, n := closingServer(t, "Org/Model-GGUF")
	h, p := hostPort(t, srv.URL)
	if code := ta.run("bench", "org/model-gguf", "--recipes", dir, "--host", h, "--port", p, "--runs", "3", "--tokens", "10", "--json"); code != 0 {
		t.Fatalf("exit %d: %s", code, ta.stderr)
	}
	if *n != 4 {
		t.Errorf("%d completions served, want a warm-up and 3 runs", *n)
	}
}
