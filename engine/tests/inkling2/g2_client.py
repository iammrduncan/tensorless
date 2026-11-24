import json, os, sys, time, urllib.request
port, pid = int(sys.argv[1]), int(sys.argv[2])
base = f"http://127.0.0.1:{port}"
t0 = time.time()
while True:
    try:
        if b"ok" in urllib.request.urlopen(base + "/health", timeout=2).read():
            break
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        print("server died"); sys.exit(1)
    if time.time() - t0 > 900:
        print("timeout"); sys.exit(1)
    time.sleep(1)
print(f"ready after {time.time() - t0:.1f} s", flush=True)
qs = ["What is the capital of Australia? Answer in one sentence.",
      "Write a Python function that returns the n-th Fibonacci number.",
      "Explain in two sentences why the sky looks blue.",
      "Translate into French: The train leaves at seven in the morning."]
for q in qs:
    p = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
         f"<|message_user|><|content_text|>{q}<|end_message|>")
    body = json.dumps({"prompt": p, "n_predict": 64, "temperature": 0, "ignore_eos": True, "cache_prompt": False}).encode()
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/completion", body, {"Content-Type": "application/json"}), timeout=1200).read())
    except Exception as e:
        print("request failed", e); break
    print(f"{time.time() - t:.1f}s", repr(r.get("content")), {k: r.get("timings", {}).get(k) for k in ("prompt_per_second", "predicted_per_second")}, flush=True)
