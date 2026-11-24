#!/usr/bin/env python3
"""Compare llama-tokenize with the pack's tokenizer.json (Hugging Face tokenizers, add_special_tokens=False).

    PYTHONPATH=<dir with tokenizers> python3 check_tokenizer.py <pack folder> <llama-tokenize> <model gguf> <work dir>

The texts are this script's own (prose, code, ten languages, numbers/whitespace/emoji, chat markup) plus random fuzz.
Texts are written and read with newline='' so that \\r\\n survives.
"""
import os
import random
import subprocess
import sys

from tokenizers import Tokenizer

TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog. It's a well-known pangram; they'll say it's used for "
          "testing typewriters, fonts and keyboards. Isn't it? We've seen it, I'd say, in 1,234 places.\n\n"
          "In 1898, the U.S.A. annexed Hawaii — an event that's still debated today.",
    "code": "def fib(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n"
            "    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n\nif __name__ == '__main__':\n"
            "\tprint([fib(i) for i in range(10)])  # [0, 1, 1, 2, ...]\r\n    x = {'k': 3.14e-10, \"v\": None}\n",
    "multi": "Le train part à sept heures du matin. Der Zug fährt um sieben Uhr morgens ab. El tren sale a las siete. "
             "Поезд отправляется в семь утра. 火车早上七点出发。列車は朝7時に出発します。기차는 아침 7시에 출발합니다. "
             "القطار يغادر في السابعة صباحا. ट्रेन सुबह सात बजे निकलती है। Η αμαξοστοιχία φεύγει στις επτά. "
             "รถไฟออกเวลาเจ็ดโมงเช้า বাংলা ভাষা తెలుగు",
    "nums": "12345678901234567890 3.14159 -42 1e10 0x1F 1,000,000 $9.99 50%   \t\t  \n\n\n   trailing   \n"
            "😀😃🤖👍🏽 ☕️ 🇫🇷 👨‍👩‍👧‍👦  nbsp em  ",
    "chat": "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>"
            "What is the capital of Australia? Answer in one sentence.<|end_message|><|message_model|>",
    "mixed": "I'M SHOUTING, you're Whispering. DON'T stop. McDonald's iPhone's O'Neil's naïve café résumé "
             "ﬁne ﬂow Ⅻ ½ ² x²+y²=z² … «quotes» “smart” ‘single’ a/b/c\r\nline\rcr\n",
}


def fuzz(rng, n):
    pools = [" ", "\n", "\t", "\r\n", "a", "Z", "é", "ß", "1", "23", "'s", "'LL", "!", "?", "/", "—", "中", "ж",
             "😀", "́", "क्", " the", "THE", "<|end_message|>", "  ", "...", "#", "_", "`"]
    return "".join(rng.choice(pools) for _ in range(n))


def main():
    pack, tool, model, work = sys.argv[1:5]
    tok = Tokenizer.from_file(os.path.join(pack, "tokenizer.json"))
    rng = random.Random(0)
    texts = dict(TEXTS)
    for i in range(30):
        texts[f"fuzz{i}"] = fuzz(rng, rng.randint(5, 200))
    bad = 0
    for name, text in texts.items():
        p = os.path.join(work, f"tok_{name}.txt")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        with open(p, encoding="utf-8", newline="") as f:
            text = f.read()
        ref = tok.encode(text, add_special_tokens=False).ids
        r = subprocess.run([tool, "-m", model, "-f", p, "--ids", "--no-bos", "--no-escape", "--log-disable"],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        try:
            got = [int(x) for x in r.stdout.strip().strip("[]").split(",") if x.strip()]
        except ValueError:
            got = None
        if got != ref:
            bad += 1
            print("MISMATCH", name, r.returncode, r.stderr[-300:])
            if got:
                k = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
                print("  at", k, "got", got[k:k + 8], "ref", ref[k:k + 8])
    print(f"texts {len(texts)} mismatches {bad}")


if __name__ == "__main__":
    main()
