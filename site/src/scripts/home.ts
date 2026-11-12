// Page-wide behaviour of the home page: reveal on scroll, copy buttons and counters.
import { motion, onVisible } from "./motion";

// Reveal: elements with data-reveal fade in once, when they first come on screen.
const io = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("is-in");
        io.unobserve(e.target);
      }
    }
  },
  { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
);
document.querySelectorAll("[data-reveal]").forEach((el) => io.observe(el));

// Copy: a button with data-copy copies that text, and says so for a moment.
document.querySelectorAll<HTMLButtonElement>("button[data-copy]").forEach((btn) => {
  const label = btn.querySelector("[data-label]");
  btn.addEventListener("click", async () => {
    const text = btn.dataset.copy ?? "";
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    btn.setAttribute("data-done", "");
    if (label) label.textContent = "Copied";
    window.setTimeout(() => {
      btn.removeAttribute("data-done");
      if (label) label.textContent = "Copy";
    }, 1600);
  });
});

// Counters: an element with data-count="30.48" counts up to it, once, when it comes on screen.
document.querySelectorAll<HTMLElement>("[data-count]").forEach((el) => {
  const target = Number(el.dataset.count);
  const from = Number(el.dataset.from ?? 0);
  const decimals = (el.dataset.count?.split(".")[1] ?? "").length;
  const fmt = (v: number) =>
    v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  if (!motion.running) {
    el.textContent = fmt(target);
    return;
  }
  el.textContent = fmt(from);
  let started = false;
  const obs = onVisible(el, (visible) => {
    if (!visible || started) return;
    started = true;
    obs.disconnect();
    const dur = Number(el.dataset.duration ?? 1600);
    const t0 = performance.now();
    const step = (now: number) => {
      const p = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 4);
      el.textContent = fmt(from + (target - from) * eased);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
});
