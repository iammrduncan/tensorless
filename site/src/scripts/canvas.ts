// Small helpers for the home page's 2D canvas animations: a crisp canvas at the device's pixel
// ratio, a soft glow sprite, a loop that runs only while the canvas is on screen, and easings.
import { motion, onVisible } from "./motion";

export type Scene = { ctx: CanvasRenderingContext2D; w: number; h: number };

export function crisp(canvas: HTMLCanvasElement, onResize: (s: Scene) => void): Scene {
  const ctx = canvas.getContext("2d")!;
  const s: Scene = { ctx, w: 0, h: 0 };
  const fit = () => {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    s.w = Math.max(1, r.width);
    s.h = Math.max(1, r.height);
    canvas.width = Math.round(s.w * dpr);
    canvas.height = Math.round(s.h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    onResize(s);
  };
  new ResizeObserver(fit).observe(canvas);
  fit();
  return s;
}

const sprites = new Map<string, HTMLCanvasElement>();
/** A soft round glow of one colour, drawn once and reused. */
export function glow(color: string): HTMLCanvasElement {
  let c = sprites.get(color);
  if (c) return c;
  c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, color);
  grad.addColorStop(0.25, color);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad;
  g.globalAlpha = 1;
  g.fillRect(0, 0, 64, 64);
  sprites.set(color, c);
  return c;
}

/** Runs draw(t) every frame while el is on screen and motion is allowed; t is seconds, looping. */
export function loop(el: Element, period: number, draw: (t: number) => void, still: number) {
  let raf = 0;
  let visible = false;
  let t = 0;
  let last = 0;
  let jumped = false;
  const tick = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!jumped) t = (t + dt) % period;
    jumped = false;
    draw(t);
    raf = requestAnimationFrame(tick);
  };
  const sync = () => {
    cancelAnimationFrame(raf);
    if (visible && motion.running) {
      last = performance.now();
      raf = requestAnimationFrame(tick);
    } else draw(motion.running ? t : still);
  };
  onVisible(el, (v) => {
    visible = v;
    sync();
  });
  motion.on(sync);
  return {
    seek(to: number) {
      t = to;
      jumped = true;
      if (!visible || !motion.running) draw(t);
    },
    redraw() {
      if (!visible || !motion.running) draw(motion.running ? t : still);
    },
  };
}

export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export const ease = (x: number) => {
  const k = clamp01(x);
  return k * k * (3 - 2 * k);
};
export const easeOut = (x: number) => 1 - Math.pow(1 - clamp01(x), 3);
/** Progress of t through [a, b]. */
export const span = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
