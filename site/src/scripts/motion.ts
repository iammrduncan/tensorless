// Motion on the home page respects the reader's reduced-motion setting. Each animation also has
// its own control where it runs continuously (the model map's play button).
type Listener = (running: boolean) => void;

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
const listeners = new Set<Listener>();
reduced.addEventListener("change", () => listeners.forEach((fn) => fn(!reduced.matches)));

export const motion = {
  get running() {
    return !reduced.matches;
  },
  on(fn: Listener) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

/** Calls fn(true) when el is on screen and fn(false) when it leaves. */
export function onVisible(el: Element, fn: (visible: boolean) => void, rootMargin = "0px") {
  const io = new IntersectionObserver((entries) => entries.forEach((e) => fn(e.isIntersecting)), { rootMargin });
  io.observe(el);
  return io;
}
