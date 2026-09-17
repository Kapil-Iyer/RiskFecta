import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver; PlotlyChart uses one to keep charts responsive.
if (typeof globalThis.ResizeObserver === "undefined") {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  globalThis.ResizeObserver = ResizeObserverStub;
}

// jsdom has no IntersectionObserver; framer-motion's `whileInView` (used by
// ResearchPipelineStory) needs one to mount at all. The stub never actually
// fires a callback — tests assert the elements are present in the DOM, not
// that the scroll-reveal transition itself played.
if (typeof globalThis.IntersectionObserver === "undefined") {
  class IntersectionObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }
  // @ts-expect-error - a test-only stub, not a spec-complete implementation
  globalThis.IntersectionObserver = IntersectionObserverStub;
}
