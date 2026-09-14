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
