import { useEffect, useState } from "react";
import { ApiError } from "../api/client";

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; message: string };

/**
 * Runs `fn` whenever `deps` changes, tracking loading/success/error state.
 * Every data-fetching component in the app goes through this (or `useAsync`
 * directly) so loading/error handling stays consistent — see api/client.ts
 * for why `message` is always safe to render (never a raw exception).
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });

    fn()
      .then((data) => {
        if (!cancelled) setState({ status: "success", data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message = err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
        setState({ status: "error", message });
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
