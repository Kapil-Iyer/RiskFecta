interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

/** Renders an API/network error. `message` must already be UI-safe (see
 * api/client.ts's ApiError) — never pass a raw exception/stack trace here. */
export default function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="state state--error" role="alert">
      <span>{message}</span>
      {onRetry && (
        <button type="button" className="retry-button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
