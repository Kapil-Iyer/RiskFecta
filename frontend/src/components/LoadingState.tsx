interface LoadingStateProps {
  label?: string;
}

export default function LoadingState({ label = "Loading…" }: LoadingStateProps) {
  return (
    <div className="state state--loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {label}
    </div>
  );
}
