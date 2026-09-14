interface DateRangeFilterProps {
  start: string;
  end: string;
  onStartChange: (value: string) => void;
  onEndChange: (value: string) => void;
  onClear: () => void;
  invalid: boolean;
}

export default function DateRangeFilter({
  start,
  end,
  onStartChange,
  onEndChange,
  onClear,
  invalid,
}: DateRangeFilterProps) {
  return (
    <div className="date-filter">
      <label className="date-filter__field">
        <span>Start</span>
        <input type="date" value={start} onChange={(e) => onStartChange(e.target.value)} />
      </label>
      <label className="date-filter__field">
        <span>End</span>
        <input type="date" value={end} onChange={(e) => onEndChange(e.target.value)} />
      </label>
      {(start || end) && (
        <button type="button" className="retry-button" onClick={onClear}>
          Clear
        </button>
      )}
      {invalid && (
        <p className="field-error" role="alert">
          Start date must not be after end date.
        </p>
      )}
    </div>
  );
}
