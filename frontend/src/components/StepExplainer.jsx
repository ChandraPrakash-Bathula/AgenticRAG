import { useState } from 'react'

/**
 * Per-step teaching panel: a one-line summary that's always visible, and an
 * expandable "what happens here" list for the full explanation.
 */
export default function StepExplainer({ summary, details, children }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="explainer">
      <div className="explainer-summary">📖 {summary}</div>
      <button
        type="button"
        className="explainer-toggle"
        aria-expanded={open}
        onClick={() => setOpen(o => !o)}
      >
        {open ? 'Hide details ▲' : 'What happens here? ▼'}
      </button>
      {open && (
        <div className="explainer-body">
          {details && (
            <ul className="explainer-list">
              {details.map((d, i) => <li key={i}>{d}</li>)}
            </ul>
          )}
          {children}
        </div>
      )}
    </div>
  )
}
