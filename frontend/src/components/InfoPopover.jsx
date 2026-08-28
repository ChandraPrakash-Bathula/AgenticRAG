import { useEffect, useRef, useState } from 'react'

/**
 * Small ⓘ button that toggles an explanatory popover. Kept OUTSIDE any parent
 * <button> (nested buttons are invalid HTML), place it in a positioned wrapper
 * next to the element it describes. Closes on outside click or Escape.
 */
export default function InfoPopover({ label, children }) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <span className="info-pop" ref={wrapRef}>
      <button
        type="button"
        className="info-pop-btn"
        aria-expanded={open}
        aria-label={`What is ${label}?`}
        onClick={(e) => { e.stopPropagation(); setOpen(o => !o) }}
      >
        i
      </button>
      {open && (
        <div className="info-pop-panel" role="note" aria-label={`About ${label}`}>
          <div className="info-pop-title">{label}</div>
          <div className="info-pop-body">{children}</div>
        </div>
      )}
    </span>
  )
}
