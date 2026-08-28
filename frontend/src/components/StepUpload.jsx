import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { API } from '../config'
import StepExplainer from './StepExplainer'

export default function StepUpload({
  onComplete, onDataChanged, pdfData,
  exampleMode, exampleLoading, exampleError, onLoadExample, onExitExample,
}) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(pdfData)
  const [error, setError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post(`${API}/upload-pdf`, formData)
      setResult(res.data)
      onDataChanged?.()
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }, [onDataChanged])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
    },
    maxFiles: 1,
  })

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon">📄</div>
        <div>
          <div className="card-title">Upload Document</div>
          <div className="card-description">PDF (text extracted via PyMuPDF), plain text, or Markdown</div>
        </div>
      </div>

      <StepExplainer
        summary="An LLM can't read a PDF file, only plain text. This step extracts the text layer from your PDF so every later step has something to work with."
        details={[
          'PyMuPDF walks the PDF page by page and pulls out its text layer. There is no OCR, so a scanned or image-only PDF will come back empty.',
          'Everything downstream (chunking, embeddings, retrieval, answers) operates on this extracted text, extraction quality caps the quality of the whole pipeline.',
          'Check the preview below after uploading: broken words, repeated headers/footers, and page numbers are noise that will flow into your chunks.',
        ]}
      />

      {!result && (
        <div className="try-mode-grid">
          <div {...getRootProps()} className={`dropzone try-mode-col ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            <div className="dropzone-icon">📁</div>
            <div className="dropzone-text">
              {isDragActive ? 'Drop your document here...' : 'Try it yourself, drag & drop, or click to browse'}
            </div>
            <div className="dropzone-hint">Supports .pdf, .txt, and .md files (scanned/image-only PDFs won't work, no OCR)</div>
          </div>

          <div className="try-mode-col example-cta">
            <div className="dropzone-icon">📖</div>
            <div className="dropzone-text">See a worked example instead</div>
            <div className="dropzone-hint">
              A real excerpt of the open textbook <em>Human Nutrition: 2020 Edition</em> (Ch. 3–6: Water &amp;
              Electrolytes, Carbohydrates, Lipids, Protein), pre-chunked and pre-embedded, with curated
              questions chosen to expose the gap: plain LLM guesses, naive RAG retrieves the exact figure,
              agentic RAG double-checks it.
            </div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={onLoadExample}
              disabled={exampleLoading}
              style={{ marginTop: '14px' }}
            >
              {exampleLoading ? 'Loading example (first time takes ~10s)...' : '📖 Try this example'}
            </button>
            {exampleError && (
              <div style={{ color: 'var(--error)', marginTop: '10px', fontSize: '0.8rem' }}>{exampleError}</div>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-container">
          <div className="spinner" />
          <div className="loading-text">Extracting text with PyMuPDF...</div>
        </div>
      )}

      {error && (
        <div style={{ color: 'var(--error)', padding: '12px', textAlign: 'center' }}>{error}</div>
      )}

      {result && (
        <div>
          <div className="success-banner">
            ✅ {exampleMode ? 'Loaded worked example:' : 'Successfully processed'} {result.filename}
          </div>
          {exampleMode && result.note && (
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '-8px', marginBottom: '16px' }}>{result.note}</div>
          )}

          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-value">{result.total_pages}</div>
              <div className="stat-label">Pages</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{(result.total_characters / 1000).toFixed(1)}k</div>
              <div className="stat-label">Characters</div>
            </div>
          </div>

          <div className="section-title">Text Preview</div>
          <div className="text-preview">{result.preview}</div>

          {exampleMode ? (
            <div style={{ marginTop: '20px', display: 'flex', gap: '12px', justifyContent: 'flex-end', alignItems: 'center' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginRight: 'auto' }}>
                Chunking and embedding are already done for this example, jump ahead with the step bar above.
              </span>
              <button className="btn btn-secondary" onClick={onExitExample}>
                Use my own document instead
              </button>
            </div>
          ) : (
            <div style={{ marginTop: '20px', display: 'flex', gap: '12px', justifyContent: 'flex-end', alignItems: 'center' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginRight: 'auto' }}>
                Uploading a new PDF resets all later steps.
              </span>
              <button className="btn btn-secondary" onClick={() => { setResult(null); setError(null) }}>
                Re-upload
              </button>
              <button className="btn btn-primary" onClick={() => onComplete(result)}>
                Continue to Chunking →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
