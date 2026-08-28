import { useState, useEffect } from 'react'
import axios from 'axios'
import { API } from '../config'
import StepExplainer from './StepExplainer'
import InfoPopover from './InfoPopover'

export default function StepChunking({ onComplete, onDataChanged, chunkData }) {
  const [strategies, setStrategies] = useState([])
  const [selected, setSelected] = useState(null)
  const [params, setParams] = useState({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(chunkData)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API}/chunking-strategies`).then(r => setStrategies(r.data.strategies))
  }, [])

  // When strategy changes, initialize its default params
  const selectStrategy = (strategy) => {
    setSelected(strategy.id)
    setResult(null)
    setError(null)
    const defaults = {}
    ;(strategy.params || []).forEach(p => {
      defaults[p.key] = p.default
    })
    setParams(defaults)
  }

  const updateParam = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }

  const selectedStrategy = strategies.find(s => s.id === selected)

  const runChunking = async () => {
    if (!selected) return
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API}/chunk`, {
        strategy: selected,
        ...params,
      })
      setResult(res.data)
      onDataChanged?.()
    } catch (err) {
      setResult(null)
      setError(err.response?.data?.detail || 'Chunking failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon">✂️</div>
        <div>
          <div className="card-title">Chunking Strategy</div>
          <div className="card-description">Select how to split your document into chunks</div>
        </div>
      </div>

      <StepExplainer
        summary="The document text gets split into small pieces called chunks. Later, retrieval finds whole chunks. A chunk is the smallest unit of evidence your pipeline can ever fetch."
        details={[
          "Why not give the LLM the whole document? Cost, context-window limits, and diluted attention. The model answers better from a few focused passages than from 50 pages.",
          'Chunk size is the core trade-off: too small and ideas get cut in half mid-sentence; too big and retrieval gets imprecise (a "relevant" chunk is mostly off-topic filler).',
          'Overlap repeats a little text between neighboring chunks so a sentence on a boundary is not lost to both.',
          'The four strategies differ only in WHERE they cut: fixed ignores structure, recursive/sentence/paragraph respect increasingly natural boundaries. Compare the previews below, same document, very different pieces.',
        ]}
      />

      <div className="section-title">Choose Strategy</div>
      <div className="select-grid">
        {strategies.map(s => (
          <div key={s.id} className="select-card-wrap">
            <button type="button" className={`select-card ${selected === s.id ? 'selected' : ''}`} aria-pressed={selected === s.id} onClick={() => selectStrategy(s)}>
              <div className="select-card-icon">{s.icon}</div>
              <div className="select-card-name">{s.name}</div>
              <div className="select-card-desc">{s.description}</div>
            </button>
            {s.explainer && <InfoPopover label={s.name}>{s.explainer}</InfoPopover>}
          </div>
        ))}
      </div>

      {selected && selectedStrategy && (
        <div style={{ marginTop: '24px' }}>
          <div className="section-title">
            Parameters for {selectedStrategy.name}
          </div>

          {(selectedStrategy.params || []).map(p => (
            <div className="slider-group" key={p.key}>
              <div className="slider-label">
                <span>{p.label}</span>
                <span className="slider-value">
                  {params[p.key] ?? p.default}{p.key.includes('size') || p.key.includes('length') || p.key.includes('overlap') ? ' chars' : ''}
                </span>
              </div>
              <input
                type="range"
                aria-label={p.label}
                min={p.min}
                max={p.max}
                step={p.step}
                value={params[p.key] ?? p.default}
                onChange={e => updateParam(p.key, +e.target.value)}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                <span>{p.min}</span>
                <span>{p.max}</span>
              </div>
            </div>
          ))}

          <button className="btn btn-primary" onClick={runChunking} disabled={loading} style={{ marginTop: '8px' }}>
            {loading ? 'Chunking...' : 'Run Chunking'}
          </button>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Re-running chunking resets the embedding and later steps.
          </div>
        </div>
      )}

      {loading && <div className="loading-container"><div className="spinner" /><div className="loading-text">Splitting document...</div></div>}

      {error && (
        <div style={{ color: 'var(--error)', padding: '12px', textAlign: 'center' }}>{error}</div>
      )}

      {result && (
        <div style={{ marginTop: '24px' }}>
          <div className="success-banner">✅ Created {result.total_chunks} chunks using {result.strategy}</div>
          <div className="stats-row">
            <div className="stat-card"><div className="stat-value">{result.total_chunks}</div><div className="stat-label">Total Chunks</div></div>
            <div className="stat-card"><div className="stat-value">{result.avg_chunk_length}</div><div className="stat-label">Avg Length</div></div>
            <div className="stat-card"><div className="stat-value">{result.min_chunk_length}</div><div className="stat-label">Min Length</div></div>
            <div className="stat-card"><div className="stat-value">{result.max_chunk_length}</div><div className="stat-label">Max Length</div></div>
          </div>
          <div className="section-title">Chunks Preview</div>
          <div className="chunks-container">
            {result.chunks_preview.map(c => (
              <div key={c.index} className="chunk-item">
                <div className="chunk-header">
                  <span className="chunk-index">Chunk #{c.index}</span>
                  <span className="chunk-length">{c.length} chars</span>
                </div>
                {c.text}
              </div>
            ))}
          </div>
          <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
            <button className="btn btn-primary" onClick={() => onComplete(result)}>Continue to Embedding →</button>
          </div>
        </div>
      )}
    </div>
  )
}
