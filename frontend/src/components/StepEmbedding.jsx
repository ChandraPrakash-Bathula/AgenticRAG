import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import Plotly from 'plotly.js/dist/plotly.min.js'
import { API } from '../config'
import StepExplainer from './StepExplainer'
import InfoPopover from './InfoPopover'

/** Lightweight Plotly wrapper, avoids react-plotly.js CJS issues */
/** Viewport width, tracked live, so plot heights can shrink on small screens.
 *  Plotly takes height as a number in the layout object, so a CSS media query
 *  cannot reach it. */
function useViewportWidth() {
  const [width, setWidth] = useState(() => (typeof window === 'undefined' ? 1200 : window.innerWidth))
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return width
}

function PlotChart({ data, layout, config, style }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current) return
    Plotly.react(ref.current, data, layout, config)
    return () => { if (ref.current) Plotly.purge(ref.current) }
  }, [data, layout, config])
  return <div ref={ref} style={style} />
}

export default function StepEmbedding({ onComplete, onDataChanged, embedData }) {
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(embedData)
  const [vizMode, setVizMode] = useState('3d')
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API}/embedding-models`).then(r => setModels(r.data.models))
  }, [])

  const runEmbedding = async () => {
    if (!selected) return
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API}/embed`, { model_name: selected })
      setResult(res.data)
      onDataChanged?.()
    } catch (err) {
      setError(err.response?.data?.detail || 'Embedding failed')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Chunk index → hue sweep, so position in the document stays readable in the plot.
  // Tuned for a white ground: lower lightness than a dark theme needs, or every marker
  // washes out against the paper.
  const viewportWidth = useViewportWidth()
  const isNarrow = viewportWidth < 640
  const plot2dHeight = isNarrow ? 320 : 450
  const plot3dHeight = isNarrow ? 340 : 500

  const plotColors = result?.chunk_labels?.map((_, i) => {
    const hue = (i / (result.chunk_labels.length || 1)) * 300
    return `hsl(${hue}, 58%, 46%)`
  }) || []

  // Plot chrome, matched to the CSS design tokens (--ink-faint, --rule, --paper).
  const PLOT_INK = '#78787f'
  const PLOT_GRID = '#e3e3df'
  const PLOT_ZERO = '#cbcbc4'
  const PLOT_PAPER = '#ffffff'

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon">🧬</div>
        <div>
          <div className="card-title">Embedding Model</div>
          <div className="card-description">Select a model to embed your chunks into vector space</div>
        </div>
      </div>

      <StepExplainer
        summary="Each chunk becomes an embedding, a list of numbers (a vector) where similar meaning lands at nearby points. When you ask a question later, it gets embedded the same way and the nearest chunks are your search results."
        details={[
          'The embedding model reads each chunk and outputs a vector (e.g. 384 or 768 numbers). All of them go into a FAISS index built for fast nearest-neighbor search.',
          '"Nearest" means cosine similarity: how aligned two vectors are, from −1 to 1. This one number is the entire basis of retrieval, no keywords, no exact matching.',
          'Different models produce different geometry: the same two chunks can be neighbors under one model and far apart under another. That is why the model choice changes retrieval results.',
          'The plots below squash hundreds of dimensions into 2 or 3 just for viewing, clusters are suggestive, not exact. Use the Table View for the actual projected numbers.',
        ]}
      />

      <div className="select-grid">
        {models.map(m => (
          <div key={m.id} className="select-card-wrap">
          <button type="button" className={`select-card ${selected === m.id ? 'selected' : ''}`} aria-pressed={selected === m.id} onClick={() => { setSelected(m.id); setResult(null); setError(null) }}>
            <div className="select-card-icon">{m.icon}</div>
            <div className="select-card-name">{m.name}</div>
            <div className="select-card-desc">{m.description}</div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <div className="select-card-badge">{m.dim}d</div>
              {m.cached ? (
                <div className="select-card-badge" style={{ background: 'rgba(52,211,153,0.15)', color: 'var(--success)' }}>
                  ✓ downloaded
                </div>
              ) : (
                <div className="select-card-badge" style={{ background: 'rgba(251,191,36,0.15)', color: 'var(--warning)' }}
                     title="The model weights are fetched from HuggingFace the first time you use this model, so expect a wait.">
                  ⬇ {m.download_size} on first use
                </div>
              )}
            </div>
          </button>
          {m.explainer && <InfoPopover label={m.name}>{m.explainer}</InfoPopover>}
          </div>
        ))}
      </div>

      {selected && !result && (
        <div style={{ marginTop: '20px' }}>
          <button className="btn btn-primary" onClick={runEmbedding} disabled={loading}>
            {loading ? 'Embedding...' : 'Generate Embeddings'}
          </button>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Re-embedding resets the LLM and query steps.
          </div>
        </div>
      )}

      {loading && <div className="loading-container"><div className="spinner" /><div className="loading-text">Downloading model & generating embeddings (first run may take a minute)...</div></div>}

      {error && <div style={{ color: 'var(--error)', padding: '16px', textAlign: 'center', marginTop: '16px' }}>{error}</div>}

      {result && (
        <div style={{ marginTop: '24px' }}>
          <div className="success-banner">✅ Generated {result.num_embeddings} embeddings ({result.embedding_dim}d) with {result.model}</div>
          <div className="stats-row">
            <div className="stat-card"><div className="stat-value">{result.num_embeddings}</div><div className="stat-label">Vectors</div></div>
            <div className="stat-card"><div className="stat-value">{result.embedding_dim}</div><div className="stat-label">Dimensions</div></div>
          </div>

          <div className="section-title">Embedding Space Visualization</div>
          {(result.visualization_error || !result.visualization_2d || !result.visualization_3d) ? (
            <div style={{
              padding: '20px', borderRadius: 'var(--radius)', background: 'var(--warning-wash)',
              border: '1px solid var(--warning-line)', color: 'var(--warning)', fontSize: '0.85rem', lineHeight: 1.6,
            }}>
              ⚠️ {result.visualization_error || 'Embedding visualization unavailable.'}
              <div style={{ color: 'var(--text-muted)', marginTop: '6px' }}>
                The embeddings and retrieval index were built successfully, only the plot is unavailable.
                No placeholder points are shown, because a fake scatter would suggest structure that isn't real.
              </div>
            </div>
          ) : (
          <>
          <div className="viz-toggle">
            <button className={`viz-toggle-btn ${vizMode === '2d' ? 'active' : ''}`} aria-pressed={vizMode === '2d'} onClick={() => setVizMode('2d')}>2D View</button>
            <button className={`viz-toggle-btn ${vizMode === '3d' ? 'active' : ''}`} aria-pressed={vizMode === '3d'} onClick={() => setVizMode('3d')}>3D View</button>
            <button className={`viz-toggle-btn ${vizMode === 'table' ? 'active' : ''}`} aria-pressed={vizMode === 'table'} onClick={() => setVizMode('table')}>Table View</button>
          </div>

          {vizMode === 'table' ? (
            <div className="viz-table-wrap">
              <table className="viz-table">
                <caption className="sr-only">Projected embedding coordinates per chunk</caption>
                <thead>
                  <tr>
                    <th scope="col">Chunk</th>
                    <th scope="col">Text (start)</th>
                    <th scope="col">2D x</th>
                    <th scope="col">2D y</th>
                    <th scope="col">3D x</th>
                    <th scope="col">3D y</th>
                    <th scope="col">3D z</th>
                  </tr>
                </thead>
                <tbody>
                  {result.chunk_labels.map((label, i) => (
                    <tr key={i}>
                      <td>#{i}</td>
                      <td>{label}</td>
                      <td className="num">{result.visualization_2d[i]?.[0]?.toFixed(3)}</td>
                      <td className="num">{result.visualization_2d[i]?.[1]?.toFixed(3)}</td>
                      <td className="num">{result.visualization_3d[i]?.[0]?.toFixed(3)}</td>
                      <td className="num">{result.visualization_3d[i]?.[1]?.toFixed(3)}</td>
                      <td className="num">{result.visualization_3d[i]?.[2]?.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
          <div className="plot-container">
            {vizMode === '2d' ? (
              <PlotChart
                data={[{
                  x: result.visualization_2d.map(p => p[0]),
                  y: result.visualization_2d.map(p => p[1]),
                  mode: 'markers+text',
                  type: 'scatter',
                  marker: { size: 10, color: plotColors, opacity: 0.9, line: { width: 1, color: PLOT_PAPER } },
                  text: result.chunk_labels.map((l, i) => `#${i}`),
                  textposition: 'top center',
                  textfont: { size: 9, color: PLOT_INK },
                  hovertext: result.chunk_labels,
                  hoverinfo: 'text',
                }]}
                layout={{
                  paper_bgcolor: PLOT_PAPER, plot_bgcolor: PLOT_PAPER,
                  font: { color: PLOT_INK, family: 'Inter' },
                  margin: isNarrow ? { t: 10, b: 32, l: 32, r: 10 } : { t: 20, b: 40, l: 40, r: 20 },
                  xaxis: { gridcolor: PLOT_GRID, zerolinecolor: PLOT_ZERO, linecolor: PLOT_GRID },
                  yaxis: { gridcolor: PLOT_GRID, zerolinecolor: PLOT_ZERO, linecolor: PLOT_GRID },
                  height: plot2dHeight, autosize: true,
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />
            ) : (
              <PlotChart
                data={[{
                  x: result.visualization_3d.map(p => p[0]),
                  y: result.visualization_3d.map(p => p[1]),
                  z: result.visualization_3d.map(p => p[2]),
                  mode: 'markers',
                  type: 'scatter3d',
                  marker: { size: 5, color: plotColors, opacity: 0.95, line: { width: 0.5, color: PLOT_PAPER } },
                  text: result.chunk_labels,
                  hoverinfo: 'text',
                }]}
                layout={{
                  paper_bgcolor: PLOT_PAPER,
                  scene: {
                    bgcolor: PLOT_PAPER,
                    xaxis: { gridcolor: PLOT_GRID, showbackground: false, zerolinecolor: PLOT_ZERO },
                    yaxis: { gridcolor: PLOT_GRID, showbackground: false, zerolinecolor: PLOT_ZERO },
                    zaxis: { gridcolor: PLOT_GRID, showbackground: false, zerolinecolor: PLOT_ZERO },
                  },
                  font: { color: PLOT_INK, family: 'Inter' },
                  margin: { t: 10, b: 10, l: 10, r: 10 },
                  height: plot3dHeight, autosize: true,
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />
            )}
          </div>
          )}
          <div style={{ marginTop: '10px', fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            ℹ️ These 2D/3D views are separate UMAP/t-SNE projections of the same high-dimensional vectors.
            Projections distort distances, apparent neighbors here are suggestive, not a faithful distance
            metric, and the 2D and 3D views may disagree with each other. Explore with it; don't measure with it.
          </div>
          </>
          )}

          <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
            <button className="btn btn-primary" onClick={() => onComplete(result)}>Continue to LLM Selection →</button>
          </div>
        </div>
      )}
    </div>
  )
}
