import { useState, useEffect } from 'react'
import axios from 'axios'
import { API } from '../config'
import StepExplainer from './StepExplainer'
import InfoPopover from './InfoPopover'

export default function StepLLM({ onComplete, llmModel }) {
  const [models, setModels] = useState([])
  const [ollamaStatus, setOllamaStatus] = useState(null)
  const [graderInfo, setGraderInfo] = useState(null)
  const [selected, setSelected] = useState(llmModel)

  useEffect(() => {
    axios.get(`${API}/llm-models`).then(r => {
      setModels(r.data.models)
      setOllamaStatus(r.data.ollama_status)
      setGraderInfo(r.data.agentic_grader)
    })
  }, [])

  const selectModel = async (modelId) => {
    setSelected(modelId)
    await axios.post(`${API}/select-llm`, { model_name: modelId })
  }

  const installed = ollamaStatus?.installed_models || []
  const provider = ollamaStatus?.provider || 'ollama'
  const isGroq = provider === 'groq'

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon">🤖</div>
        <div>
          <div className="card-title">Select LLM</div>
          <div className="card-description">
            {isGroq
              ? 'Choose a model, powered by Groq Cloud (blazing fast ⚡)'
              : 'Choose an open-source LLM via Ollama for inference'}
          </div>
        </div>
      </div>

      <StepExplainer
        summary="Pick the model that WRITES the answers. This is separate from retrieval: the embedding model (previous step) finds the evidence, this model reads that evidence and composes the response."
        details={[
          'In the next step the retrieved chunks get pasted into this model\'s prompt with the instruction "answer ONLY from this context", that instruction plus good retrieval is what suppresses hallucination.',
          'A bigger generator gives better reading and writing of the evidence, not better retrieval, if the right chunk was never retrieved, no generator can recover it.',
          'Local (Ollama) = private and free but small models on your hardware; Groq Cloud = much larger models, very fast, needs an API key.',
          'Agentic mode additionally uses a small fast model behind the scenes for its routing/grading checks, regardless of what you pick here.',
        ]}
      />

      {ollamaStatus && (
        <div className="success-banner" style={
          !ollamaStatus.running
            ? { background: 'rgba(248,113,113,0.1)', borderColor: 'rgba(248,113,113,0.2)', color: 'var(--error)' }
            : isGroq
              ? { background: 'rgba(34,211,238,0.1)', borderColor: 'rgba(34,211,238,0.2)', color: 'var(--accent-3)' }
              : {}
        }>
          {!ollamaStatus.running
            ? '⚠️ Ollama is not running. Start it with: ollama serve'
            : isGroq
              ? '⚡ Groq Cloud API, all models available instantly with 500+ tokens/sec'
              : `✅ Ollama is running, ${installed.length} model(s) installed`
          }
        </div>
      )}

      {graderInfo && !graderInfo.available && (
        <div style={{
          padding: '12px 16px', borderRadius: 'var(--radius)', marginBottom: '16px',
          background: 'var(--warning-wash)', border: '1px solid var(--warning-line)',
          fontSize: '0.78rem', color: 'var(--warning)', lineHeight: 1.6
        }} role="status">
          ⚠️ {graderInfo.note}
        </div>
      )}

      {isGroq && (
        <div style={{
          padding: '12px 16px', borderRadius: 'var(--radius)', marginBottom: '16px',
          background: 'var(--accent-wash)', border: '1px solid var(--accent-line)',
          fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6
        }}>
          💡 <strong style={{ color: 'var(--accent-ink)' }}>Groq serves these models (7B–120B) in the cloud</strong>, free, and 10-20x faster than local GPU inference. Pick the 7B to see how much of the answer quality comes from retrieval rather than model size.
          <div style={{ marginTop: '8px' }}>
            Groq is an <strong>inference host, not a model vendor</strong>. Every model listed here is
            open-weight and downloadable from HuggingFace, so nothing in this pipeline depends on a
            closed model. Retrieval is local either way: your embeddings are computed on this machine
            by sentence-transformers and never leave it. Only the generation call goes to Groq, and
            switching <code>LLM_PROVIDER=ollama</code> moves that on-device too.
          </div>
        </div>
      )}

      <div className="select-grid">
        {models.map(m => {
          const isInstalled = isGroq || installed.some(im => im.startsWith(m.id.split(':')[0]) || im.startsWith(m.ollama_id?.split(':')[0]))
          return (
            <div key={m.id} className="select-card-wrap">
            <button
              type="button"
              className={`select-card ${selected === m.id ? 'selected' : ''}`}
              aria-pressed={selected === m.id}
              onClick={() => selectModel(m.id)}
            >
              <div className="select-card-icon">{m.icon}</div>
              <div className="select-card-name">{m.name}</div>
              <div className="select-card-desc">{m.description}</div>
              <div style={{ display: 'flex', gap: '8px', marginTop: '8px', flexWrap: 'wrap' }}>
                <div className="select-card-badge">{m.size}</div>
                {isGroq ? (
                  <div className="select-card-badge" style={{ background: 'rgba(34,211,238,0.15)', color: 'var(--accent-3)' }}>
                    ⚡ Groq Cloud
                  </div>
                ) : (
                  <div className="select-card-badge" style={isInstalled
                    ? { background: 'rgba(52,211,153,0.15)', color: 'var(--success)' }
                    : { background: 'rgba(251,191,36,0.15)', color: 'var(--warning)' }
                  }>
                    {isInstalled ? '✓ Installed' : 'Not pulled'}
                  </div>
                )}
              </div>
            </button>
            {m.explainer && <InfoPopover label={m.name}>{m.explainer}</InfoPopover>}
            </div>
          )
        })}
      </div>

      {!ollamaStatus?.running && !isGroq && (
        <div style={{ marginTop: '16px', padding: '16px', borderRadius: 'var(--radius)', background: 'var(--glass)', border: '1px solid var(--glass-border)' }}>
          <div className="section-title" style={{ marginTop: 0 }}>Quick Setup</div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.7, fontFamily: "'JetBrains Mono', monospace" }}>
            1. Install Ollama: <span style={{ color: 'var(--accent-1)' }}>brew install ollama</span><br/>
            2. Start server: <span style={{ color: 'var(--accent-1)' }}>ollama serve</span><br/>
            3. Pull a model: <span style={{ color: 'var(--accent-1)' }}>ollama pull llama3.2</span><br/>
            4. For Agentic mode's grader: <span style={{ color: 'var(--accent-1)' }}>ollama pull phi4-mini</span>
          </div>
        </div>
      )}

      {selected && (
        <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-primary" onClick={() => onComplete(selected)}>Continue to Query →</button>
        </div>
      )}
    </div>
  )
}
