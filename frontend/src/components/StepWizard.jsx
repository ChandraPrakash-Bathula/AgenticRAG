import { useState } from 'react'
import axios from 'axios'
import { API } from '../config'
import StepUpload from './StepUpload'
import StepChunking from './StepChunking'
import StepEmbedding from './StepEmbedding'
import StepLLM from './StepLLM'
import StepQuery from './StepQuery'
import { completeAndInvalidate, invalidateDownstream, mergeAndClearDownstream } from '../utils/wizardState'

const STEPS = [
  { id: 1, label: 'Upload PDF', icon: '📄' },
  { id: 2, label: 'Chunking', icon: '✂️' },
  { id: 3, label: 'Embedding', icon: '🧬' },
  { id: 4, label: 'LLM', icon: '🤖' },
  { id: 5, label: 'Query', icon: '💬' },
]

const BLANK_PIPELINE_DATA = { pdfData: null, chunkData: null, embedData: null, llmModel: null }

export default function StepWizard() {
  const [currentStep, setCurrentStep] = useState(1)
  const [completedSteps, setCompletedSteps] = useState(new Set())
  const [pipelineData, setPipelineData] = useState(BLANK_PIPELINE_DATA)

  // "See a worked example" fast-forwards Steps 1-3 with a real, pre-chunked,
  // pre-embedded excerpt of an open nutrition textbook instead of the user's own
  // document, see backend/app/example_data.py for why and what it contains.
  const [exampleMode, setExampleMode] = useState(false)
  const [exampleQuestions, setExampleQuestions] = useState([])
  const [exampleLoading, setExampleLoading] = useState(false)
  const [exampleError, setExampleError] = useState(null)

  // Re-completing an earlier step invalidates everything after it, the backend has
  // already reset its downstream state (re-chunking wipes embeddings, etc.), and the
  // progress bar must agree with the backend, not contradict it.
  const completeStep = (step, data = {}) => {
    setCompletedSteps(prev => completeAndInvalidate(prev, step))
    setPipelineData(prev => mergeAndClearDownstream(prev, step, data))
    if (step < 5) setCurrentStep(step + 1)
  }

  // Fired by a step the moment its action re-runs successfully (before any Continue
  // click), otherwise a user could re-chunk and still jump to Query via the
  // still-checkmarked progress bar.
  const stepDataChanged = (step) => {
    setCompletedSteps(prev => invalidateDownstream(prev, step))
    setPipelineData(prev => mergeAndClearDownstream(prev, step, {}))
    if (step === 1) setExampleMode(false)
  }

  const loadExample = async () => {
    setExampleLoading(true)
    setExampleError(null)
    try {
      const res = await axios.post(`${API}/load-example`)
      setPipelineData({
        pdfData: res.data.upload,
        chunkData: res.data.chunk,
        embedData: res.data.embed,
        llmModel: null,
      })
      setExampleQuestions(res.data.questions)
      setCompletedSteps(new Set([1, 2, 3]))
      setExampleMode(true)
      setCurrentStep(4)
    } catch (err) {
      setExampleError(err.response?.data?.detail || 'Failed to load the worked example.')
    } finally {
      setExampleLoading(false)
    }
  }

  // Full reset back to "bring your own document", used by the exit link shown
  // throughout example mode, not just on Step 1.
  const exitExample = () => {
    setExampleMode(false)
    setExampleQuestions([])
    setExampleError(null)
    setPipelineData(BLANK_PIPELINE_DATA)
    setCompletedSteps(new Set())
    setCurrentStep(1)
  }

  const getStepStatus = (stepId) => {
    if (stepId === currentStep) return 'active'
    if (completedSteps.has(stepId)) return 'completed'
    return 'locked'
  }

  const goToStep = (stepId) => {
    if (completedSteps.has(stepId) || stepId === currentStep) {
      setCurrentStep(stepId)
    }
  }

  return (
    <div>
      <div className="step-progress">
        {STEPS.map((step, i) => (
          <div key={step.id} style={{ display: 'flex', alignItems: 'center', flex: i < STEPS.length - 1 ? 1 : 'none' }}>
            <button
              type="button"
              className={`step-indicator ${getStepStatus(step.id)}`}
              onClick={() => goToStep(step.id)}
              disabled={!completedSteps.has(step.id) && step.id !== currentStep}
              aria-current={step.id === currentStep ? 'step' : undefined}
              aria-label={`Step ${step.id}: ${step.label}${completedSteps.has(step.id) ? ' (completed)' : ''}`}
              style={{ cursor: completedSteps.has(step.id) || step.id === currentStep ? 'pointer' : 'default' }}
            >
              <div className="step-number">
                {completedSteps.has(step.id) ? '✓' : step.id}
              </div>
              <span className="step-label">{step.label}</span>
            </button>
            {i < STEPS.length - 1 && (
              <div className={`step-connector ${completedSteps.has(step.id) ? 'completed' : ''}`} />
            )}
          </div>
        ))}
      </div>

      {exampleMode && currentStep !== 1 && (
        <div className="example-mode-banner" role="status">
          <span>📖 Worked example active, <em>Human Nutrition: 2020 Edition</em> (Ch. 3–6), pre-chunked and pre-embedded for real.</span>
          <button type="button" className="example-mode-exit" onClick={exitExample}>Use my own document instead →</button>
        </div>
      )}

      {currentStep === 1 && (
        <StepUpload
          onComplete={(data) => completeStep(1, { pdfData: data })}
          onDataChanged={() => stepDataChanged(1)}
          pdfData={pipelineData.pdfData}
          exampleMode={exampleMode}
          exampleLoading={exampleLoading}
          exampleError={exampleError}
          onLoadExample={loadExample}
          onExitExample={exitExample}
        />
      )}
      {currentStep === 2 && <StepChunking onComplete={(data) => completeStep(2, { chunkData: data })} onDataChanged={() => stepDataChanged(2)} chunkData={pipelineData.chunkData} />}
      {currentStep === 3 && <StepEmbedding onComplete={(data) => completeStep(3, { embedData: data })} onDataChanged={() => stepDataChanged(3)} embedData={pipelineData.embedData} />}
      {currentStep === 4 && <StepLLM onComplete={(data) => completeStep(4, { llmModel: data })} llmModel={pipelineData.llmModel} />}
      {currentStep === 5 && <StepQuery exampleMode={exampleMode} exampleQuestions={exampleQuestions} />}
    </div>
  )
}
