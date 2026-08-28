import { useState } from 'react'
import './App.css'
import Home from './components/Home'
import StepWizard from './components/StepWizard'

function App() {
  // Two views rather than a router: the wizard holds live pipeline state in the backend
  // session, so unmounting it on navigation would strand that state. Keeping this to a
  // single boolean means "back to overview" never costs the user their progress.
  const [view, setView] = useState('home')

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo-container">
          <button
            type="button"
            className="logo-home"
            onClick={() => setView('home')}
            aria-label="Back to the overview"
          >
            <span className="logo-icon">
              <svg viewBox="0 0 40 40" fill="none" aria-hidden="true">
                <rect x="2" y="2" width="36" height="36" rx="8" stroke="currentColor" strokeWidth="2.5"/>
                <path d="M12 28V16l8-6 8 6v12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="20" cy="20" r="4" fill="currentColor"/>
              </svg>
            </span>
            <span className="logo-text">
              <span className="app-title">Naive RAG vs Agentic RAG</span>
              <span className="app-subtitle">From naive RAG to self-correcting agentic RAG, interactive playground.</span>
            </span>
          </button>

          <nav className="app-nav" aria-label="Sections">
            <button
              type="button"
              className={`app-nav-link ${view === 'home' ? 'active' : ''}`}
              aria-current={view === 'home' ? 'page' : undefined}
              onClick={() => setView('home')}
            >
              Overview
            </button>
            <button
              type="button"
              className={`app-nav-link ${view === 'wizard' ? 'active' : ''}`}
              aria-current={view === 'wizard' ? 'page' : undefined}
              onClick={() => setView('wizard')}
            >
              Playground
            </button>
          </nav>
        </div>
      </header>

      <main className="app-main">
        {view === 'home'
          ? <Home onStart={() => setView('wizard')} />
          : <StepWizard />}
      </main>
    </div>
  )
}

export default App
