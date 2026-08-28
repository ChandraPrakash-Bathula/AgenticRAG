// Central API configuration
// In production, set VITE_API_URL env var in Vercel dashboard
import axios from 'axios'

const rawApiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Auto-sanitize to ensure it ends with /api (without duplicating trailing slashes)
const sanitizedUrl = rawApiUrl.replace(/\/+$/, '')
export const API = sanitizedUrl.endsWith('/api') ? sanitizedUrl : `${sanitizedUrl}/api`

// Per-tab session ID so the backend scopes pipeline state to this client instead of
// sharing one global session across all users. sessionStorage = one session per tab.
const SESSION_KEY = 'ragStudio.sessionId'
let sessionId = sessionStorage.getItem(SESSION_KEY)
if (!sessionId) {
  sessionId = (crypto.randomUUID && crypto.randomUUID()) || `${Date.now()}-${Math.random().toString(36).slice(2)}`
  sessionStorage.setItem(SESSION_KEY, sessionId)
}
axios.defaults.headers.common['X-Session-Id'] = sessionId
export const SESSION_ID = sessionId

