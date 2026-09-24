const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function queryHealRAG(question) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  })
  if (!res.ok) throw new Error(`API error: ${res.statusText}`)
  return res.json()
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(`API error: ${res.statusText}`)
  return res.json()
}

export async function getEvidence(limit = 30) {
  const res = await fetch(`${API_BASE}/evidence?limit=${limit}`)
  if (!res.ok) throw new Error(`API error: ${res.statusText}`)
  return res.json()
}

export async function getEvaluation() {
  const res = await fetch(`${API_BASE}/evaluation`)
  if (!res.ok) throw new Error(`API error: ${res.statusText}`)
  return res.json()
}

export async function runEvaluation() {
  const res = await fetch(`${API_BASE}/evaluate`, { method: 'POST' })
  if (!res.ok) throw new Error(`API error: ${res.statusText}`)
  return res.json()
}

export async function triggerIngest() {
  const res = await fetch(`${API_BASE}/ingest`, { method: 'POST' })
  if (!res.ok) throw new Error(`API error: ${res.statusText}`)
  return res.json()
}

export async function triggerRecovery() {
  const res = await fetch(`${API_BASE}/recover`, { method: 'POST' })
  if (!res.ok) throw new Error(`API error: ${res.statusText}`)
  return res.json()
}
