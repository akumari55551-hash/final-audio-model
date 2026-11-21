import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Shell } from './components/Shell'
import { api } from './services/api'
import type { AnalysisViewResult, BackendHealth, BackendStatus } from './types'
import { AudioAnalysis, Assistant, Dashboard, Monitoring, Settings, ThreatHistory } from './pages/Pages'

export default function App() {
  const [status, setStatus] = useState<BackendStatus>('checking')
  const [health, setHealth] = useState<BackendHealth | null>(null)
  const [latestAnalysis, setLatestAnalysis] = useState<AnalysisViewResult | null>(null)
  useEffect(() => { let active = true; void api.health().then((result) => { if (active) { setHealth(result); setStatus(result.status) } }).catch(() => { if (active) { setHealth(null); setStatus('unavailable') } }); return () => { active = false } }, [])
  const inferenceVerified = latestAnalysis?.detection.status === 'AVAILABLE'
  return <Routes><Route element={<Shell status={status} inferenceVerified={inferenceVerified} />}><Route index element={<Dashboard status={status} modelLoaded={health?.modelLoaded ?? false} />} /><Route path="monitoring" element={<Monitoring />} /><Route path="analysis" element={<AudioAnalysis status={status} modelLoaded={health?.modelLoaded ?? false} modelError={health?.error ?? null} onResult={setLatestAnalysis} />} /><Route path="history" element={<ThreatHistory status={status} />} /><Route path="assistant" element={<Assistant status={status} context={latestAnalysis ?? undefined} />} /><Route path="settings" element={<Settings status={status} inferenceVerified={inferenceVerified} />} /><Route path="*" element={<Navigate to="/" replace />} /></Route></Routes>
}
