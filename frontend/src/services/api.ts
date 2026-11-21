import type { AnalysisRecord, AnalysisViewResult, BackendHealth, ChatResponse } from '../types'

const baseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''
const routes = { health: '/api/health', history: '/api/v1/analyses', analyze: '/api/analyze', chat: '/api/v1/chat' } as const

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!baseUrl) throw new Error('Backend URL is not configured. Set VITE_API_BASE_URL in frontend/.env.local.')
  let response: Response
  try { response = await fetch(`${baseUrl}${path}`, init) }
  catch { throw new Error('The A.U.D.I.O backend could not be reached.') }
  if (!response.ok) {
    let detail = `Backend request failed (${response.status}).`
    try { const body = await response.json() as { detail?: string }; if (body.detail) detail = body.detail } catch { /* Keep the status message. */ }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  async health(): Promise<BackendHealth> {
    const body = await request<{ status: string; model_loaded: boolean; device?: string | null; model?: BackendHealth['model']; error?: string | null }>(routes.health)
    return { status: body.status === 'online' ? 'connected' : 'unavailable', modelLoaded: body.model_loaded, device: body.device, model: body.model, error: body.error }
  },
  async history(): Promise<AnalysisRecord[]> {
    const body = await request<{ items: AnalysisRecord[] } | AnalysisRecord[]>(routes.history)
    return Array.isArray(body) ? body : body.items
  },
  analyze(file: File, onProgress?: (percent: number) => void): Promise<unknown> {
    if (!baseUrl) return Promise.reject(new Error('Backend URL is not configured. Set VITE_API_BASE_URL in frontend/.env.local.'))
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${baseUrl}${routes.analyze}`)
      xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(Math.round(event.loaded / event.total * 100)) }
      xhr.onerror = () => reject(new Error('The A.U.D.I.O backend could not be reached.'))
      xhr.onload = () => {
        let body: unknown
        try { body = JSON.parse(xhr.responseText) } catch { body = null }
        if (xhr.status < 200 || xhr.status >= 300) {
          const detail = body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string' ? body.detail : `Backend request failed (${xhr.status}).`
          reject(new Error(detail)); return
        }
        resolve(body)
      }
      const form = new FormData(); form.append('file', file, file.name); xhr.send(form)
    })
  },
  chat(message: string, context?: AnalysisViewResult): Promise<ChatResponse> {
    return request<ChatResponse>(routes.chat, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, context }) })
  },
}

export const backendRoutes = routes
