import { afterEach, describe, expect, it, vi } from 'vitest'

class MockXhr {
  static instance: MockXhr
  upload: { onprogress: ((event: ProgressEvent) => void) | null } = { onprogress: null }
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  status = 200
  responseText = ''
  sentBody: FormData | null = null
  constructor() { MockXhr.instance = this }
  open() {}
  send(body: FormData) { this.sentBody = body }
  complete(body: unknown, status = 200) {
    this.status = status
    this.responseText = JSON.stringify(body)
    this.onload?.()
  }
}

async function loadApi() {
  vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8000')
  vi.stubGlobal('XMLHttpRequest', MockXhr)
  vi.resetModules()
  return import('./api')
}

afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('A.U.D.I.O API integration', () => {
  it('uploads the original audio and reports progress with the actual response', async () => {
    const { api } = await loadApi()
    const original = new File(['recording bytes'], 'voice.wav', { type: 'audio/wav' })
    const progress: number[] = []
    const promise = api.analyze(original, (value) => progress.push(value))
    MockXhr.instance.upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 100 } as ProgressEvent)
    MockXhr.instance.complete({ detection: { status: 'AVAILABLE', authenticity: 'SPOOF', spoof_score: 0.8, bona_fide_score: 0.2, provenance: { model_name: 'LFCCCNNDetector', inference_id: 'test-run' } } })
    const response = await promise
    const sentFile = MockXhr.instance.sentBody?.get('file') as File
    expect(sentFile.name).toBe(original.name)
    expect(await sentFile.text()).toBe(await original.text())
    expect(progress).toEqual([50])
    expect(response).toMatchObject({ detection: { authenticity: 'SPOOF', spoof_score: 0.8 } })
  })

  it('reports the backend online separately when the model is unavailable', async () => {
    const { api } = await loadApi()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'online', model_loaded: false, error: 'Model unavailable' }), { status: 200 })))
    await expect(api.health()).resolves.toMatchObject({ status: 'connected', modelLoaded: false, error: 'Model unavailable' })
  })

  it('surfaces a backend-offline error without inventing a result', async () => {
    const { api } = await loadApi()
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    await expect(api.health()).rejects.toThrow('The A.U.D.I.O backend could not be reached.')
  })
})
