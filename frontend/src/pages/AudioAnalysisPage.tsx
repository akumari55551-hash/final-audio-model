import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { Activity, CircleAlert, FileAudio2, Upload, X } from 'lucide-react'
import { AudioPreview } from '../components/AudioPreview'
import { api } from '../services/api'
import { createAnalysisView, createOfflineAnalysis } from '../services/analysisSemantics'
import { getAudioFormat, MAX_AUDIO_FILE_BYTES } from '../services/audioDecoder'
import { LFCC_TARGET_SAMPLE_RATE_HZ, validateAudioFile } from '../services/audioPreflight'
import type { AnalysisViewResult, AudioValidation, BackendStatus } from '../types'

const validAudio = (file: File) => Boolean(getAudioFormat(file)) && file.size > 0 && file.size <= MAX_AUDIO_FILE_BYTES
const formatBytes = (bytes: number) => bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`

type Props = { status: BackendStatus; modelLoaded: boolean; modelError: string | null; onResult: (result: AnalysisViewResult | null) => void }

export default function AudioAnalysisPage({ status, modelLoaded, modelError, onResult }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [validation, setValidation] = useState<AudioValidation | null>(null)
  const [preprocessedFile, setPreprocessedFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<AnalysisViewResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [verified, setVerified] = useState(false)
  const [attempted, setAttempted] = useState(false)
  const [previewUrl, setPreviewUrl] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const objectUrl = useRef('')

  useEffect(() => () => { if (objectUrl.current) URL.revokeObjectURL(objectUrl.current) }, [])
  useEffect(() => {
    if (!preprocessedFile) { setPreviewUrl(''); return }
    const url = URL.createObjectURL(preprocessedFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [preprocessedFile])
  useEffect(() => {
    if (!file) { setValidation(null); return }
    let active = true
    setValidation(null)
    void validateAudioFile(file).then(({ validation: next, preprocessedFile: converted }) => {
      if (active) { setValidation(next); setPreprocessedFile(converted) }
    }).catch((cause: unknown) => {
      if (active) setError(cause instanceof Error ? cause.message : 'Audio decoding failed.')
    })
    return () => { active = false }
  }, [file])

  const selectFile = (candidate?: File) => {
    if (!candidate) return
    setError(''); setResult(null); setValidation(null); setPreprocessedFile(null); setVerified(false); setAttempted(false); onResult(null)
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = ''
    if (!validAudio(candidate)) {
      setFile(null)
      setError(candidate.size === 0 ? 'The selected audio file is empty.' : candidate.size > MAX_AUDIO_FILE_BYTES ? 'This file exceeds the 50 MB limit.' : 'Choose an M4A, WAV, FLAC, MP3, OGG, or AIFF audio file.')
      return
    }
    objectUrl.current = URL.createObjectURL(candidate)
    setFile(candidate)
  }
  const onDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]) }
  const onPick = (event: ChangeEvent<HTMLInputElement>) => { selectFile(event.target.files?.[0]); event.target.value = '' }
  const clearFile = () => {
    setFile(null); setValidation(null); setPreprocessedFile(null); setResult(null); setError(''); setVerified(false); setAttempted(false); onResult(null)
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = ''
  }
  const runInference = async () => {
    if (!file || !validation || !preprocessedFile || validation.status !== 'PASS') return
    setBusy(true); setError(''); setResult(null); setAttempted(true); setVerified(false); setUploadProgress(0)
    try {
      // Always upload the original recording; the backend performs its own canonical conversion.
      const view = createAnalysisView(await api.analyze(file, setUploadProgress), validation)
      setResult(view); setVerified(view.detection.status === 'AVAILABLE'); onResult(view)
      if (view.detection.status !== 'AVAILABLE') setError('The analysis endpoint returned no model inference result.')
    } catch (cause) {
      setResult(createOfflineAnalysis(validation)); onResult(null)
      setError(cause instanceof Error ? cause.message : 'Inference request could not be completed.')
    } finally { setBusy(false); setUploadProgress(null) }
  }

  const detection = result?.detection.status === 'AVAILABLE' ? result.detection : null
  const spoofScore = detection?.spoofScore
  const genuineScore = detection?.bonaFideScore
  const classification = detection ? detection.authenticity === 'BONA_FIDE' ? 'LIKELY GENUINE' : detection.authenticity === 'SPOOF' ? 'LIKELY AI' : 'INCONCLUSIVE' : result ? 'NOT ANALYSED' : '—'
  const decision = detection ? detection.authenticity === 'BONA_FIDE' ? 'PASS' : detection.authenticity === 'SPOOF' ? 'FAIL' : 'REVIEW' : 'REVIEW'
  const connection = verified ? 'INFERENCE VERIFIED' : attempted && !busy ? 'INFERENCE FAILED' : status === 'checking' ? 'CHECKING MODEL' : status === 'connected' && modelLoaded ? 'MODEL HEALTHY · INFERENCE UNVERIFIED' : 'MODEL UNAVAILABLE'

  return <section className="detection-page">
    <header className="detection-header">
      <div><p className="detection-kicker">A.U.D.I.O</p><h1>Voice detection</h1><p className="detection-platform">Federated Voice Threat Intelligence Platform</p></div>
      <div className={`model-connection${verified ? ' verified' : ''}`} role="status"><span className={`status-dot ${verified ? 'connected' : status}`} />{connection}</div>
    </header>

    <div className="detection-grid">
      <section className="detection-panel upload-panel" aria-labelledby="upload-title">
        <div className="panel-heading"><div><p className="detection-kicker">AUDIO INPUT</p><h2 id="upload-title">Upload a recording</h2></div></div>
        {!file ? <div className={`detection-dropzone${dragging ? ' dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop} onClick={() => inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click() }}>
          <input ref={inputRef} type="file" accept=".m4a,.wav,.flac,.mp3,.ogg,.aiff,.aif,audio/mp4,audio/wav,audio/flac,audio/mpeg,audio/ogg,audio/aiff" onChange={onPick} hidden />
          <span className="detection-upload-icon"><Upload size={20} /></span><strong>Choose or drop an audio file</strong><span>M4A · WAV · FLAC · MP3 · OGG · AIFF · up to 50 MB</span>
        </div> : <div className="detection-selected"><div className="detection-file-row"><span className="detection-file-icon"><FileAudio2 size={19} /></span><div className="detection-file-meta"><strong title={file.name}>{file.name}</strong><span>{getAudioFormat(file)} · {formatBytes(file.size)}</span></div><button className="icon-button" aria-label="Remove recording" onClick={clearFile}><X size={16} /></button></div><AudioPreview file={preprocessedFile ?? file} /><audio src={previewUrl || objectUrl.current} preload="metadata" controls className="native-audio" /></div>}
        {error && <p className="detection-error" role="alert"><CircleAlert size={16} />{error}</p>}
        {modelError && status === 'connected' && !modelLoaded && <p className="detection-error" role="status">{modelError}</p>}
        {file && <button className="button button-primary detection-run" disabled={!validation || !preprocessedFile || validation.status !== 'PASS' || busy} onClick={() => void runInference()}>
          {busy ? <><Activity size={16} className="detection-spin" />{uploadProgress !== null && uploadProgress < 100 ? `Uploading · ${uploadProgress}%` : 'Running model inference…'}</> : 'Run Detection'}
        </button>}
      </section>

      <section className="detection-panel result-panel" aria-labelledby="result-title" aria-live="polite">
        <div className="panel-heading"><div><p className="detection-kicker">MODEL RESULT</p><h2 id="result-title">Detection summary</h2></div></div>
        {busy ? <div className="detection-loading"><span className="progress-spinner" /><p>Uploading the original recording and waiting for model inference…</p></div> : <>
          <div className="detection-score-grid">
            <Score label="AI Voice Risk" value={spoofScore} />
            <Score label="Genuine Voice Probability" value={genuineScore} />
            <Score label="AI Voice Probability" value={spoofScore} />
          </div>
          <div className="detection-conclusion">
            <div><span>Classification</span><strong className={detection ? `class-${detection.authenticity.toLowerCase()}` : ''}>{classification}</strong></div>
            <div><span>Decision</span><strong className={`decision-${decision.toLowerCase()}`}>{decision}</strong></div>
          </div>
          <p className="detection-honesty">Model scores are estimates, not necessarily calibrated probabilities of fraud. A result is a screening signal, not proof of identity.</p>
        </>}
      </section>
    </div>

    <details className="detection-diagnostics">
      <summary>Advanced Diagnostics</summary>
      {validation ? <div className="diagnostic-grid">
        <span>Audio validation</span><b>{validation.status}</b><span>Original format</span><b>{validation.originalFormat ?? 'Unknown'}</b>
        <span>Original duration</span><b>{validation.durationSeconds === null ? 'Unavailable' : `${validation.durationSeconds.toFixed(2)} s`}</b>
        <span>Original sample rate</span><b>{validation.sampleRateHz ? `${validation.sampleRateHz.toLocaleString()} Hz` : 'Unavailable'}</b>
        <span>Channels</span><b>{validation.channelCount ?? 'Unavailable'}</b><span>SHA-256 (original bytes)</span><code>{validation.sha256 ?? 'Unavailable'}</code>
        <span>LFCC compatibility</span><b>{validation.filterbankCompatibility}</b><span>Prepared audio</span><b>Mono · {LFCC_TARGET_SAMPLE_RATE_HZ.toLocaleString()} Hz · PCM WAV</b>
        <span>Inference status</span><b>{detection ? 'AVAILABLE' : attempted ? 'UNAVAILABLE' : 'NOT RUN'}</b>
        {detection && <><span>Model version</span><b>{detection.modelVersion}</b><span>Processing time</span><b>{detection.processingTimeMs?.toFixed(1)} ms</b><span>Class mapping</span><code>{JSON.stringify(detection.provenance?.classMapping)}</code><span>Checkpoint SHA-256</span><code>{detection.provenance?.checkpointSha256}</code><span>Request ID</span><code>{detection.provenance?.inferenceId}</code><span>Device</span><b>{detection.provenance?.device ?? 'Not reported'}</b><span>Score aggregation</span><b>{detection.provenance?.scoreAggregation ?? 'Not reported'}</b></>}
        {validation.message && <><span>Validation detail</span><b>{validation.message}</b></>}
      </div> : <p className="diagnostic-empty">Select a file to inspect validation and preprocessing diagnostics.</p>}
      {detection?.windowPredictions?.length ? <div className="window-diagnostics"><strong>Window-level logits and probabilities</strong><div className="window-diagnostics-head"><span>Window</span><span>Logits [bona fide, spoof]</span><span>Softmax scores [bona fide, spoof]</span><span>Class</span></div>{detection.windowPredictions.map((window) => <div className="window-diagnostics-row" key={window.windowIndex}><span>{window.windowIndex + 1}</span><code>[{window.logits.map((score) => score.toFixed(4)).join(', ')}]</code><code>[{window.bonaFideScore.toFixed(4)}, {window.spoofScore.toFixed(4)}]</code><b>{window.predictedClass}</b></div>)}</div> : null}
    </details>
  </section>
}

function Score({ label, value }: { label: string; value?: number }) {
  return <div className="detection-score"><span>{label}</span><strong>{typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—'}</strong></div>
}
