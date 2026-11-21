import { useCallback, useEffect, useRef, useState } from 'react'

export interface AudioDevice { deviceId: string; label: string }
export function useMicrophone() {
  const [devices, setDevices] = useState<AudioDevice[]>([])
  const [deviceId, setDeviceId] = useState('')
  const [active, setActive] = useState(false)
  const [error, setError] = useState('')
  const [seconds, setSeconds] = useState(0)
  const streamRef = useRef<MediaStream | null>(null)
  const contextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef = useRef<number | null>(null)
  const startedRef = useRef(0)

  const stop = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null; analyserRef.current = null
    const context = contextRef.current; contextRef.current = null
    if (context && context.state !== 'closed') void context.close()
    setActive(false)
  }, [])

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return
    const inputs = (await navigator.mediaDevices.enumerateDevices()).filter((item) => item.kind === 'audioinput')
    setDevices(inputs.map((item, index) => ({ deviceId: item.deviceId, label: item.label || `Microphone ${index + 1}` })))
    if (!deviceId && inputs[0]) setDeviceId(inputs[0].deviceId)
  }, [deviceId])

  const start = useCallback(async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) { setError('Microphone capture is unavailable in this browser or context. Use localhost or HTTPS.'); return }
    let stream: MediaStream | null = null
    let context: AudioContext | null = null
    try {
      let preferences: { echoCancellation?: boolean; noiseSuppression?: boolean; autoGainControl?: boolean } = {}
      try { preferences = JSON.parse(localStorage.getItem('audio-preferences') || localStorage.getItem('sonara-preferences') || '{}') as typeof preferences } catch { /* Use browser defaults if preferences are unavailable. */ }
      const audio = { echoCancellation: preferences.echoCancellation ?? false, noiseSuppression: preferences.noiseSuppression ?? false, autoGainControl: preferences.autoGainControl ?? false, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) }
      stream = await navigator.mediaDevices.getUserMedia({ audio })
      context = new AudioContext(); const analyser = context.createAnalyser(); analyser.fftSize = 1024; analyser.smoothingTimeConstant = 0.78
      context.createMediaStreamSource(stream).connect(analyser)
      streamRef.current = stream; contextRef.current = context; analyserRef.current = analyser
      startedRef.current = Date.now(); setSeconds(0); setActive(true)
      await refreshDevices()
    } catch (cause) {
      stream?.getTracks().forEach((track) => track.stop())
      if (context && context.state !== 'closed') void context.close()
      const name = cause instanceof DOMException ? cause.name : ''
      setError(name === 'NotAllowedError' ? 'Microphone permission was denied. Allow access in your browser settings and try again.' : name === 'NotFoundError' ? 'No microphone was found. Connect a microphone and retry.' : 'Could not start microphone capture. Check that your device is available and not in use.')
    }
  }, [deviceId, refreshDevices])

  useEffect(() => {
    if (!active) return
    const tick = () => {
      const canvas = document.querySelector<HTMLCanvasElement>('#mic-waveform')
      const analyser = analyserRef.current
      const ctx = canvas?.getContext('2d')
      if (canvas && analyser && ctx) {
        const data = new Uint8Array(analyser.frequencyBinCount); analyser.getByteTimeDomainData(data)
        const { width, height } = canvas; ctx.clearRect(0, 0, width, height); ctx.lineWidth = 2; ctx.strokeStyle = '#9B5DE5'; ctx.beginPath()
        data.forEach((value, index) => { const x = (index / (data.length - 1)) * width; const y = (value / 255) * height; if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y) })
        ctx.stroke()
      }
      setSeconds(Math.floor((Date.now() - startedRef.current) / 1000))
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => { if (rafRef.current !== null) cancelAnimationFrame(rafRef.current) }
  }, [active])

  useEffect(() => () => { streamRef.current?.getTracks().forEach((track) => track.stop()); const context = contextRef.current; if (context && context.state !== 'closed') void context.close() }, [])
  return { devices, deviceId, setDeviceId, active, error, seconds, start, stop, refreshDevices }
}
