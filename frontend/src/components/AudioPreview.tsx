import { useEffect, useRef, useState } from 'react'

export function AudioPreview({ file }: { file: File }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let cancelled = false
    const draw = async () => {
      try {
        const context = new AudioContext()
        try {
          const buffer = await context.decodeAudioData(await file.arrayBuffer())
          if (cancelled) return
          const canvas = canvasRef.current; const ctx = canvas?.getContext('2d')
          if (!canvas || !ctx) return
          const channel = buffer.getChannelData(0); const count = canvas.width; const stride = Math.ceil(channel.length / count)
          ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.strokeStyle = '#9B5DE5'; ctx.lineWidth = 1.5; ctx.beginPath()
          for (let x = 0; x < count; x += 1) {
            let low = 1; let high = -1
            for (let i = x * stride; i < Math.min((x + 1) * stride, channel.length); i += 1) { low = Math.min(low, channel[i]); high = Math.max(high, channel[i]) }
            const center = canvas.height / 2; ctx.moveTo(x, center + low * center * 0.86); ctx.lineTo(x, center + high * center * 0.86)
          }
          ctx.stroke()
        } finally { await context.close() }
      } catch { if (!cancelled) setError('Waveform preview is not supported for this encoding. The audio file is still available to play.') }
    }
    void draw()
    return () => { cancelled = true }
  }, [file])
  return <div className="file-wave"><canvas ref={canvasRef} width={800} height={80} aria-label="Waveform decoded from selected audio" />{error && <small>{error}</small>}</div>
}
