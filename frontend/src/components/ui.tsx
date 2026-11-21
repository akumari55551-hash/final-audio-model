import { useEffect, useState, type ReactNode } from 'react'
import { ArrowDownRight, ArrowUpRight, Check, CircleHelp, Clock3, LockKeyhole, ShieldAlert, Sparkles } from 'lucide-react'

export function Eyebrow({ children, dot = false }: { children: ReactNode; dot?: boolean }) { return <div className="eyebrow">{dot && <i className="eyebrow-dot" />}{children}</div> }
export function SectionTitle({ title, detail, action }: { title: string; detail?: string; action?: ReactNode }) { return <div className="section-title"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{action}</div> }
export function Card({ children, className = '' }: { children: ReactNode; className?: string }) { return <section className={`panel ${className}`}>{children}</section> }
export function StatusPill({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'good' | 'warning' | 'bad' | 'violet' }) { return <span className={`pill ${tone}`}><i />{children}</span> }
export function EmptyState({ icon = 'clock', title, text }: { icon?: 'clock' | 'shield' | 'sparkles'; title: string; text: string }) {
  const Icon = icon === 'shield' ? ShieldAlert : icon === 'sparkles' ? Sparkles : Clock3
  return <div className="empty-state"><span className="empty-icon"><Icon size={19} /></span><strong>{title}</strong><p>{text}</p></div>
}
export function MetricCard({ label, value, hint, type }: { label: string; value: string; hint: string; type: 'all' | 'human' | 'spoof' }) {
  const Icon = type === 'all' ? CircleHelp : type === 'human' ? Check : ShieldAlert
  const IconTrend = type === 'spoof' ? ArrowUpRight : ArrowDownRight
  const [animated, setAnimated] = useState(value)
  useEffect(() => {
    const target = Number(value)
    if (!Number.isFinite(target)) { setAnimated(value); return }
    const duration = 560
    let frame = 0
    let start = 0
    const tick = (now: number) => {
      if (!start) start = now
      const progress = Math.min(1, (now - start) / duration)
      setAnimated(String(Math.round(target * (1 - (1 - progress) ** 3))))
      if (progress < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [value])
  return <Card className={`metric-card metric-${type}`}><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon size={16} /></span></div><div className="metric-value">{animated}</div><div className="metric-foot"><span>{hint}</span><IconTrend size={14} /></div></Card>
}
export function TrustNote({ children }: { children: ReactNode }) { return <div className="trust-note"><LockKeyhole size={14} /><span>{children}</span></div> }
