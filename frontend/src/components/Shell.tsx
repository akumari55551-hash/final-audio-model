import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Activity, AudioLines, Bell, ChevronRight, Command, LayoutDashboard, Menu, MessageCircle, Radio, Settings2, Shield, X } from 'lucide-react'
import type { BackendStatus } from '../types'

const navigation = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/monitoring', label: 'Live monitoring', icon: Radio },
  { to: '/analysis', label: 'Audio analysis', icon: AudioLines },
  { to: '/history', label: 'Threat history', icon: Shield },
  { to: '/assistant', label: 'A.U.D.I.O Assistant', icon: MessageCircle },
]
const titles: Record<string, string> = { '/': 'Dashboard', '/monitoring': 'Live monitoring', '/analysis': 'Audio analysis', '/history': 'Threat history', '/assistant': 'A.U.D.I.O Assistant', '/settings': 'Settings' }

export function Shell({ status, inferenceVerified = false }: { status: BackendStatus; inferenceVerified?: boolean }) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const name = titles[location.pathname] ?? 'Dashboard'
  const reduceMotion = useReducedMotion() || document.documentElement.dataset.motion === 'reduced'
  const sidebar = <>
    <div className="brand-lockup"><div className="brand-mark"><Activity size={19} strokeWidth={2.2} /></div><div className="brand-copy"><strong>A.U.D.I.O</strong><small className="brand-expansion">AI-Driven Uncertainty and Detection for Impersonation Overwatch</small><small className="brand-platform-title">Federated Voice Threat Intelligence Platform</small></div>{mobileOpen && <button className="icon-button mobile-close" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={18} /></button>}</div>
    <div className="workspace-label">WORKSPACE</div>
    <nav aria-label="Main navigation" className="side-nav">{navigation.map(({ to, label, icon: Icon, end }) => <NavLink key={to} to={to} end={end} onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}><Icon size={18} strokeWidth={1.8} /><span>{label}</span>{label === 'Live monitoring' && <i className="nav-live-dot" />}</NavLink>)}</nav>
    <div className="sidebar-bottom"><div className="sidebar-status"><span className={`status-dot ${inferenceVerified ? 'connected' : status}`} /><span className="status-copy"><b>Model endpoint</b><small>{inferenceVerified ? 'Inference verified' : status === 'connected' ? 'Health OK · inference unverified' : status === 'checking' ? 'Checking health' : 'Unavailable'}</small></span><span className="status-pulse" /></div><NavLink to="/settings" onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link settings-link${isActive ? ' active' : ''}`}><Settings2 size={18} /><span>Settings</span></NavLink><div className="sidebar-foot"><span className="tiny-logo">A</span><span>A.U.D.I.O <small>· PRIVATE WORKSPACE</small></span></div></div>
  </>
  return <div className="app-shell">
    <aside className={`sidebar${mobileOpen ? ' mobile-open' : ''}`}>{sidebar}</aside>
    {mobileOpen && <button className="mobile-scrim" aria-label="Close menu" onClick={() => setMobileOpen(false)} />}
    <div className="main-column"><header className="topbar"><div className="topbar-left"><button className="icon-button menu-trigger" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={19} /></button><div className="breadcrumb"><span>A.U.D.I.O</span><span className="slash">/</span><b>{name}</b></div></div><div className="topbar-actions"><div className="secure-badge"><span className="secure-dot" /> LOCAL SESSION</div><button className="icon-button notification-button" aria-label="Notifications"><Bell size={18} /><i /></button><span className="topbar-divider" /><NavLink to="/settings" className="profile-button" aria-label="Open settings"><span className="profile-avatar">A</span><span className="profile-name">Analyst</span><ChevronRight className="profile-chevron" size={14} /></NavLink></div></header><main className="page-wrap"><AnimatePresence mode="wait" initial={false}><motion.div key={location.pathname} className="page-enter" initial={reduceMotion ? false : { opacity: 0, y: 10, filter: 'blur(4px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }} exit={reduceMotion ? undefined : { opacity: 0, y: -6 }} transition={{ duration: reduceMotion ? 0 : 0.24, ease: 'easeOut' }}><Outlet /></motion.div></AnimatePresence><footer className="page-footer"><span><Command size={12} /> A.U.D.I.O · FEDERATED VOICE THREAT INTELLIGENCE PLATFORM</span><span>DETECTION SIGNALS REQUIRE HUMAN REVIEW</span></footer></main></div>
  </div>
}
