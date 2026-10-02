import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  AlertTriangle, ArrowRight, BarChart3, Bell, CheckCircle2, ChevronDown,
  CircleHelp, ClipboardCheck, Gauge, Home, Info, Landmark, Leaf, Menu,
  Printer, Search, ShieldCheck, Sparkles, TrendingUp, Users, X, Zap,
} from 'lucide-react'
import {
  Area, AreaChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import './styles.css'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const STORAGE_KEY = 'drishti-assessments'
const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Home },
  { id: 'assessment', label: 'New assessment', icon: ClipboardCheck },
  { id: 'about', label: 'About DRISHTI', icon: CircleHelp },
]
const palette = { High: '#ef6b70', Medium: '#f4b860', Low: '#48c3b3' }
const initialForm = {
  project_type: '', state: '', land_area_hectares: '', total_parcels: '',
  private_land_pct: '', forest_land_pct: '', govt_land_pct: '',
  affected_families_count: '', rr_settlement_status: '',
  forest_clearance_stage: '', env_clearance_stage: '', circle_rate_per_sqm: '',
  compensation_disbursed_pct: '', cadastral_survey_discrepancy: '',
  active_litigations_count: '', stay_order_present: '', inter_agency_nocs_pending: '',
  historical_district_delay_score: '',
}
const selectOptions = {
  project_type: ['Industrial Corridor', 'Highway', 'Railway', 'Irrigation', 'Urban Development', 'Renewable Energy'],
  state: ['Andhra Pradesh', 'Maharashtra', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'West Bengal'],
  rr_settlement_status: ['Not started', 'In progress', 'Mostly Completed', 'Completed', 'Not applicable'],
  forest_clearance_stage: ['Not initiated', 'Proposal submitted', 'Stage-1 Approved', 'Stage-2 Approved', 'Not applicable'],
  env_clearance_stage: ['Not initiated', 'Under review', 'EC Granted', 'Not applicable'],
  stay_order_present: ['0', '1'],
}
const fieldMeta = [
  ['project_type', 'Project type', 'select', 'Project classification', true],
  ['state', 'State', 'select', 'Project state', true],
  ['land_area_hectares', 'Land area', 'number', 'Hectares', true],
  ['total_parcels', 'Total land parcels', 'number', 'Parcel count', true],
  ['private_land_pct', 'Private land', 'number', '% of total land', true],
  ['forest_land_pct', 'Forest land', 'number', '% of total land', true],
  ['govt_land_pct', 'Government land', 'number', '% of total land', true],
  ['affected_families_count', 'Affected families', 'number', 'Family count', true],
  ['rr_settlement_status', 'R&R settlement status', 'select', 'Rehabilitation and resettlement', true],
  ['forest_clearance_stage', 'Forest clearance', 'select', 'Current stage', true],
  ['env_clearance_stage', 'Environmental clearance', 'select', 'Current stage', true],
  ['circle_rate_per_sqm', 'Circle rate', 'number', 'INR / m²', true],
  ['compensation_disbursed_pct', 'Compensation disbursed', 'number', '% complete', true],
  ['cadastral_survey_discrepancy', 'Cadastral discrepancy', 'number', 'Count', true],
  ['active_litigations_count', 'Active litigations', 'number', 'Matter count', true],
  ['stay_order_present', 'Stay order present', 'select', '0 = No, 1 = Yes', true],
  ['inter_agency_nocs_pending', 'Pending inter-agency NOCs', 'number', 'Count', true],
  ['historical_district_delay_score', 'Historical district delay score', 'number', '0–100 score', true],
]

const loadHistory = () => {
  try { return JSON.parse(sessionStorage.getItem(STORAGE_KEY)) || [] } catch { return [] }
}

function App() {
  const [page, setPage] = useState('dashboard')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [history, setHistory] = useState(loadHistory)
  const [result, setResult] = useState(null)
  const [apiError, setApiError] = useState('')
  useEffect(() => sessionStorage.setItem(STORAGE_KEY, JSON.stringify(history)), [history])
  const counts = useMemo(() => ['High', 'Medium', 'Low'].reduce((acc, risk) => {
    acc[risk] = history.filter(item => item.delay_risk_level === risk).length
    return acc
  }, {}), [history])
  const go = id => { setPage(id); setMobileOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  const handleSubmit = async values => {
    setApiError('')
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.message || 'The prediction could not be completed.')
      const assessment = { ...body, assessed_at: new Date().toISOString(), input: values }
      setHistory(current => [assessment, ...current].slice(0, 20))
      setResult(assessment)
      setPage('results')
    } catch (error) {
      setApiError(error.message || 'Unable to reach the DRISHTI inference service.')
    }
  }
  return <div className="app-shell">
    <Sidebar page={page} go={go} mobileOpen={mobileOpen} close={() => setMobileOpen(false)} />
    <div className="main-shell">
      <Header onMenu={() => setMobileOpen(true)} onSearch={() => go('dashboard')} />
      <main className="page-wrap">
        {page === 'dashboard' && <Dashboard records={history} counts={counts} go={go} />}
        {page === 'assessment' && <Assessment onSubmit={handleSubmit} onCancel={() => go('dashboard')} error={apiError} />}
        {page === 'results' && <Results result={result} go={go} />}
        {page === 'about' && <About />}
      </main>
    </div>
  </div>
}

function Sidebar({ page, go, mobileOpen, close }) {
  return <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
    <div className="brand-row"><div className="brand-mark"><Landmark size={21} /></div><div><div className="brand-name">DRISHTI</div><div className="brand-sub">Risk intelligence</div></div><button className="icon-button mobile-close" onClick={close}><X size={18} /></button></div>
    <div className="demo-badge"><span className="pulse-dot" /> MODEL CONNECTED</div>
    <nav className="nav-list"><div className="nav-label">WORKSPACE</div>{navItems.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => go(id)} className={`nav-item ${page === id || (page === 'results' && id === 'assessment') ? 'active' : ''}`}><Icon size={18} /><span>{label}</span>{id === 'assessment' && <span className="nav-plus">+</span>}</button>)}</nav>
    <div className="sidebar-bottom"><div className="security-card"><ShieldCheck size={18} /><div><strong>Private workspace</strong><span>Session-only result history</span></div></div><div className="sidebar-footer"><div className="avatar">AS</div><div><strong>Analyst session</strong><span>Connected to API</span></div><ChevronDown size={14} /></div></div>
  </aside>
}

function Header({ onMenu, onSearch }) {
  return <header className="topbar"><button className="icon-button menu-button" onClick={onMenu}><Menu size={21} /></button><div className="crumbs"><span>Workspace</span><span>/</span><strong>DRISHTI Intelligence</strong></div><div className="top-actions"><button className="search-control" onClick={onSearch}><Search size={16} /><span>Search projects</span><kbd>⌘ K</kbd></button><button className="icon-button notification"><Bell size={18} /><i /></button><div className="top-avatar">AS</div></div></header>
}

function Dashboard({ records, counts, go }) {
  const total = records.length
  const chart = ['High', 'Medium', 'Low'].map(name => ({ name, value: counts[name], color: palette[name] }))
  return <div className="animate-in"><div className="page-heading"><div><div className="eyebrow"><span className="live-dot" /> OPERATIONAL OVERVIEW</div><h1>Good morning, analyst.</h1><p>Monitor acquisition risk across your infrastructure portfolio.</p></div><button className="primary-button" onClick={() => go('assessment')}><Sparkles size={17} /> New assessment <ArrowRight size={16} /></button></div>
    <div className="demo-callout"><Info size={17} /><span><strong>LIVE MODEL RESULTS</strong> — Results are returned by the configured DRISHTI API. The current model was trained on synthetic data.</span><button onClick={() => go('about')}>Learn more</button></div>
    <section className="metric-grid"><MetricCard label="Session assessments" value={total} icon={ClipboardCheck} detail="Current browser session" accent="cyan" /><MetricCard label="High risk projects" value={counts.High} icon={AlertTriangle} detail={total ? `${Math.round(counts.High / total * 100)}% of portfolio` : 'No assessments yet'} accent="red" /><MetricCard label="Medium risk projects" value={counts.Medium} icon={Gauge} detail={total ? `${Math.round(counts.Medium / total * 100)}% of portfolio` : 'No assessments yet'} accent="amber" /><MetricCard label="Low risk projects" value={counts.Low} icon={CheckCircle2} detail={total ? `${Math.round(counts.Low / total * 100)}% of portfolio` : 'No assessments yet'} accent="green" /></section>
    <div className="dashboard-grid"><section className="panel chart-panel"><div className="panel-head"><div><h2>Risk distribution</h2><p>Current assessment portfolio</p></div><button className="ghost-button"><span className="status-dot" /> API session</button></div><div className="chart-wrap">{total ? <><ResponsiveContainer width="52%" height={245}><PieChart><Pie data={chart} innerRadius={75} outerRadius={103} paddingAngle={4} dataKey="value" stroke="none">{chart.map(entry => <Cell key={entry.name} fill={entry.color} />)}</Pie><Tooltip content={<ChartTooltip />} /></PieChart></ResponsiveContainer><div className="legend-list">{chart.map(item => <div className="legend-row" key={item.name}><span className="legend-label"><i style={{ background: item.color }} />{item.name}</span><strong>{item.value}</strong><span className="legend-percent">{Math.round(item.value / total * 100)}%</span></div>)}</div></> : <EmptyChart go={go} />}</div></section>
      <section className="panel signal-panel"><div className="panel-head"><div><h2>Risk outlook</h2><p>Latest regression score</p></div><TrendingUp size={19} className="muted-icon" /></div><div className="signal-value"><span>Latest score</span><strong>{records[0] ? Math.round(records[0].risk_probability * 100) : '—'}</strong><small>/ 100</small></div><div className="mini-chart"><ResponsiveContainer width="100%" height={100}><AreaChart data={records.slice(0, 7).reverse().map(item => ({ v: Math.round(item.risk_probability * 100) }))}><Area type="monotone" dataKey="v" stroke="#28bfd0" fill="none" strokeWidth={2} dot={false} /><XAxis hide dataKey="v" /><YAxis hide domain={[0, 100]} /></AreaChart></ResponsiveContainer></div><div className="signal-note"><span className="arrow-up">↗</span><span><strong>Use model context</strong><br />Review factors before acting.</span></div></section></div>
    <section className="panel recent-panel"><div className="panel-head"><div><h2>Recent assessments</h2><p>Latest projects evaluated in this session</p></div><button className="text-button" onClick={() => go('assessment')}>New assessment <ArrowRight size={14} /></button></div>{total ? <div className="table-wrap"><table><thead><tr><th>Risk level</th><th>Model score</th><th>Model version</th><th>Assessed</th></tr></thead><tbody>{records.slice(0, 8).map((record, index) => <tr key={`${record.assessed_at}-${index}`}><td><span className={`risk-pill risk-${record.delay_risk_level.toLowerCase()}`}><i />{record.delay_risk_level}</span></td><td><div className="score-cell"><strong>{Math.round(record.risk_probability * 100)}</strong><div className="score-bar"><span style={{ width: `${record.risk_probability * 100}%`, background: palette[record.delay_risk_level] }} /></div></div></td><td>{record.model_version}</td><td className="muted-text">{new Date(record.assessed_at).toLocaleString()}</td></tr>)}</tbody></table></div> : <div className="empty-state"><ClipboardCheck size={24} /><strong>No assessments yet</strong><span>Connect to the API and run your first model assessment.</span><button className="primary-button" onClick={() => go('assessment')}>Start assessment <ArrowRight size={15} /></button></div>}</section>
  </div>
}

function MetricCard({ label, value, icon: Icon, detail, accent }) { return <div className={`metric-card metric-${accent}`}><div className="metric-top"><div className="metric-icon"><Icon size={18} /></div><span className="metric-kicker">SESSION</span></div><div className="metric-value">{value}</div><div className="metric-label">{label}</div><div className="metric-detail">{detail}</div></div> }
function ChartTooltip({ active, payload }) { return active && payload?.length ? <div className="chart-tooltip">{payload[0].value} assessments</div> : null }
function EmptyChart({ go }) { return <div className="empty-chart"><div className="empty-ring"><BarChart3 size={22} /></div><strong>Awaiting your first assessment</strong><button onClick={() => go('assessment')}>Start now <ArrowRight size={14} /></button></div> }

function Assessment({ onSubmit, onCancel, error }) {
  const [form, setForm] = useState(initialForm)
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const update = (key, value) => setForm(current => ({ ...current, [key]: value }))
  const submit = async event => {
    event.preventDefault()
    const nextErrors = Object.fromEntries(fieldMeta.filter(([key, , , , required]) => required && !form[key]).map(([key]) => [key, 'This field is required']))
    if (Object.keys(nextErrors).length) { setErrors(nextErrors); return }
    setLoading(true); await onSubmit(Object.fromEntries(Object.entries(form).map(([key, value]) => [key, value === '' ? null : ['project_type', 'state', 'rr_settlement_status', 'forest_clearance_stage', 'env_clearance_stage'].includes(key) ? value : Number(value)]))); setLoading(false)
  }
  return <div className="animate-in"><div className="page-heading"><div><div className="eyebrow"><ClipboardCheck size={14} /> MODEL INPUTS</div><h1>Assess a project.</h1><p>Submit the exact feature set accepted by the fitted DRISHTI model.</p></div><button className="ghost-button" onClick={onCancel}>Cancel</button></div><div className="demo-callout"><Info size={17} /><span><strong>MODEL CONTRACT</strong> — Do not add identifiers, target columns, realized outcomes, or downstream estimates to this request.</span></div>{error && <div className="error-banner"><AlertTriangle size={17} /><span>{error}</span></div>}<form className="assessment-form" onSubmit={submit}><FormSection icon={Landmark} title="Project and land" subtitle="Core project characteristics and land composition">{fieldMeta.slice(0, 8).map(field => <Field key={field[0]} meta={field} value={form[field[0]]} error={errors[field[0]]} update={update} />)}</FormSection><FormSection icon={Leaf} title="Approvals and rehabilitation" subtitle="Clearance and resettlement signals">{fieldMeta.slice(8, 11).map(field => <Field key={field[0]} meta={field} value={form[field[0]]} error={errors[field[0]]} update={update} />)}</FormSection><FormSection icon={Users} title="Compensation and legal signals" subtitle="Financial, cadastral, litigation, and agency indicators">{fieldMeta.slice(11).map(field => <Field key={field[0]} meta={field} value={form[field[0]]} error={errors[field[0]]} update={update} />)}</FormSection><div className="form-actions"><button type="button" className="ghost-button" onClick={() => { setForm(initialForm); setErrors({}) }}>Reset form</button><button type="submit" className="primary-button" disabled={loading}>{loading ? 'Requesting model...' : 'Run model assessment'} <ArrowRight size={16} /></button></div></form></div>
}

function Field({ meta, value, error, update }) {
  const [key, label, type, hint, required] = meta
  return <label className={`field ${error ? 'field-error' : ''}`}><span>{label}{required && <b>*</b>}</span><div className="input-wrap">{type === 'select' ? <><select value={value} onChange={event => update(key, event.target.value)}><option value="">Select option</option>{selectOptions[key].map(option => <option key={option} value={option}>{option}</option>)}</select><ChevronDown size={15} className="select-chevron" /></> : <input type="number" min="0" max={key.includes('pct') || key.includes('score') ? 100 : undefined} step={key.includes('count') || key.includes('parcels') || key.includes('discrepancy') || key.includes('nocs') ? 1 : 'any'} value={value} onChange={event => update(key, event.target.value)} placeholder={hint} />}</div>{error ? <small className="error-text">{error}</small> : <small className="field-hint">{hint}</small>}</label>
}
function FormSection({ icon: Icon, title, subtitle, children }) { return <section className="form-section"><div className="section-title"><div className="section-icon"><Icon size={17} /></div><div><h2>{title}</h2><p>{subtitle}</p></div></div><div className="form-grid">{children}</div></section> }

function Results({ result, go }) {
  if (!result) return <div className="empty-state"><strong>No result selected</strong><button className="primary-button" onClick={() => go('assessment')}>New assessment</button></div>
  return <div className="animate-in"><div className="page-heading"><div><div className="eyebrow"><CheckCircle2 size={14} /> MODEL RESPONSE</div><h1>Assessment results.</h1><p>Returned by the DRISHTI inference service.</p></div><div className="top-actions"><button className="ghost-button" onClick={() => window.print()}><Printer size={16} /> Print report</button><button className="primary-button" onClick={() => go('assessment')}>Analyze another <ArrowRight size={16} /></button></div></div><div className="demo-callout"><Info size={17} /><span><strong>MODEL LIMITATION</strong> — Results are model-specific; risk factors are not causal proof. Validate decisions with domain experts.</span></div><div className="result-hero"><div><span className={`risk-pill risk-${result.delay_risk_level.toLowerCase()}`}><i />{result.delay_risk_level} risk</span><div className="signal-value"><span>Risk probability score</span><strong>{Math.round(result.risk_probability * 100)}</strong><small>/ 100</small></div><p>Continuous regression score returned by the model, not the classifier's High-class probability.</p></div><div><h2>Top model-attributed factors</h2>{result.top_risk_factors?.length ? result.top_risk_factors.map(factor => <div className="factor-row" key={factor.feature}><strong>{factor.display_name}</strong><span>{factor.contribution > 0 ? '+' : ''}{factor.contribution.toFixed(3)}</span><small>{factor.direction.replaceAll('_', ' ')} · {factor.description}</small></div>) : <p>No factors returned.</p>}</div></div><section className="panel stack-panel"><div className="panel-head"><div><h2>Response metadata</h2><p>Use this information when integrating or auditing the result.</p></div></div><div className="stack-list"><span><strong>Model version</strong><small>{result.model_version}</small></span><span><strong>API endpoint</strong><small>POST /predict</small></span><span><strong>Factors returned</strong><small>{result.top_risk_factors?.length || 0} of 3 maximum</small></span></div></section></div>
}

function About() {
  return <div className="animate-in about-page"><div className="about-hero"><div><div className="eyebrow"><Landmark size={14} /> ABOUT THE PROJECT</div><h1>Making land acquisition<br /><em>more predictable.</em></h1><p>DRISHTI is an AI-powered predictive risk system designed to help infrastructure teams identify delay signals earlier and act with greater confidence.</p></div><div className="about-orbit"><div className="orbit-center"><Landmark size={27} /></div><span className="orbit-node node-a"><Leaf size={16} /></span><span className="orbit-node node-b"><BarChart3 size={16} /></span><span className="orbit-node node-c"><Zap size={16} /></span></div></div><div className="about-grid"><InfoBlock icon={AlertTriangle} title="The problem" text="Land acquisition delays are often discovered after they have already affected schedules, budgets, and public commitments. Fragmented signals make early intervention difficult." /><InfoBlock icon={TrendingUp} title="The objective" text="Surface a clear risk view across land, approvals, legal, compensation, and district-level signals before delays compound." /><InfoBlock icon={ShieldCheck} title="The contract" text="The frontend sends only the 18 fitted model features. The backend remains the source of truth for validation, prediction, model version, and explanations." /></div><section className="panel stack-panel"><div className="panel-head"><div><h2>Technology stack</h2><p>Integration-ready full-stack foundation</p></div></div><div className="stack-list"><span><strong>React + Vite</strong><small>Frontend application</small></span><span><strong>Python + FastAPI</strong><small>Inference API at /predict</small></span><span><strong>Scikit-learn + XGBoost</strong><small>Dual-target model workflow</small></span><span><strong>SHAP</strong><small>Model-attributed explanations</small></span></div></section><div className="limitation-banner"><AlertTriangle size={19} /><div><strong>Synthetic-data limitation</strong><p>The current model was trained on synthetic demonstration data. It is not evidence of real-world predictive validity and requires domain validation before operational use.</p></div></div></div>
}
function InfoBlock({ icon: Icon, title, text }) { return <div className="info-block"><div className="info-icon"><Icon size={19} /></div><h2>{title}</h2><p>{text}</p></div> }

createRoot(document.getElementById('root')).render(<App />)
