import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  AlertTriangle, ArrowRight, BarChart3, Bell, CheckCircle2, ChevronDown, CircleHelp,
  ClipboardCheck, Clock3, Download, FileText, Gauge, Home, Info, Landmark,
  Leaf, Menu, Printer, RotateCcw, Scale, Search, ShieldCheck, Sparkles,
  TrendingUp, Users, X, Zap,
} from 'lucide-react'
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import './styles.css'

const STORAGE_KEY = 'drishti-demo-assessments'
const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Home },
  { id: 'assessment', label: 'New assessment', icon: ClipboardCheck },
  { id: 'about', label: 'About DRISHTI', icon: CircleHelp },
]

const initialForm = {
  projectId: '', projectName: '', state: '', district: '', projectType: '',
  projectStartDate: '', landArea: '', totalParcels: '', privateLandPct: '',
  forestLandPct: '', govtLandPct: '', affectedFamilies: '', cadastralDiscrepancy: '',
  forestClearance: '', environmentalClearance: '', pendingNocs: '',
  activeLitigations: '', stayOrder: 'No', circleRate: '', compensationPct: '',
  rrStatus: '', historicalDelayScore: '', targetDuration: '',
}

const demoSeed = [
  { id: 'DR-2409', name: 'Eastern Freight Corridor', state: 'West Bengal', risk: 'High', score: 82, date: 'Today, 09:42', demo: true },
  { id: 'DR-2408', name: 'Greenfield Solar Park', state: 'Rajasthan', risk: 'Medium', score: 54, date: 'Yesterday, 16:18', demo: true },
  { id: 'DR-2407', name: 'Coastal Link Road', state: 'Maharashtra', risk: 'Low', score: 27, date: 'Yesterday, 11:06', demo: true },
]

const palette = { High: '#ef6b70', Medium: '#f4b860', Low: '#48c3b3' }
const riskColor = risk => palette[risk] || '#7890aa'
const loadHistory = () => {
  try { return JSON.parse(sessionStorage.getItem(STORAGE_KEY)) || [] } catch { return [] }
}

function App() {
  const [page, setPage] = useState('dashboard')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [history, setHistory] = useState(loadHistory)
  const [result, setResult] = useState(null)

  useEffect(() => sessionStorage.setItem(STORAGE_KEY, JSON.stringify(history)), [history])
  const records = history
  const dashboardRecords = records.length ? records : demoSeed
  const counts = useMemo(() => ({
    High: dashboardRecords.filter(item => item.risk === 'High').length,
    Medium: dashboardRecords.filter(item => item.risk === 'Medium').length,
    Low: dashboardRecords.filter(item => item.risk === 'Low').length,
  }), [dashboardRecords])

  const go = id => { setPage(id); setMobileOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  const handleSubmit = form => {
    const scored = calculateDemo(form)
    const assessment = { ...scored, id: form.projectId || `DR-${Date.now().toString().slice(-4)}`, name: form.projectName || 'Untitled infrastructure project', state: form.state || 'State not specified', date: 'Just now', demo: true, form }
    setHistory(current => [assessment, ...current].slice(0, 20))
    setResult(assessment)
    setPage('results')
  }

  return (
    <div className="app-shell">
      <Sidebar page={page} go={go} mobileOpen={mobileOpen} close={() => setMobileOpen(false)} />
      <div className="main-shell">
        <Header onMenu={() => setMobileOpen(true)} onSearch={() => go('dashboard')} />
        <main className="page-wrap">
          {page === 'dashboard' && <Dashboard records={dashboardRecords} counts={counts} go={go} hasHistory={records.length > 0} />}
          {page === 'assessment' && <Assessment onSubmit={handleSubmit} onCancel={() => go('dashboard')} />}
          {page === 'results' && <Results result={result} go={go} />}
          {page === 'about' && <About />}
        </main>
      </div>
    </div>
  )
}

function Sidebar({ page, go, mobileOpen, close }) {
  return <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
    <div className="brand-row"><div className="brand-mark"><Landmark size={21} /></div><div><div className="brand-name">DRISHTI</div><div className="brand-sub">Risk intelligence</div></div><button className="icon-button mobile-close" onClick={close}><X size={18} /></button></div>
    <div className="demo-badge"><span className="pulse-dot" /> DEMO MODE</div>
    <nav className="nav-list">
      <div className="nav-label">WORKSPACE</div>
      {navItems.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => go(id)} className={`nav-item ${page === id || (page === 'results' && id === 'assessment') ? 'active' : ''}`}><Icon size={18} /><span>{label}</span>{id === 'assessment' && <span className="nav-plus">+</span>}</button>)}
    </nav>
    <div className="sidebar-bottom">
      <div className="security-card"><ShieldCheck size={18} /><div><strong>Private workspace</strong><span>Session-only demo data</span></div></div>
      <div className="sidebar-footer"><div className="avatar">AS</div><div><strong>Analyst session</strong><span>Local workspace</span></div><ChevronDown size={14} /></div>
    </div>
  </aside>
}

function Header({ onMenu, onSearch }) {
  return <header className="topbar"><button className="icon-button menu-button" onClick={onMenu}><Menu size={21} /></button><div className="crumbs"><span>Workspace</span><span>/</span><strong>DRISHTI Intelligence</strong></div><div className="top-actions"><button className="search-control" onClick={onSearch}><Search size={16} /><span>Search projects</span><kbd>⌘ K</kbd></button><button className="icon-button notification"><Bell size={18} /><i /></button><div className="top-avatar">AS</div></div></header>
}

function Dashboard({ records, counts, go, hasHistory }) {
  const total = records.length
  const chart = [{ name: 'High', value: counts.High, color: palette.High }, { name: 'Medium', value: counts.Medium, color: palette.Medium }, { name: 'Low', value: counts.Low, color: palette.Low }]
  return <div className="animate-in">
    <div className="page-heading"><div><div className="eyebrow"><span className="live-dot" /> OPERATIONAL OVERVIEW</div><h1>Good morning, analyst.</h1><p>Monitor acquisition risk across your infrastructure portfolio.</p></div><button className="primary-button" onClick={() => go('assessment')}><Sparkles size={17} /> New assessment <ArrowRight size={16} /></button></div>
    <div className="demo-callout"><Info size={17} /><span><strong>DEMO DATA ACTIVE</strong> — {hasHistory ? 'Your session assessments are shown below.' : 'Seeded examples are shown for demonstration. Submit an assessment to add your own.'}</span><button onClick={() => go('about')}>Learn more</button></div>
    <section className="metric-grid">
      <MetricCard label="Session assessments" value={total} icon={ClipboardCheck} detail="Current browser session" accent="cyan" />
      <MetricCard label="High risk projects" value={counts.High} icon={AlertTriangle} detail={total ? `${Math.round((counts.High / total) * 100)}% of portfolio` : 'No assessments yet'} accent="red" />
      <MetricCard label="Medium risk projects" value={counts.Medium} icon={Gauge} detail={total ? `${Math.round((counts.Medium / total) * 100)}% of portfolio` : 'No assessments yet'} accent="amber" />
      <MetricCard label="Low risk projects" value={counts.Low} icon={CheckCircle2} detail={total ? `${Math.round((counts.Low / total) * 100)}% of portfolio` : 'No assessments yet'} accent="green" />
    </section>
    <div className="dashboard-grid">
      <section className="panel chart-panel"><div className="panel-head"><div><h2>Risk distribution</h2><p>Current assessment portfolio</p></div><button className="ghost-button"><span className="status-dot" /> Live session</button></div><div className="chart-wrap">{total ? <><ResponsiveContainer width="52%" height={245}><PieChart><Pie data={chart} innerRadius={75} outerRadius={103} paddingAngle={4} dataKey="value" stroke="none">{chart.map(entry => <Cell key={entry.name} fill={entry.color} />)}</Pie><Tooltip content={<ChartTooltip />} /></PieChart></ResponsiveContainer><div className="legend-list">{chart.map(item => <div className="legend-row" key={item.name}><span className="legend-label"><i style={{ background: item.color }} />{item.name}</span><strong>{item.value}</strong><span className="legend-percent">{Math.round((item.value / total) * 100)}%</span></div>)}</div></> : <EmptyChart go={go} />}</div></section>
      <section className="panel signal-panel"><div className="panel-head"><div><h2>Risk outlook</h2><p>Illustrative trend signal</p></div><TrendingUp size={19} className="muted-icon" /></div><div className="signal-value"><span>Portfolio index</span><strong>{total ? Math.round((counts.High * 82 + counts.Medium * 54 + counts.Low * 27) / total) : '—'}</strong><small>/ 100</small></div><div className="mini-chart"><ResponsiveContainer width="100%" height={100}><AreaChart data={[{v: 31},{v: 38},{v: 34},{v: 48},{v: 42},{v: 57},{v: total ? 61 : 52}]}><defs><linearGradient id="cyanFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#28bfd0" stopOpacity=".35" /><stop offset="100%" stopColor="#28bfd0" stopOpacity="0" /></linearGradient></defs><Area type="monotone" dataKey="v" stroke="#28bfd0" fill="url(#cyanFill)" strokeWidth={2} dot={false} /><XAxis hide dataKey="v" /><YAxis hide domain={['dataMin - 10', 'dataMax + 10']} /><Tooltip content={<ChartTooltip />} /></AreaChart></ResponsiveContainer></div><div className="signal-note"><span className="arrow-up">↗</span><span><strong>Monitor leading indicators</strong><br />Update assessments as approvals evolve.</span></div></section>
    </div>
    <section className="panel recent-panel"><div className="panel-head"><div><h2>Recent assessments</h2><p>Latest projects evaluated in this session</p></div><button className="text-button" onClick={() => go('assessment')}>View all <ArrowRight size={15} /></button></div>{records.length ? <div className="table-wrap"><table><thead><tr><th>Project</th><th>Location</th><th>Risk level</th><th>Score</th><th>Assessed</th><th /></tr></thead><tbody>{records.slice(0, 5).map(record => <AssessmentRow key={record.id} record={record} />)}</tbody></table></div> : <div className="empty-state"><div className="empty-icon"><FileText size={22} /></div><h3>No assessments in this session</h3><p>Start with a new project assessment to build your risk portfolio.</p><button className="secondary-button" onClick={() => go('assessment')}>Create first assessment <ArrowRight size={15} /></button></div>}</section>
  </div>
}

function MetricCard({ label, value, icon: Icon, detail, accent }) { return <div className={`metric-card metric-${accent}`}><div className="metric-top"><div className="metric-icon"><Icon size={18} /></div><span className="metric-kicker">SESSION</span></div><div className="metric-value">{value}</div><div className="metric-label">{label}</div><div className="metric-detail">{detail}</div></div> }
function ChartTooltip({ active, payload }) { if (!active || !payload?.length) return null; return <div className="chart-tooltip">{payload[0].value} assessments</div> }
function EmptyChart({ go }) { return <div className="empty-chart"><div className="empty-ring"><BarChart3 size={22} /></div><strong>Awaiting your first assessment</strong><button onClick={() => go('assessment')}>Start now <ArrowRight size={14} /></button></div> }
function AssessmentRow({ record }) { return <tr><td><div className="project-cell"><span className="project-symbol"><Landmark size={15} /></span><span><strong>{record.name}</strong><small>{record.id}</small></span></div></td><td>{record.state}</td><td><span className={`risk-pill risk-${record.risk.toLowerCase()}`}><i />{record.risk}</span></td><td><div className="score-cell"><strong>{record.score}</strong><div className="score-bar"><span style={{ width: `${record.score}%`, background: riskColor(record.risk) }} /></div></div></td><td className="muted-text">{record.date}</td><td><button className="row-menu">•••</button></td></tr> }

function Assessment({ onSubmit, onCancel }) {
  const [form, setForm] = useState(initialForm)
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const update = (key, value) => setForm(current => ({ ...current, [key]: value }))
  const submit = event => { event.preventDefault(); const next = {}; ['projectName', 'state', 'projectType', 'landArea', 'totalParcels'].forEach(key => { if (!form[key]) next[key] = 'This field is required' }); if (Object.keys(next).length) { setErrors(next); return } setErrors({}); setLoading(true); setTimeout(() => { onSubmit(form); setLoading(false) }, 750) }
  return <div className="animate-in"><div className="page-heading assessment-heading"><div><div className="eyebrow"><ClipboardCheck size={14} /> PROJECT RISK ASSESSMENT</div><h1>Assess a new project</h1><p>Enter the available project signals to generate a transparent demo assessment.</p></div><span className="demo-chip"><span className="pulse-dot" /> DEMO MODE</span></div><form onSubmit={submit}><div className="form-layout"><div className="form-main"><FormSection icon={Landmark} title="Project details" subtitle="Core identifiers and project classification"><div className="form-grid"><Input label="Project ID" value={form.projectId} onChange={v => update('projectId', v)} placeholder="e.g. DR-2025-001" /><Input label="Project name" required error={errors.projectName} value={form.projectName} onChange={v => update('projectName', v)} placeholder="Enter project name" /><Select label="State" required error={errors.state} value={form.state} onChange={v => update('state', v)} options={['Andhra Pradesh','Maharashtra','Rajasthan','Tamil Nadu','Telangana','West Bengal']} /><Input label="District" value={form.district} onChange={v => update('district', v)} placeholder="Enter district" /><Select label="Project type" required error={errors.projectType} value={form.projectType} onChange={v => update('projectType', v)} options={['Industrial Corridor','Highway','Railway','Irrigation','Urban Development','Renewable Energy']} /><Input label="Project start date" type="date" value={form.projectStartDate} onChange={v => update('projectStartDate', v)} /><Input label="Land area" required error={errors.landArea} value={form.landArea} onChange={v => update('landArea', v)} placeholder="0.00" suffix="ha" type="number" /><Input label="Total land parcels" required error={errors.totalParcels} value={form.totalParcels} onChange={v => update('totalParcels', v)} placeholder="0" type="number" /></div></FormSection><FormSection icon={Leaf} title="Land details" subtitle="Land composition and affected stakeholders"><div className="form-grid"><Input label="Private land" value={form.privateLandPct} onChange={v => update('privateLandPct', v)} placeholder="0–100" suffix="%" type="number" /><Input label="Forest land" value={form.forestLandPct} onChange={v => update('forestLandPct', v)} placeholder="0–100" suffix="%" type="number" /><Input label="Government land" value={form.govtLandPct} onChange={v => update('govtLandPct', v)} placeholder="0–100" suffix="%" type="number" /><Input label="Affected families" value={form.affectedFamilies} onChange={v => update('affectedFamilies', v)} placeholder="0" type="number" /><Input label="Cadastral discrepancy" value={form.cadastralDiscrepancy} onChange={v => update('cadastralDiscrepancy', v)} placeholder="0" type="number" /></div></FormSection><FormSection icon={Scale} title="Approvals & legal" subtitle="Clearances, inter-agency dependencies, and litigation signals"><div className="form-grid"><Select label="Forest clearance" value={form.forestClearance} onChange={v => update('forestClearance', v)} options={['Not initiated','Proposal submitted','Stage-1 Approved','Stage-2 Approved','Not applicable']} /><Select label="Environmental clearance" value={form.environmentalClearance} onChange={v => update('environmentalClearance', v)} options={['Not initiated','Under review','EC Granted','Not applicable']} /><Input label="Pending inter-agency NOCs" value={form.pendingNocs} onChange={v => update('pendingNocs', v)} placeholder="0" type="number" /><Input label="Active litigations" value={form.activeLitigations} onChange={v => update('activeLitigations', v)} placeholder="0" type="number" /><Select label="Stay order present" value={form.stayOrder} onChange={v => update('stayOrder', v)} options={['No','Yes']} /></div></FormSection><FormSection icon={Users} title="Compensation & rehabilitation" subtitle="Financial readiness and resettlement progress"><div className="form-grid"><Input label="Circle rate" value={form.circleRate} onChange={v => update('circleRate', v)} placeholder="0.00" suffix="₹ / m²" type="number" /><Input label="Compensation disbursed" value={form.compensationPct} onChange={v => update('compensationPct', v)} placeholder="0–100" suffix="%" type="number" /><Select label="R&R settlement status" value={form.rrStatus} onChange={v => update('rrStatus', v)} options={['Not started','In progress','Mostly Completed','Completed','Not applicable']} /></div></FormSection><FormSection icon={Clock3} title="Planning horizon" subtitle="Schedule expectations and historical district signal"><div className="form-grid"><Input label="Historical delay score" value={form.historicalDelayScore} onChange={v => update('historicalDelayScore', v)} placeholder="0.00–1.00" type="number" /><Input label="Target duration" value={form.targetDuration} onChange={v => update('targetDuration', v)} placeholder="0" suffix="days" type="number" /></div></FormSection></div><aside className="form-aside"><div className="aside-card"><div className="aside-icon"><Sparkles size={19} /></div><h3>Transparent demo analysis</h3><p>This frontend-only assessment uses a deterministic rule-based simulation. No backend or live ML model is connected.</p><div className="aside-list"><span><CheckCircle2 size={15} /> Instant results</span><span><CheckCircle2 size={15} /> Session-only history</span><span><CheckCircle2 size={15} /> Explainable factors</span></div></div><div className="aside-tip"><Zap size={16} /><span><strong>Tip</strong> — Complete more fields for a more representative demonstration score.</span></div></aside></div><div className="form-actions"><button type="button" className="secondary-button" onClick={() => { setForm(initialForm); setErrors({}) }}><RotateCcw size={15} /> Reset form</button><div><button type="button" className="ghost-button cancel-button" onClick={onCancel}>Cancel</button><button className="primary-button" disabled={loading}>{loading ? <><span className="spinner" /> Analyzing demo…</> : <><Sparkles size={16} /> Generate demo assessment <ArrowRight size={16} /></>}</button></div></div></form></div>
}

function FormSection({ icon: Icon, title, subtitle, children }) { return <section className="form-section"><div className="section-title"><div className="section-icon"><Icon size={17} /></div><div><h2>{title}</h2><p>{subtitle}</p></div></div>{children}</section> }
function Input({ label, required, error, value, onChange, placeholder, type = 'text', suffix }) { return <label className={`field ${error ? 'field-error' : ''}`}><span>{label}{required && <b>*</b>}</span><div className="input-wrap"><input type={type} min={type === 'number' ? 0 : undefined} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} />{suffix && <em>{suffix}</em>}</div>{error && <small className="error-text">{error}</small>}</label> }
function Select({ label, required, error, value, onChange, options }) { return <label className={`field ${error ? 'field-error' : ''}`}><span>{label}{required && <b>*</b>}</span><div className="input-wrap"><select value={value} onChange={e => onChange(e.target.value)}><option value="">Select option</option>{options.map(option => <option key={option}>{option}</option>)}</select><ChevronDown size={15} className="select-chevron" /></div>{error && <small className="error-text">{error}</small>}</label> }

function Results({ result, go }) {
  if (!result) return <div className="empty-state standalone"><div className="empty-icon"><FileText size={22} /></div><h3>No assessment selected</h3><p>Create an assessment to view a demo report.</p><button className="primary-button" onClick={() => go('assessment')}>Start assessment <ArrowRight size={16} /></button></div>
  return <div className="animate-in report-page"><div className="page-heading"><div><div className="eyebrow"><span className="live-dot" /> DEMO ANALYSIS COMPLETE</div><h1>Assessment results</h1><p>{result.name} · {result.state}</p></div><div className="result-actions"><button className="secondary-button" onClick={() => window.print()}><Printer size={16} /> Print report</button><button className="primary-button" onClick={() => go('assessment')}><Sparkles size={16} /> Analyze another</button></div></div><div className="demo-callout"><AlertTriangle size={17} /><span><strong>DEMO MODE RESULT</strong> — This is a deterministic frontend simulation, not a prediction from the DRISHTI ML service.</span></div><div className="result-grid"><section className={`result-hero risk-hero-${result.risk.toLowerCase()}`}><div className="result-hero-top"><span className="risk-pill risk-large"><i />{result.risk} risk</span><span className="illustrative">Illustrative estimate</span></div><div className="result-score"><div><span>Demo risk score</span><strong>{result.score}<small>/100</small></strong></div><div className="score-ring" style={{ '--score': `${result.score * 3.6}deg`, '--ring-color': riskColor(result.risk) }}><div>{result.score}<small>risk index</small></div></div></div><div className="result-meter"><span style={{ width: `${result.score}%`, background: riskColor(result.risk) }} /></div><p className="hero-note">The score reflects the signals entered for this demonstration and is intended for interface evaluation only.</p></section><section className="panel delay-card"><div className="panel-head"><div><h2>Delay outlook</h2><p>Illustrative estimate based on demo rules</p></div><Clock3 className="muted-icon" size={20} /></div><div className="delay-number">{result.delayMonths}<span> months</span></div><p>Potential schedule extension</p><div className="mini-callout"><Info size={15} /> Not a live model output</div></section></div><div className="result-columns"><section className="panel factors-panel"><div className="panel-head"><div><h2>Contributing demo factors</h2><p>Signals that influenced this illustrative result</p></div><span className="factor-count">04 factors</span></div><div className="factor-list">{result.factors.map(factor => <div className="factor-item" key={factor.label}><div className={`factor-icon ${factor.tone}`}><factor.icon size={18} /></div><div className="factor-content"><div><strong>{factor.label}</strong><span className={`factor-impact ${factor.tone}`}>{factor.impact}</span></div><p>{factor.detail}</p><div className="factor-bar"><span style={{ width: `${factor.weight}%`, background: factor.color }} /></div></div></div>)}</div></section><section className="panel rec-panel"><div className="panel-head"><div><h2>Recommended actions</h2><p>General guidance for review</p></div><ShieldCheck size={19} className="muted-icon" /></div><ul className="rec-list">{result.recommendations.map(item => <li key={item}><span><CheckCircle2 size={15} /></span>{item}</li>)}</ul><div className="disclaimer-box"><Info size={16} /><span>These are general recommendations for demonstration. Validate with domain experts.</span></div></section></div><section className="panel submitted-panel"><div className="panel-head"><div><h2>Submitted project details</h2><p>Inputs captured for this session assessment</p></div><span className="demo-chip">DEMO RECORD</span></div><div className="submitted-grid">{[['Project ID', result.id], ['Project name', result.name], ['State', result.state], ['Project type', result.form?.projectType || '—'], ['Land area', result.form?.landArea ? `${result.form.landArea} ha` : '—'], ['Land parcels', result.form?.totalParcels || '—'], ['Start date', result.form?.projectStartDate || '—'], ['Assessment time', result.date]].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div></section></div>
}

function About() { return <div className="animate-in about-page"><div className="about-hero"><div><div className="eyebrow"><Landmark size={14} /> ABOUT THE PROJECT</div><h1>Making land acquisition<br /><em>more predictable.</em></h1><p>DRISHTI is an AI-powered predictive risk system designed to help infrastructure teams identify delay signals earlier and act with greater confidence.</p></div><div className="about-orbit"><div className="orbit-center"><Landmark size={27} /></div><span className="orbit-node node-a"><Leaf size={16} /></span><span className="orbit-node node-b"><BarChart3 size={16} /></span><span className="orbit-node node-c"><Scale size={16} /></span></div></div><div className="about-grid"><InfoBlock icon={AlertTriangle} title="The problem" text="Land acquisition delays are often discovered after they have already affected schedules, budgets, and public commitments. Fragmented signals make early intervention difficult." /><InfoBlock icon={TrendingUp} title="Our objective" text="Surface a clear, evidence-informed risk view across land, approvals, legal, compensation, and district-level signals — before delays compound." /><InfoBlock icon={Zap} title="The workflow" text="Validate project inputs → prepare features → train dual classification and regression pipelines → explain key model-attributed factors → support action." /></div><section className="panel stack-panel"><div className="panel-head"><div><h2>Technology stack</h2><p>Built for a transparent SIH demonstration</p></div></div><div className="stack-list"><span><strong>React + Vite</strong><small>Responsive frontend experience</small></span><span><strong>Python + FastAPI</strong><small>Inference service architecture</small></span><span><strong>Scikit-learn + XGBoost</strong><small>Dual-target model workflow</small></span><span><strong>SHAP</strong><small>Model-attributed explanations</small></span></div></section><div className="limitation-banner"><AlertTriangle size={19} /><div><strong>Important limitation</strong><p>The current project models and this frontend use synthetic demonstration data. Demo scores are not real predictions, causal conclusions, or a substitute for domain validation and operational monitoring.</p></div></div></div> }
function InfoBlock({ icon: Icon, title, text }) { return <div className="info-block"><div className="info-icon"><Icon size={19} /></div><h2>{title}</h2><p>{text}</p></div> }

function calculateDemo(form) {
  let score = 24; const factors = []
  const add = (condition, amount, factor) => { if (condition) { score += amount; factors.push(factor) } }
  add(Number(form.activeLitigations) > 0, Math.min(Number(form.activeLitigations || 0) * 7, 20), { label: 'Active litigation exposure', impact: 'Elevated', detail: `${form.activeLitigations || 0} active matter(s) may affect acquisition continuity.`, tone: 'red', weight: 76, color: palette.High, icon: Scale })
  add(form.stayOrder === 'Yes', 19, { label: 'Stay order present', impact: 'Critical', detail: 'A stay order can pause field activity and downstream clearances.', tone: 'red', weight: 92, color: palette.High, icon: AlertTriangle })
  add(Number(form.pendingNocs) > 1, 12, { label: 'Inter-agency dependencies', impact: 'Watch', detail: `${form.pendingNocs} pending NOC(s) increase coordination overhead.`, tone: 'amber', weight: 58, color: palette.Medium, icon: Landmark })
  add(Number(form.compensationPct) < 50 && form.compensationPct !== '', 13, { label: 'Compensation progress', impact: 'Needs attention', detail: 'Lower disbursement progress can slow consent and rehabilitation activity.', tone: 'amber', weight: 64, color: palette.Medium, icon: Users })
  add(Number(form.cadastralDiscrepancy) > 0, 10, { label: 'Cadastral discrepancy', impact: 'Needs attention', detail: 'Survey discrepancies may require additional ground verification.', tone: 'amber', weight: 52, color: palette.Medium, icon: Search })
  add(Number(form.forestLandPct) > 20 && form.forestLandPct !== '', 9, { label: 'Forest land exposure', impact: 'Monitor', detail: 'Higher forest land share can add clearance dependencies.', tone: 'cyan', weight: 43, color: '#28bfd0', icon: Leaf })
  if (!factors.length) factors.push({ label: 'Stable project signals', impact: 'Favorable', detail: 'No major risk flags were detected in the provided demo inputs.', tone: 'green', weight: 28, color: palette.Low, icon: CheckCircle2 })
  score = Math.min(96, Math.round(score)); const risk = score >= 68 ? 'High' : score >= 43 ? 'Medium' : 'Low'
  const recs = factors.some(f => f.label.includes('litigation')) ? ['Review litigation timelines with legal counsel.', 'Create a weekly clearance dependency review.', 'Refresh the assessment when a matter changes.'] : ['Maintain a current clearance tracker.', 'Confirm compensation and R&R milestones monthly.', 'Re-assess after major project approvals.']
  return { risk, score, delayMonths: Math.max(1, Math.round(score / 13)), factors: factors.slice(0, 4), recommendations: recs }
}

createRoot(document.getElementById('root')).render(<App />)
