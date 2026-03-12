import React from 'react';

// ── Page wrapper ──────────────────────────────────────────────────────
export function Page({ children }) {
    return (
        <div style={{ paddingTop: 72, minHeight: '100vh', background: '#020617' }}>
            <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 20px 60px' }}>
                {children}
            </div>
        </div>
    );
}

// ── Page title ────────────────────────────────────────────────────────
export function PageTitle({ title, sub }) {
    return (
        <div style={{ marginBottom: 28 }}>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: '#f1f5f9', marginBottom: 6 }}>
                {title}
            </h1>
            <p style={{ color: '#64748b', fontSize: 14 }}>{sub}</p>
        </div>
    );
}

// ── Section title ─────────────────────────────────────────────────────
export function SectionTitle({ title, sub }) {
    return (
        <div style={{ marginBottom: 14 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#f1f5f9', marginBottom: 3 }}>
                {title}
            </h2>
            {sub && <p style={{ fontSize: 12, color: '#64748b' }}>{sub}</p>}
        </div>
    );
}

// ── Card ──────────────────────────────────────────────────────────────
export function Card({ children, style = {} }) {
    return (
        <div style={{
            background: '#111827',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 12,
            padding: 20,
            ...style
        }}>
            {children}
        </div>
    );
}

// ── Stat card ─────────────────────────────────────────────────────────
export function StatCard({ icon, label, value, color = '#ef4444', sub }) {
    return (
        <Card style={{ display: 'flex', alignItems: 'center', gap: 14, padding: 16 }}>
            <div style={{
                background: `${color}20`,
                borderRadius: 10,
                padding: 10,
                fontSize: 22,
                lineHeight: 1
            }}>
                {icon}
            </div>
            <div>
                <div style={{
                    fontSize: 10, color: '#64748b',
                    textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 2
                }}>
                    {label}
                </div>
                <div style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', lineHeight: 1 }}>
                    {value}
                </div>
                {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
            </div>
        </Card>
    );
}

// ── Tab bar ───────────────────────────────────────────────────────────
export function TabBar({ tabs, active, onChange }) {
    return (
        <div style={{
            display: 'flex', gap: 4,
            background: '#0b1120',
            borderRadius: 8, padding: 4,
            marginBottom: 16
        }}>
            {tabs.map(t => (
                <button
                    key={t}
                    onClick={() => onChange(t)}
                    style={{
                        flex: 1, padding: '7px 12px',
                        borderRadius: 6, border: 'none',
                        fontSize: 12, fontWeight: 600,
                        background: active === t ? '#ef4444' : 'transparent',
                        color: active === t ? '#fff' : '#64748b',
                        transition: 'all 0.15s',
                    }}
                >
                    {t}
                </button>
            ))}
        </div>
    );
}

// ── Select dropdown ───────────────────────────────────────────────────
export function Select({ label, options, value, onChange }) {
    return (
        <div>
            {label && (
                <div style={{
                    fontSize: 11, color: '#64748b',
                    marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5
                }}>
                    {label}
                </div>
            )}
            <select
                value={value}
                onChange={e => onChange(e.target.value)}
                style={{
                    background: '#0b1120',
                    border: '1px solid rgba(255,255,255,0.08)',
                    color: '#f1f5f9',
                    borderRadius: 6,
                    padding: '7px 10px',
                    fontSize: 13,
                    width: '100%',
                    cursor: 'pointer',
                }}
            >
                {options.map(o => (
                    <option key={o} value={o}>{o}</option>
                ))}
            </select>
        </div>
    );
}   