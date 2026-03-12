import React from 'react';
import { Link } from 'react-router-dom';
import overviewData from '../data/overview.json';

const { paper, dataset, groups } = overviewData;

export default function Home() {
    return (
        <div style={{ paddingTop: 56, minHeight: '100vh', background: '#020617' }}>

            {/* ── Hero section ───────────────────────────────────────────── */}
            <div style={{
                background: 'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(239,68,68,0.15), transparent)',
                padding: '80px 20px 60px',
                textAlign: 'center',
                maxWidth: 860,
                margin: '0 auto'
            }}>

                {/* Top badge */}
                <div style={{
                    display: 'inline-block',
                    background: 'rgba(167,139,250,0.12)',
                    border: '1px solid rgba(167,139,250,0.3)',
                    borderRadius: 20,
                    padding: '5px 16px',
                    fontSize: 12,
                    color: '#c4b5fd',
                    marginBottom: 20,
                    fontWeight: 500,
                }}>
                    Final Year Project · {paper.venue}
                </div>

                {/* Main title */}
                <h1 style={{
                    fontSize: 'clamp(24px, 5vw, 44px)',
                    fontWeight: 800,
                    lineHeight: 1.2,
                    marginBottom: 16,
                    letterSpacing: -0.5,
                }}>
                    <span style={{
                        background: 'linear-gradient(135deg, #f1f5f9, #94a3b8)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>
                        Data-Driven Crime Analysis
                    </span>
                    <br />
                    <span style={{
                        background: 'linear-gradient(135deg, #ef4444, #f97316)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>
                        & Prediction Using ML
                    </span>
                </h1>

                {/* Subtitle */}
                <p style={{
                    fontSize: 15,
                    color: '#94a3b8',
                    lineHeight: 1.7,
                    maxWidth: 600,
                    margin: '0 auto 10px',
                }}>
                    {paper.subtitle}
                </p>

                {/* Dataset quick facts */}
                <p style={{ fontSize: 13, color: '#475569', marginBottom: 36 }}>
                    NCRB India · 2017–2022 · {dataset.total_records.toLocaleString()} records
                    · {dataset.states} states · {dataset.districts}+ districts
                </p>

                {/* Key numbers row */}
                <div style={{
                    display: 'inline-flex',
                    gap: 0,
                    background: 'rgba(167,139,250,0.08)',
                    border: '1px solid rgba(167,139,250,0.2)',
                    borderRadius: 12,
                    padding: '14px 24px',
                    marginBottom: 40,
                    flexWrap: 'wrap',
                    justifyContent: 'center',
                    gap: 20,
                }}>
                    <Kpi label="Our Model" value="FC-MT-LSTM" color="#a78bfa" />
                    <Divider />
                    <Kpi label="R² Score" value="0.9922" color="#a78bfa" />
                    <Divider />
                    <Kpi label="Fairness Ratio" value="1.99" color="#34d399" />
                    <Divider />
                    <Kpi label="Groups Covered" value="4" color="#64B5F6" />
                </div>

                {/* Buttons */}
                <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
                    <Link to="/overview" style={{
                        background: 'linear-gradient(135deg, #ef4444, #f97316)',
                        color: '#fff',
                        borderRadius: 8,
                        padding: '10px 24px',
                        fontSize: 14,
                        fontWeight: 600,
                        display: 'inline-block',
                    }}>
                        Explore the Data →
                    </Link>
                    <Link to="/trends" style={{
                        background: 'transparent',
                        color: '#94a3b8',
                        borderRadius: 8,
                        padding: '10px 24px',
                        fontSize: 14,
                        fontWeight: 500,
                        border: '1px solid rgba(255,255,255,0.08)',
                        display: 'inline-block',
                    }}>
                        View Trends
                    </Link>
                </div>
            </div>

            {/* ── Protected groups ───────────────────────────────────────── */}
            <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 20px 48px' }}>
                <h2 style={{
                    textAlign: 'center', fontSize: 17, fontWeight: 700,
                    color: '#64748b', marginBottom: 20,
                }}>
                    4 Protected Groups Studied
                </h2>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                    gap: 14,
                }}>
                    {groups.map(g => (
                        <div key={g.name} style={{
                            background: '#111827',
                            border: `1px solid ${g.color}22`,
                            borderLeft: `3px solid ${g.color}`,
                            borderRadius: 12,
                            padding: 18,
                        }}>
                            <div style={{
                                display: 'flex', alignItems: 'center',
                                gap: 8, marginBottom: 8,
                            }}>
                                <div style={{
                                    width: 10, height: 10,
                                    borderRadius: 2, background: g.color,
                                }} />
                                <span style={{ fontWeight: 700, color: g.color, fontSize: 15 }}>
                                    {g.name}
                                </span>
                                <span style={{ marginLeft: 'auto', fontSize: 11, color: '#475569' }}>
                                    {g.records.toLocaleString()} records
                                </span>
                            </div>
                            <p style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
                                {g.desc}
                            </p>
                            <div style={{ marginTop: 8, fontSize: 11, color: '#475569' }}>
                                {g.categories} crime categories
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* ── Navigation cards ───────────────────────────────────────── */}
            <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 20px 48px' }}>
                <h2 style={{
                    textAlign: 'center', fontSize: 17, fontWeight: 700,
                    color: '#64748b', marginBottom: 20,
                }}>
                    Explore the Dashboard
                </h2>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                    gap: 14,
                }}>
                    {navCards.map(card => (
                        <Link key={card.to} to={card.to} style={{
                            background: '#111827',
                            border: '1px solid rgba(255,255,255,0.06)',
                            borderRadius: 12,
                            padding: '20px',
                            display: 'block',
                            transition: 'border-color 0.2s',
                        }}
                            onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(239,68,68,0.4)'}
                            onMouseLeave={e => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'}
                        >
                            <div style={{ fontSize: 28, marginBottom: 10 }}>{card.icon}</div>
                            <div style={{ fontWeight: 700, fontSize: 14, color: '#f1f5f9', marginBottom: 4 }}>
                                {card.label}
                            </div>
                            <div style={{ fontSize: 12, color: '#475569' }}>{card.desc}</div>
                            <div style={{ marginTop: 12, fontSize: 12, color: '#ef4444', fontWeight: 500 }}>
                                Explore →
                            </div>
                        </Link>
                    ))}
                </div>
            </div>

            {/* ── Team footer ────────────────────────────────────────────── */}
            <div style={{
                borderTop: '1px solid rgba(255,255,255,0.06)',
                padding: '32px 20px',
                textAlign: 'center',
            }}>
                <div style={{ fontSize: 12, color: '#475569', marginBottom: 10 }}>
                    Research Team
                </div>
                <div style={{
                    display: 'flex', gap: 8,
                    justifyContent: 'center', flexWrap: 'wrap',
                    marginBottom: 8,
                }}>
                    {paper.team.map(name => (
                        <span key={name} style={{
                            background: '#111827',
                            border: '1px solid rgba(255,255,255,0.06)',
                            borderRadius: 20,
                            padding: '4px 14px',
                            fontSize: 12,
                            color: '#94a3b8',
                        }}>
                            {name}
                        </span>
                    ))}
                </div>
                <div style={{ fontSize: 12, color: '#475569' }}>
                    Guided by {paper.guide}
                </div>
            </div>

        </div>
    );
}

// ── Small helper components ───────────────────────────────────────────
function Kpi({ label, value, color }) {
    return (
        <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 20, fontWeight: 800, color, lineHeight: 1 }}>
                {value}
            </div>
            <div style={{ fontSize: 10, color: '#475569', marginTop: 3 }}>
                {label}
            </div>
        </div>
    );
}

function Divider() {
    return (
        <div style={{ width: 1, background: 'rgba(255,255,255,0.08)', alignSelf: 'stretch' }} />
    );
}

const navCards = [
    {
        to: '/overview',
        icon: '📊',
        label: 'Dataset Overview',
        desc: '21K records · 36 states · 188 features',
    },
    {
        to: '/trends',
        icon: '📈',
        label: 'Crime Trends',
        desc: '2017–2022 patterns by group and state',
    },
    {
        to: '/cities',
        icon: '🏙️',
        label: 'Metro Cities',
        desc: '34 major cities · 2021–2023 analysis',
    },
];