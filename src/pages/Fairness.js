import React, { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, Cell, CartesianGrid,
    RadarChart, Radar, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import { Page, PageTitle, Card, SectionTitle, TabBar } from '../components/UI';

const GC = {
    SC: '#64B5F6', ST: '#81C784',
    Women: '#FF7043', Children: '#FFB74D',
};

// ── Exact numbers from paper Table 5 ─────────────────────────────────
const GROUP_METRICS = [
    { group: 'SC', mae: 2.08, rmse: 3.15, r2: 0.9673, color: '#64B5F6' },
    { group: 'ST', mae: 1.40, rmse: 1.54, r2: 0.0000, color: '#81C784' },
    { group: 'Women', mae: 7.89, rmse: 10.81, r2: 0.9981, color: '#FF7043' },
    { group: 'Children', mae: 14.01, rmse: 29.12, r2: 0.9852, color: '#FFB74D' },
];

// Fairness ratios across all models
const FAIRNESS_DATA = [
    { model: 'SARIMA', ratio: 1.17, gap: 20.85, color: '#64B5F6' },
    { model: 'Prophet', ratio: 1.22, gap: 17.60, color: '#60A5FA' },
    { model: 'Random Forest', ratio: 4.74, gap: 1.08, color: '#FCD34D' },
    { model: 'XGBoost', ratio: 3.73, gap: 0.81, color: '#F59E0B' },
    { model: 'FC-MT-LSTM', ratio: 1.99, gap: 12.61, color: '#A78BFA', is_ours: true },
];

function CustomTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: '#111827',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 8, padding: '8px 12px',
        }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{label}</div>
            {payload.map((p, i) => (
                <div key={i} style={{ color: p.color || '#f1f5f9', fontSize: 12, fontWeight: 600 }}>
                    {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
                </div>
            ))}
        </div>
    );
}

export default function Fairness() {
    const [tab, setTab] = useState('Per Group');

    // Radar data — normalise MAE so radar looks meaningful
    const maxMAE = Math.max(...GROUP_METRICS.map(g => g.mae));
    const radarData = GROUP_METRICS.map(g => ({
        group: g.group,
        MAE: parseFloat(((g.mae / maxMAE) * 100).toFixed(1)),
        RMSE: parseFloat(((g.rmse / Math.max(...GROUP_METRICS.map(x => x.rmse))) * 100).toFixed(1)),
        'R² Score': parseFloat((g.r2 * 100).toFixed(1)),
    }));

    return (
        <Page>
            <PageTitle
                title="⚖️ Fairness Analysis"
                sub="FC-MT-LSTM per-group breakdown · Paper Table 5 · Why fairness matters"
            />

            {/* ── What is fairness banner ───────────────────────────────── */}
            <div style={{
                background: 'rgba(52,211,153,0.06)',
                border: '1px solid rgba(52,211,153,0.2)',
                borderRadius: 14, padding: 20, marginBottom: 20,
            }}>
                <div style={{
                    fontWeight: 700, color: '#34d399',
                    fontSize: 14, marginBottom: 8,
                }}>
                    💡 What is Fairness in ML?
                </div>
                <p style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.7, margin: 0 }}>
                    A model is <b style={{ color: '#f1f5f9' }}>unfair</b> if it predicts
                    well for one group but poorly for another.
                    For example — if a model predicts crimes against Women with MAE=2
                    but crimes against Children with MAE=50, policymakers would
                    over-allocate resources to Women and under-protect Children.
                    The <b style={{ color: '#34d399' }}>Fairness Ratio</b> measures this:
                    it is the ratio of the worst group's error to the best group's error.
                    A ratio of <b style={{ color: '#34d399' }}>1.0 = perfectly fair</b>.
                    FC-MT-LSTM achieves <b style={{ color: '#a78bfa' }}>1.99</b> vs
                    Random Forest's <b style={{ color: '#f87171' }}>4.74</b>.
                </p>
            </div>

            <TabBar
                tabs={['Per Group', 'Fairness Ratio', 'Radar', 'Why It Matters']}
                active={tab}
                onChange={setTab}
            />

            {/* ── Tab 1 : Per Group ─────────────────────────────────────── */}
            {tab === 'Per Group' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

                    {/* group cards */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                        gap: 12,
                    }}>
                        {GROUP_METRICS.map(g => (
                            <Card key={g.group} style={{ borderLeft: `3px solid ${g.color}` }}>
                                <div style={{
                                    fontWeight: 700, color: g.color,
                                    fontSize: 15, marginBottom: 12,
                                }}>
                                    {g.group}
                                </div>
                                <div style={{
                                    display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8,
                                }}>
                                    {[['MAE', g.mae], ['RMSE', g.rmse], ['R²', g.r2]].map(([l, v]) => (
                                        <div key={l} style={{
                                            background: '#0b1120', borderRadius: 6,
                                            padding: '8px', textAlign: 'center',
                                        }}>
                                            <div style={{ fontSize: 15, fontWeight: 800, color: g.color }}>
                                                {v}
                                            </div>
                                            <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                                                {l}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </Card>
                        ))}
                    </div>

                    {/* MAE bar chart per group */}
                    <Card>
                        <SectionTitle
                            title="MAE per Group — FC-MT-LSTM"
                            sub="From paper Table 5 · Children hardest to predict"
                        />
                        <ResponsiveContainer width="100%" height={260}>
                            <BarChart
                                data={GROUP_METRICS}
                                margin={{ left: 0, right: 10, top: 5, bottom: 10 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="group" tick={{ fontSize: 11, fill: '#64748b' }} />
                                <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                                <Tooltip content={<CustomTooltip />} />
                                <Bar dataKey="mae" radius={[5, 5, 0, 0]} name="MAE">
                                    {GROUP_METRICS.map((g, i) => (
                                        <Cell key={i} fill={g.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </Card>

                    {/* RMSE bar chart */}
                    <Card>
                        <SectionTitle
                            title="RMSE per Group — FC-MT-LSTM"
                            sub="Children also have highest RMSE — more variance in predictions"
                        />
                        <ResponsiveContainer width="100%" height={240}>
                            <BarChart
                                data={GROUP_METRICS}
                                margin={{ left: 0, right: 10, top: 5, bottom: 10 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="group" tick={{ fontSize: 11, fill: '#64748b' }} />
                                <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                                <Tooltip content={<CustomTooltip />} />
                                <Bar dataKey="rmse" radius={[5, 5, 0, 0]} name="RMSE">
                                    {GROUP_METRICS.map((g, i) => (
                                        <Cell key={i} fill={g.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </Card>
                </div>
            )}

            {/* ── Tab 2 : Fairness Ratio ────────────────────────────────── */}
            {tab === 'Fairness Ratio' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    <Card>
                        <SectionTitle
                            title="Fairness Ratio Across Models"
                            sub="Ratio = worst group MAE ÷ best group MAE · closer to 1.0 = fairer"
                        />
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart
                                data={FAIRNESS_DATA}
                                margin={{ left: 0, right: 10, top: 5, bottom: 60 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis
                                    dataKey="model"
                                    tick={{ fontSize: 9, fill: '#64748b' }}
                                    angle={-30}
                                    textAnchor="end"
                                    interval={0}
                                />
                                <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                                <Tooltip content={<CustomTooltip />} />
                                <Bar dataKey="ratio" radius={[5, 5, 0, 0]} name="Fairness Ratio">
                                    {FAIRNESS_DATA.map((m, i) => (
                                        <Cell
                                            key={i}
                                            fill={m.is_ours ? '#a78bfa' : m.color}
                                            opacity={0.9}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>

                        {/* annotation */}
                        <div style={{
                            marginTop: 12,
                            display: 'flex', flexDirection: 'column', gap: 8,
                        }}>
                            {FAIRNESS_DATA.map(m => (
                                <div key={m.model} style={{
                                    display: 'grid',
                                    gridTemplateColumns: '130px 1fr auto',
                                    gap: 10, alignItems: 'center',
                                }}>
                                    <span style={{
                                        fontSize: 11,
                                        color: m.is_ours ? '#ddd6fe' : '#94a3b8',
                                        fontWeight: m.is_ours ? 700 : 400,
                                    }}>
                                        {m.is_ours ? '★ ' : ''}{m.model}
                                    </span>
                                    <div style={{
                                        background: '#0b1120', borderRadius: 3,
                                        height: 6, overflow: 'hidden',
                                    }}>
                                        <div style={{
                                            width: `${(m.ratio / 5) * 100}%`,
                                            height: '100%',
                                            background: m.is_ours ? '#a78bfa' : m.color,
                                            borderRadius: 3,
                                        }} />
                                    </div>
                                    <span style={{
                                        fontSize: 12, fontWeight: 700,
                                        color: m.is_ours ? '#a78bfa' : m.color,
                                        minWidth: 32, textAlign: 'right',
                                    }}>
                                        {m.ratio}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </Card>

                    {/* insight card */}
                    <Card>
                        <SectionTitle
                            title="Key Insight"
                            sub="What these fairness ratios mean in practice"
                        />
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                            {[
                                {
                                    model: 'Random Forest (4.74×)',
                                    color: '#FCD34D',
                                    insight: 'The worst-served group gets nearly 5× higher prediction error. If deployed for resource allocation, one group would be severely under-protected.',
                                },
                                {
                                    model: 'XGBoost (3.73×)',
                                    color: '#F59E0B',
                                    insight: 'Still 3.7× disparity. Despite excellent overall MAE of 1.83, one group always bears a disproportionate error burden.',
                                },
                                {
                                    model: 'FC-MT-LSTM (1.99×) ★',
                                    color: '#A78BFA',
                                    insight: 'Our model limits the worst group to only 2× the best group\'s error — nearly halving RF\'s disparity while keeping R²=0.9922.',
                                },
                            ].map(b => (
                                <div key={b.model} style={{
                                    display: 'flex', gap: 12, padding: 12,
                                    background: '#0b1120', borderRadius: 8,
                                    border: `1px solid ${b.color}22`,
                                }}>
                                    <div style={{
                                        width: 3, borderRadius: 2,
                                        background: b.color, flexShrink: 0,
                                    }} />
                                    <div>
                                        <div style={{
                                            fontWeight: 700, color: b.color,
                                            fontSize: 12, marginBottom: 4,
                                        }}>
                                            {b.model}
                                        </div>
                                        <p style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6, margin: 0 }}>
                                            {b.insight}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>
            )}

            {/* ── Tab 3 : Radar ─────────────────────────────────────────── */}
            {tab === 'Radar' && (
                <Card>
                    <SectionTitle
                        title="Per-Group Performance Radar"
                        sub="FC-MT-LSTM · All 4 groups · Normalised scores"
                    />
                    <ResponsiveContainer width="100%" height={340}>
                        <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                            <PolarGrid stroke="rgba(255,255,255,0.08)" />
                            <PolarAngleAxis
                                dataKey="group"
                                tick={{ fontSize: 12, fill: '#94a3b8', fontWeight: 600 }}
                            />
                            <PolarRadiusAxis
                                angle={30} domain={[0, 100]}
                                tick={{ fontSize: 9, fill: '#475569' }}
                            />
                            <Radar
                                name="MAE (normalised)"
                                dataKey="MAE"
                                stroke="#ef4444" fill="#ef4444" fillOpacity={0.15}
                            />
                            <Radar
                                name="RMSE (normalised)"
                                dataKey="RMSE"
                                stroke="#f97316" fill="#f97316" fillOpacity={0.10}
                            />
                            <Radar
                                name="R² Score"
                                dataKey="R² Score"
                                stroke="#34d399" fill="#34d399" fillOpacity={0.15}
                            />
                            <Tooltip content={<CustomTooltip />} />
                        </RadarChart>
                    </ResponsiveContainer>

                    {/* radar legend */}
                    <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 8 }}>
                        {[
                            ['MAE (norm)', '#ef4444'],
                            ['RMSE (norm)', '#f97316'],
                            ['R² Score', '#34d399'],
                        ].map(([l, c]) => (
                            <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
                                <span style={{ fontSize: 11, color: '#64748b' }}>{l}</span>
                            </div>
                        ))}
                    </div>

                    <div style={{
                        marginTop: 14, background: '#0b1120',
                        borderRadius: 8, padding: 12,
                        fontSize: 12, color: '#64748b', lineHeight: 1.6,
                    }}>
                        📌 <b style={{ color: '#94a3b8' }}>Note:</b> ST group shows R²=0.0
                        because test data for ST in 2022 had near-zero variance
                        — the model predicted constant values for a nearly constant target.
                        This is a data characteristic, not a model failure.
                    </div>
                </Card>
            )}

            {/* ── Tab 4 : Why it matters ────────────────────────────────── */}
            {tab === 'Why It Matters' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    <Card>
                        <SectionTitle
                            title="Real-World Impact of Unfair Predictions"
                            sub="Why fairness in crime prediction is not just a metric"
                        />
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                            {[
                                {
                                    icon: '🏛️',
                                    title: 'Policy Decisions',
                                    color: '#64B5F6',
                                    text: 'Government bodies use crime predictions to allocate police, shelters, and legal aid. If predictions systematically underestimate crimes against Children, that group receives fewer resources.',
                                },
                                {
                                    icon: '📊',
                                    title: 'Resource Allocation',
                                    color: '#FF7043',
                                    text: 'A Random Forest model with fairness ratio 4.74 means one group has nearly 5× worse predictions. Over a year, this compounds into large under-protection of the worst-served group.',
                                },
                                {
                                    icon: '⚖️',
                                    title: 'Constitutional Obligation',
                                    color: '#81C784',
                                    text: 'SC and ST groups have constitutional protections under Indian law. An unfair model could violate these obligations if used in judicial or administrative decisions.',
                                },
                                {
                                    icon: '🔬',
                                    title: 'Our Contribution',
                                    color: '#A78BFA',
                                    text: 'FC-MT-LSTM\'s fairness-constrained loss directly encodes the obligation of equal treatment into the training process — making fairness a first-class objective, not an afterthought.',
                                },
                            ].map(b => (
                                <div key={b.title} style={{
                                    display: 'flex', gap: 14, padding: 14,
                                    background: '#0b1120', borderRadius: 10,
                                    border: `1px solid ${b.color}22`,
                                }}>
                                    <div style={{ fontSize: 24, flexShrink: 0 }}>{b.icon}</div>
                                    <div>
                                        <div style={{
                                            fontWeight: 700, color: b.color,
                                            fontSize: 13, marginBottom: 5,
                                        }}>
                                            {b.title}
                                        </div>
                                        <p style={{
                                            fontSize: 12, color: '#64748b',
                                            lineHeight: 1.7, margin: 0,
                                        }}>
                                            {b.text}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>

                    {/* paper table 5 */}
                    <Card>
                        <SectionTitle
                            title="Paper Table 5 — FC-MT-LSTM Per-Group Breakdown"
                            sub="Exact numbers from the research paper"
                        />
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                                <thead>
                                    <tr style={{ background: '#0b1120' }}>
                                        {['Group', 'MAE ↓', 'RMSE ↓', 'R² ↑', 'Interpretation'].map(col => (
                                            <th key={col} style={{
                                                padding: '10px 12px', textAlign: 'left',
                                                color: '#64748b', fontSize: 11,
                                                fontWeight: 600, textTransform: 'uppercase',
                                                letterSpacing: 0.4,
                                            }}>
                                                {col}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {[
                                        { ...GROUP_METRICS[0], note: 'Strong performance — model learns SC patterns well' },
                                        { ...GROUP_METRICS[1], note: 'R²=0 due to near-zero variance in ST test data, not model failure' },
                                        { ...GROUP_METRICS[2], note: 'Excellent R²=0.9981 — Women crimes most predictable' },
                                        { ...GROUP_METRICS[3], note: 'Highest MAE — Children crimes most complex to predict' },
                                    ].map((g, i) => (
                                        <tr key={g.group} style={{
                                            borderTop: '1px solid rgba(255,255,255,0.05)',
                                        }}>
                                            <td style={{ padding: '10px 12px' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                                    <div style={{
                                                        width: 8, height: 8, borderRadius: '50%',
                                                        background: g.color, flexShrink: 0,
                                                    }} />
                                                    <span style={{ fontWeight: 700, color: g.color }}>{g.group}</span>
                                                </div>
                                            </td>
                                            <td style={{ padding: '10px 12px', color: '#f1f5f9', fontWeight: 600 }}>
                                                {g.mae}
                                            </td>
                                            <td style={{ padding: '10px 12px', color: '#f1f5f9', fontWeight: 600 }}>
                                                {g.rmse}
                                            </td>
                                            <td style={{ padding: '10px 12px', color: '#34d399', fontWeight: 600 }}>
                                                {g.r2}
                                            </td>
                                            <td style={{ padding: '10px 12px', color: '#64748b', fontSize: 11 }}>
                                                {g.note}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </Card>
                </div>
            )}
        </Page>
    );
}