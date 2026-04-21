import React, { useState } from 'react';
import { Page, PageTitle, Card, SectionTitle, TabBar } from '../components/UI';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell
} from 'recharts';

// ── Exact paper numbers — Table 5 of submitted paper ─────────────────────
const MODELS = [
    {
        name: 'SARIMA', type: 'Statistical',
        mae: 166.61, rmse: 220.34, r2: 0.000,
        fairness_gap: 20.85, fairness_ratio: 1.17, time: '0.07m',
        color: '#94a3b8', ours: false,
    },
    {
        name: 'Prophet', type: 'Statistical',
        mae: 135.46, rmse: 198.19, r2: 0.191,
        fairness_gap: 17.60, fairness_ratio: 1.22, time: '1.41m',
        color: '#64748b', ours: false,
    },
    {
        name: 'Random Forest', type: 'Ensemble ML',
        mae: 2.14, rmse: 5.63, r2: 0.9993,
        fairness_gap: 3.29, fairness_ratio: 12.28, time: '0.07m',
        color: '#3b82f6', ours: false,
    },
    {
        name: 'XGBoost', type: 'Ensemble ML',
        mae: 1.83, rmse: 4.13, r2: 0.9996,
        fairness_gap: 2.91, fairness_ratio: 10.81, time: '0.05m',
        color: '#06b6d4', ours: false,
    },
    {
        name: 'CNN-LSTM', type: 'Deep Learning',
        mae: 23.83, rmse: 57.33, r2: 0.9419,
        fairness_gap: 55.95, fairness_ratio: 27.04, time: '2.63m',
        color: '#8b5cf6', ours: false,
    },
    {
        name: 'Transformer', type: 'Deep Learning',
        mae: 5.64, rmse: 10.27, r2: 0.9981,
        fairness_gap: 6.10, fairness_ratio: 3.51, time: '3.23m',
        color: '#ec4899', ours: false,
    },
    {
        name: 'FC-MT-LSTM (Ours)', type: 'Fairness-Aware DL',
        mae: 3.79, rmse: 9.83, r2: 0.9980,
        fairness_gap: 3.84, fairness_ratio: 3.26, time: '14.7m',
        color: '#f97316', ours: true,
    },
];

// Per-group results — Table 6
const GROUP_RESULTS = [
    { group: 'SC', mae: 2.46, rmse: 4.21, r2: 0.9730, count: 934, color: '#64B5F6' },
    { group: 'ST', mae: 1.70, rmse: 2.38, r2: 0.9877, count: 890, color: '#81C784' },
    { group: 'Women', mae: 5.39, rmse: 8.94, r2: 0.9993, count: 933, color: '#FF7043' },
    { group: 'Children', mae: 5.53, rmse: 12.17, r2: 0.9974, count: 931, color: '#FFB74D' },
];

const TABS = ['Comparison Table', 'MAE Chart', 'Fairness Chart', 'Per-Group'];

const cell = (highlight) => ({
    padding: '9px 12px',
    textAlign: 'right',
    fontSize: 12,
    color: highlight ? '#f97316' : '#94a3b8',
    fontWeight: highlight ? 700 : 400,
    borderTop: '1px solid rgba(255,255,255,0.05)',
});

export default function Models() {
    const [tab, setTab] = useState('Comparison Table');
    const [sortBy, setSortBy] = useState('mae');

    const sorted = [...MODELS].sort((a, b) => {
        if (sortBy === 'mae') return a.mae - b.mae;
        if (sortBy === 'r2') return b.r2 - a.r2;
        if (sortBy === 'ratio') return a.fairness_ratio - b.fairness_ratio;
        return 0;
    });

    // Data subsets for charts
    const mlModels = MODELS.filter(m => m.mae < 30);   // exclude SARIMA/Prophet
    const sortedFairness = [...MODELS].sort((a, b) => a.fairness_ratio - b.fairness_ratio);

    return (
        <Page>
            <PageTitle
                title="📊 Model Comparison"
                sub="FC-MT-LSTM V5 Enhanced vs 6 baselines · Table 5 from submitted paper"
            />

            <TabBar tabs={TABS} active={tab} onChange={setTab} />

            {/* ── Tab 1: Comparison Table ──────────────────────────────── */}
            {tab === 'Comparison Table' && (
                <Card>
                    <SectionTitle
                        title="Full Model Comparison — Table 5"
                        sub="Paper: Fairness-Constrained Multi-Task Learning for Crime Prediction"
                    />
                    <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
                        {[['mae', 'Sort by MAE'], ['r2', 'Sort by R²'], ['ratio', 'Sort by Fairness']].map(([k, l]) => (
                            <button key={k} onClick={() => setSortBy(k)} style={{
                                padding: '4px 12px', borderRadius: 6, border: 'none', cursor: 'pointer',
                                fontSize: 11, fontWeight: 600,
                                background: sortBy === k ? '#f97316' : 'rgba(255,255,255,0.07)',
                                color: sortBy === k ? '#fff' : '#94a3b8',
                            }}>{l}</button>
                        ))}
                    </div>

                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                            <thead>
                                <tr style={{ background: '#0b1120' }}>
                                    {['Model', 'Type', 'MAE ↓', 'RMSE ↓', 'R² ↑', 'F.Gap ↓', 'F.Ratio ↓', 'Time'].map(h => (
                                        <th key={h} style={{
                                            padding: '10px 12px',
                                            textAlign: h === 'Model' || h === 'Type' ? 'left' : 'right',
                                            color: '#64748b', fontWeight: 600, fontSize: 10,
                                            textTransform: 'uppercase', letterSpacing: 0.5,
                                            borderBottom: '1px solid rgba(255,255,255,0.08)',
                                        }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {sorted.map((m) => (
                                    <tr key={m.name} style={{
                                        background: m.ours ? 'rgba(249,115,22,0.06)' : 'transparent',
                                    }}>
                                        <td style={{
                                            padding: '9px 12px', fontSize: 12,
                                            color: m.ours ? '#f97316' : '#f1f5f9',
                                            fontWeight: m.ours ? 700 : 500,
                                            borderTop: '1px solid rgba(255,255,255,0.05)',
                                        }}>
                                            {m.ours ? '⭐ ' : ''}{m.name}
                                        </td>
                                        <td style={{
                                            padding: '9px 12px', fontSize: 11, color: '#64748b',
                                            borderTop: '1px solid rgba(255,255,255,0.05)',
                                        }}>{m.type}</td>
                                        <td style={cell(m.ours)}>{m.mae.toFixed(2)}</td>
                                        <td style={cell(m.ours)}>{m.rmse.toFixed(2)}</td>
                                        <td style={cell(m.ours)}>{m.r2.toFixed(4)}</td>
                                        <td style={cell(m.ours)}>{m.fairness_gap.toFixed(2)}</td>
                                        <td style={cell(m.ours)}>{m.fairness_ratio.toFixed(2)}</td>
                                        <td style={cell(m.ours)}>{m.time}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div style={{ marginTop: 12, fontSize: 10, color: '#475569' }}>
                        ↓ = lower is better &nbsp;·&nbsp; ↑ = higher is better &nbsp;·&nbsp;
                        F.Ratio = max(group MAE) / min(group MAE) &nbsp;·&nbsp;
                        Bold orange = FC-MT-LSTM (our model)
                    </div>
                </Card>
            )}

            {/* ── Tab 2: MAE Chart ──────────────────────────────────────── */}
            {tab === 'MAE Chart' && (
                <Card>
                    <SectionTitle
                        title="MAE Comparison (lower is better)"
                        sub="Statistical models excluded from chart scale — their MAE is 100× higher"
                    />
                    <ResponsiveContainer width="100%" height={320}>
                        <BarChart
                            data={mlModels}
                            margin={{ top: 10, right: 20, left: 0, bottom: 60 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }}
                                angle={-30} textAnchor="end" interval={0} />
                            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                            <Tooltip
                                contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                                labelStyle={{ color: '#f1f5f9' }}
                                formatter={(v) => [v.toFixed(2), 'MAE']}
                            />
                            <Bar dataKey="mae" radius={[4, 4, 0, 0]}
                                label={{ position: 'top', fill: '#94a3b8', fontSize: 10, formatter: v => v.toFixed(2) }}>
                                {mlModels.map((m) => (
                                    <Cell key={m.name} fill={m.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <div style={{ marginTop: 8, fontSize: 10, color: '#475569', textAlign: 'center' }}>
                        SARIMA (MAE=166.61) and Prophet (MAE=135.46) excluded — too large for scale
                    </div>
                </Card>
            )}

            {/* ── Tab 3: Fairness Chart ─────────────────────────────────── */}
            {tab === 'Fairness Chart' && (
                <Card>
                    <SectionTitle
                        title="Fairness Ratio Comparison (lower = more fair)"
                        sub="Ratio = max group MAE / min group MAE · 1.0 = perfect parity"
                    />
                    <ResponsiveContainer width="100%" height={340}>
                        <BarChart
                            data={sortedFairness}
                            margin={{ top: 10, right: 20, left: 0, bottom: 60 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }}
                                angle={-30} textAnchor="end" interval={0} />
                            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                            <Tooltip
                                contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                                formatter={(v) => [v.toFixed(2), 'Fairness Ratio']}
                            />
                            <Bar dataKey="fairness_ratio" radius={[4, 4, 0, 0]}
                                label={{ position: 'top', fill: '#94a3b8', fontSize: 10, formatter: v => v.toFixed(2) }}>
                                {sortedFairness.map((m) => (
                                    <Cell key={m.name} fill={m.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <div style={{ marginTop: 8, fontSize: 10, color: '#475569', textAlign: 'center' }}>
                        FC-MT-LSTM: Ratio=3.26 — 88% better than CNN-LSTM (27.04), 73% better than RF (12.28)
                    </div>
                </Card>
            )}

            {/* ── Tab 4: Per-Group ──────────────────────────────────────── */}
            {tab === 'Per-Group' && (
                <>
                    <Card style={{ marginBottom: 16 }}>
                        <SectionTitle
                            title="FC-MT-LSTM V5 Per-Group Results — Table 6"
                            sub="All groups show strong performance · Fairness Ratio = 3.26"
                        />
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                            {GROUP_RESULTS.map(g => (
                                <div key={g.group} style={{
                                    background: '#0b1120', borderRadius: 10, padding: 14,
                                    border: `1px solid ${g.color}22`,
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                                        <span style={{ color: g.color, fontWeight: 700, fontSize: 14 }}>{g.group}</span>
                                        <span style={{ color: '#475569', fontSize: 11 }}>{g.count.toLocaleString()} test records</span>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                                        {[['MAE', g.mae.toFixed(2)], ['RMSE', g.rmse.toFixed(2)], ['R²', g.r2.toFixed(4)]].map(([label, val]) => (
                                            <div key={label} style={{ textAlign: 'center' }}>
                                                <div style={{ fontSize: 18, fontWeight: 800, color: g.color }}>{val}</div>
                                                <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>{label}</div>
                                            </div>
                                        ))}
                                    </div>
                                    <div style={{ marginTop: 10, background: '#111827', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                                        <div style={{
                                            width: `${(g.mae / 6.0) * 100}%`,
                                            height: '100%', background: g.color, borderRadius: 4, opacity: 0.8,
                                        }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>

                    <Card>
                        <SectionTitle
                            title="Key Insight"
                            sub="Why separate Women and Children decoders matter"
                        />
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                            {[
                                ['Women decoder 2× capacity', 'Women crimes span 43 categories — largest, most complex group. 128 hidden units vs 64 for SC/ST.'],
                                ['Children decoder 2× capacity', 'Children crimes cover 52 POCSO categories — highest category count. Needs extra model capacity.'],
                                ['Shared encoder cross-learning', 'ST group (lowest data) benefits from shared representations learned from Women group (largest data).'],
                                ['λ = 1.5 fairness weight', 'Higher fairness penalty forces the model to reduce group disparity aggressively without hurting accuracy.'],
                            ].map(([title, desc]) => (
                                <div key={title} style={{
                                    background: '#0b1120', borderRadius: 8, padding: 12,
                                    borderLeft: '3px solid #f97316',
                                }}>
                                    <div style={{ fontSize: 12, fontWeight: 600, color: '#f97316', marginBottom: 4 }}>{title}</div>
                                    <div style={{ fontSize: 11, color: '#64748b' }}>{desc}</div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </>
            )}
        </Page>
    );
}