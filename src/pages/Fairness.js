import React, { useState } from 'react';
import { Page, PageTitle, Card, SectionTitle, StatCard, TabBar } from '../components/UI';
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend
} from 'recharts';
import data from '../data/predictions.json';



// ── Exact paper numbers ────────────────────────────────────────────────────
const GROUP_METRICS = {
    SC: { mae: 2.46, rmse: 4.21, r2: 0.9730, count: 934, color: '#64B5F6' },
    ST: { mae: 1.70, rmse: 2.38, r2: 0.9877, count: 890, color: '#81C784' },
    Women: { mae: 5.39, rmse: 8.94, r2: 0.9993, count: 933, color: '#FF7043' },
    Children: { mae: 5.53, rmse: 12.17, r2: 0.9974, count: 931, color: '#FFB74D' },
};

const OVERALL = { mae: 3.79, rmse: 9.83, r2: 0.9980, fairness_ratio: 3.26, fairness_gap: 3.84 };

// Baseline fairness ratios from Table 5
const BASELINE_FAIRNESS = [
    { name: 'SARIMA', fairness_ratio: 1.17, color: '#94a3b8' },
    { name: 'Prophet', fairness_ratio: 1.22, color: '#64748b' },
    { name: 'Transformer', fairness_ratio: 3.51, color: '#ec4899' },
    { name: 'FC-MT-LSTM', fairness_ratio: 3.26, color: '#f97316' },
    { name: 'Random Forest', fairness_ratio: 12.28, color: '#3b82f6' },
    { name: 'XGBoost', fairness_ratio: 10.81, color: '#06b6d4' },
    { name: 'CNN-LSTM', fairness_ratio: 27.04, color: '#8b5cf6' },
];

const radarData = ['MAE (inv)', 'RMSE (inv)', 'R²'].map(metric => {
    const row = { metric };
    Object.entries(GROUP_METRICS).forEach(([g, m]) => {
        if (metric === 'MAE (inv)') row[g] = Math.max(0, 1 - m.mae / 10) * 100;
        if (metric === 'RMSE (inv)') row[g] = Math.max(0, 1 - m.rmse / 20) * 100;
        if (metric === 'R²') row[g] = m.r2 * 100;
    });
    return row;
});

const TABS = ['Overview', 'Per-Group Detail', 'Baseline Comparison'];

export default function Fairness() {
    const [tab, setTab] = useState('Overview');

    const maes = Object.values(GROUP_METRICS).map(m => m.mae);
    const maxMAE = Math.max(...maes);
    const minMAE = Math.min(...maes);

    return (
        <Page>
            <PageTitle
                title="⚖️ Fairness Analysis"
                sub="FC-MT-LSTM V5 · Fairness Ratio 3.26 · 62% better than CNN-LSTM"
            />

            {/* Key stats */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
                <StatCard value={OVERALL.fairness_ratio} label="Fairness Ratio" color="#f97316" />
                <StatCard value={OVERALL.fairness_gap} label="Fairness Gap" color="#a78bfa" />
                <StatCard value={Math.min(...Object.values(GROUP_METRICS).map(m => m.mae)).toFixed(2)} label="Best Group MAE" color="#81C784" />
                <StatCard value={Math.max(...Object.values(GROUP_METRICS).map(m => m.mae)).toFixed(2)} label="Worst Group MAE" color="#FFB74D" />
            </div>

            <TabBar tabs={TABS} active={tab} onChange={setTab} />

            {/* ── Overview ─────────────────────────────────────────────────── */}
            {tab === 'Overview' && (
                <>
                    <Card style={{ marginBottom: 16 }}>
                        <SectionTitle
                            title="What is Fairness Ratio?"
                            sub="max(group MAE) / min(group MAE) · closer to 1.0 = more fair"
                        />
                        <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.7 }}>
                            <p>The Fairness Ratio measures how much worse the model is for the worst-predicted group compared to the best-predicted group.</p>
                            <p>Our FC-MT-LSTM achieves <strong style={{ color: '#f97316' }}>Fairness Ratio = 3.26</strong>, meaning the Children group (MAE=5.53) is only 3.26x worse than the ST group (MAE=1.70). This is achieved through:</p>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
                            {[
                                ['Pairwise Fairness Loss', 'λ=1.5 penalty on inter-group MAE disparity during training', '#f97316'],
                                ['2x Capacity Decoders', 'Women and Children decoders have 128 hidden units vs 64 for SC/ST', '#FF7043'],
                                ['Shared Encoder', 'ST group learns from Women group patterns through shared representation', '#64B5F6'],
                                ['AdamW + Warmup', '5-epoch LR warmup + CosineAnnealing prevents overfitting to majority groups', '#81C784'],
                            ].map(([title, desc, color]) => (
                                <div key={title} style={{
                                    background: '#0b1120', borderRadius: 8, padding: 12,
                                    borderLeft: `3px solid ${color}`,
                                }}>
                                    <div style={{ fontSize: 12, fontWeight: 600, color, marginBottom: 3 }}>{title}</div>
                                    <div style={{ fontSize: 11, color: '#64748b' }}>{desc}</div>
                                </div>
                            ))}
                        </div>
                    </Card>

                    <Card>
                        <SectionTitle
                            title="Radar Chart — Per-Group Performance"
                            sub="All 4 groups across MAE, RMSE, R² — higher = better"
                        />
                        <ResponsiveContainer width="100%" height={300}>
                            <RadarChart data={radarData}>
                                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 9 }} domain={[0, 100]} />
                                {Object.entries(GROUP_METRICS).map(([g, m]) => (
                                    <Radar key={g} name={g} dataKey={g}
                                        stroke={m.color} fill={m.color} fillOpacity={0.12}
                                        strokeWidth={2} />
                                ))}
                                <Legend formatter={(v) => <span style={{ color: GROUP_METRICS[v]?.color, fontSize: 11 }}>{v}</span>} />
                                <Tooltip
                                    contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                                    formatter={(v) => [`${v.toFixed(1)}%`]}
                                />
                            </RadarChart>
                        </ResponsiveContainer>
                    </Card>
                </>
            )}

            {/* ── Per-Group Detail ─────────────────────────────────────────── */}
            {tab === 'Per-Group Detail' && (
                <Card>
                    <SectionTitle
                        title="Per-Group Metrics — Table 6"
                        sub="FC-MT-LSTM V5 on 3,688 test records (2022)"
                    />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                        {Object.entries(GROUP_METRICS).map(([group, m]) => (
                            <div key={group} style={{
                                background: '#0b1120', borderRadius: 10, padding: 16,
                                border: `1px solid ${m.color}22`,
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                                    <span style={{ color: m.color, fontWeight: 700, fontSize: 15 }}>{group}</span>
                                    <span style={{ color: '#475569', fontSize: 11 }}>{m.count.toLocaleString()} records</span>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 12 }}>
                                    {[['MAE', m.mae.toFixed(2)], ['RMSE', m.rmse.toFixed(2)], ['R²', m.r2.toFixed(4)]].map(([l, v]) => (
                                        <div key={l} style={{ textAlign: 'center', background: '#111827', borderRadius: 8, padding: 10 }}>
                                            <div style={{ fontSize: 20, fontWeight: 800, color: m.color }}>{v}</div>
                                            <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>{l}</div>
                                        </div>
                                    ))}
                                </div>

                                {/* MAE bar relative to max */}
                                <div style={{ fontSize: 10, color: '#475569', marginBottom: 4 }}>
                                    MAE relative to worst group ({maxMAE.toFixed(2)})
                                </div>
                                <div style={{ background: '#111827', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                                    <div style={{
                                        width: `${(m.mae / maxMAE) * 100}%`,
                                        height: '100%', background: m.color, borderRadius: 4, opacity: 0.85,
                                        transition: 'width 0.5s ease',
                                    }} />
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{
                        marginTop: 16, padding: 14, background: '#0b1120', borderRadius: 10,
                        border: '1px solid rgba(249,115,22,0.2)'
                    }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: '#f97316', marginBottom: 6 }}>
                            Fairness Ratio Calculation
                        </div>
                        <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.8 }}>
                            Best group (ST): MAE = {minMAE.toFixed(2)}<br />
                            Worst group (Children): MAE = {maxMAE.toFixed(2)}<br />
                            Fairness Ratio = {maxMAE.toFixed(2)} / {minMAE.toFixed(2)} = <strong style={{ color: '#f97316' }}>3.26</strong><br />
                            Fairness Gap = {maxMAE.toFixed(2)} - {minMAE.toFixed(2)} = <strong style={{ color: '#f97316' }}>3.84</strong>
                        </div>
                    </div>
                </Card>
            )}

            {/* ── Baseline Comparison ──────────────────────────────────────── */}
            {tab === 'Baseline Comparison' && (
                <Card>
                    <SectionTitle
                        title="Fairness Ratio — All Models"
                        sub="Lower = more fair · FC-MT-LSTM achieves best accuracy-fairness balance"
                    />
                    <ResponsiveContainer width="100%" height={320}>
                        <BarChart
                            data={[...BASELINE_FAIRNESS].sort((a, b) => a.fairness_ratio - b.fairness_ratio)}
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
                                fill="#64748b"
                                label={{
                                    position: 'top', fill: '#94a3b8', fontSize: 10,
                                    formatter: v => v.toFixed(2)
                                }} />
                        </BarChart>
                    </ResponsiveContainer>

                    <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {[
                            ['vs Random Forest (12.28)', `${(((12.28 - 3.26) / 12.28) * 100).toFixed(0)}% improvement`, '#3b82f6'],
                            ['vs XGBoost (10.81)', `${(((10.81 - 3.26) / 10.81) * 100).toFixed(0)}% improvement`, '#06b6d4'],
                            ['vs CNN-LSTM (27.04)', `${(((27.04 - 3.26) / 27.04) * 100).toFixed(0)}% improvement`, '#8b5cf6'],
                            ['vs Transformer (3.51)', `${(((3.51 - 3.26) / 3.51) * 100).toFixed(0)}% improvement`, '#ec4899'],
                        ].map(([label, pct, color]) => (
                            <div key={label} style={{
                                display: 'flex', justifyContent: 'space-between',
                                background: '#0b1120', borderRadius: 8, padding: '8px 14px',
                                borderLeft: `3px solid ${color}`,
                            }}>
                                <span style={{ fontSize: 12, color: '#94a3b8' }}>{label}</span>
                                <span style={{ fontSize: 12, fontWeight: 700, color: '#f97316' }}>{pct}</span>
                            </div>
                        ))}
                    </div>

                    <div style={{ marginTop: 12, fontSize: 10, color: '#475569' }}>
                        Note: SARIMA (1.17) and Prophet (1.22) have better fairness ratios but R²=0.000 and 0.191 —
                        they achieve fairness by being equally wrong for all groups, which is not useful.
                        FC-MT-LSTM achieves fairness while maintaining R²=0.9980.
                    </div>
                </Card>
            )}
        </Page>
    );
}