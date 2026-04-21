import React, { useState, useMemo, useRef } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, CartesianGrid, Cell,
    ScatterChart, Scatter, ReferenceLine
} from 'recharts';
import {
    Page, PageTitle, Card, SectionTitle,
    TabBar, Select, StatCard
} from '../components/UI';
import data from '../data/predictions.json';

const GC = {
    SC: '#64B5F6', ST: '#81C784',
    Women: '#FF7043', Children: '#FFB74D'
};

// Normalise raw group labels from JSON → short display names
function normGroup(raw) {
    if (raw === 'Scheduled Castes') return 'SC';
    if (raw === 'Scheduled Tribes') return 'ST';
    return raw; // 'Women' and 'Children' already correct
}

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
                    {p.name}: {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}
                </div>
            ))}
        </div>
    );
}

// Stable jitter — computed once per component mount, not on every render
function useStableJitter(samples) {
    const ref = useRef(null);
    if (ref.current === null || ref.current.length !== samples.length) {
        const rng = mulberry32(12345);
        ref.current = samples.map(s => ({
            ...s,
            _jx: (rng() - 0.5) * 6,
            _jy: (rng() - 0.5) * 6,
        }));
    }
    return ref.current;
}

// Simple seeded PRNG so jitter is deterministic
function mulberry32(seed) {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

export default function Predictions() {
    const [tab, setTab] = useState('By State');
    const [group, setGroup] = useState('All');

    // Normalise group labels once
    const normSamples = useMemo(() =>
        data.samples.map(s => ({ ...s, group: normGroup(s.group) })),
        []
    );

    // Stable scatter data (jitter never changes)
    const scatterBase = useStableJitter(normSamples);

    // Filtered samples for the selected group
    const filteredSamples = useMemo(() =>
        group === 'All' ? normSamples : normSamples.filter(s => s.group === group),
        [group, normSamples]
    );

    // Scatter data filtered by group
    const scatterData = useMemo(() =>
        group === 'All'
            ? scatterBase
            : scatterBase.filter(s => s.group === group),
        [group, scatterBase]
    );

    return (
        <Page>
            <PageTitle
                title="🤖 FC-MT-LSTM Predictions"
                sub="Trained on 2017–2021 · Tested on 2022 · Fairness-Constrained Multi-Task LSTM"
            />

            {/* ── Model result banner ───────────────────────────────────── */}
            <div style={{
                background: 'linear-gradient(135deg, rgba(167,139,250,0.15), rgba(239,68,68,0.08))',
                border: '1px solid rgba(167,139,250,0.3)',
                borderRadius: 14, padding: 20, marginBottom: 20,
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                    <span style={{ fontSize: 20 }}>★</span>
                    <span style={{ fontWeight: 800, fontSize: 16, color: '#ddd6fe' }}>
                        FC-MT-LSTM V5 — Results
                    </span>
                    <span style={{
                        fontSize: 11, background: 'rgba(167,139,250,0.2)',
                        color: '#c4b5fd', borderRadius: 20,
                        padding: '2px 10px', fontWeight: 600,
                    }}>
                        Submitted Paper Numbers
                    </span>
                </div>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                    gap: 10,
                }}>
                    {[
                        ['MAE', data.overall.mae, '#a78bfa'],
                        ['RMSE', data.overall.rmse, '#a78bfa'],
                        ['R²', data.overall.r2, '#34d399'],
                        ['Fairness Ratio', data.overall.fairness_ratio, '#fbbf24'],
                        ['Train Time', `${data.training_time_min}m`, '#64B5F6'],
                    ].map(([label, value, color]) => (
                        <div key={label} style={{
                            background: 'rgba(0,0,0,0.3)',
                            borderRadius: 8, padding: '10px 12px', textAlign: 'center',
                        }}>
                            <div style={{ fontSize: 18, fontWeight: 800, color }}>{value}</div>
                            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{label}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* ── Per-group stat cards ──────────────────────────────────── */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: 12, marginBottom: 20,
            }}>
                {Object.entries(data.group_metrics).map(([g, v]) => (
                    <Card key={g} style={{ borderLeft: `3px solid ${GC[g]}` }}>
                        <div style={{ fontWeight: 700, color: GC[g], marginBottom: 10 }}>{g}</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                            {[['MAE', v.mae], ['RMSE', v.rmse], ['R²', v.r2], ['Samples', v.count]].map(([l, val]) => (
                                <div key={l} style={{
                                    background: '#0b1120', borderRadius: 6,
                                    padding: '6px 8px', textAlign: 'center',
                                }}>
                                    <div style={{ fontSize: 14, fontWeight: 700, color: GC[g] }}>{val}</div>
                                    <div style={{ fontSize: 9, color: '#64748b' }}>{l}</div>
                                </div>
                            ))}
                        </div>
                    </Card>
                ))}
            </div>

            <TabBar
                tabs={['By State', 'Scatter Plot', 'Records Table']}
                active={tab}
                onChange={setTab}
            />

            {/* ── Tab 1 : By State ─────────────────────────────────────── */}
            {tab === 'By State' && (
                <Card>
                    <SectionTitle
                        title="Actual vs Predicted · State Level (2022)"
                        sub="Average crimes per record across all groups · top 20 states"
                    />
                    <ResponsiveContainer width="100%" height={340}>
                        <BarChart
                            data={data.state_preds
                                .sort((a, b) => b.actual - a.actual)
                                .slice(0, 20)}
                            margin={{ left: 0, right: 10, bottom: 80, top: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="state"
                                tick={{ fontSize: 8, fill: '#64748b' }}
                                angle={-40}
                                textAnchor="end"
                                interval={0}
                            />
                            <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="actual" fill="#ef4444" radius={[3, 3, 0, 0]} name="Actual" opacity={0.9} />
                            <Bar dataKey="predicted" fill="#a78bfa" radius={[3, 3, 0, 0]} name="FC-MT-LSTM Pred." opacity={0.9} />
                        </BarChart>
                    </ResponsiveContainer>
                    <div style={{ display: 'flex', gap: 16, marginTop: 8, justifyContent: 'center' }}>
                        {[['Actual 2022', '#ef4444'], ['FC-MT-LSTM Predicted', '#a78bfa']].map(([l, c]) => (
                            <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
                                <span style={{ fontSize: 12, color: '#64748b' }}>{l}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            )}

            {/* ── Tab 2 : Scatter Plot ──────────────────────────────────── */}
            {tab === 'Scatter Plot' && (
                <Card>
                    <SectionTitle
                        title="Actual vs Predicted Scatter"
                        sub="Perfect model = all dots on the diagonal · colored by group · 100 balanced samples"
                    />
                    <div style={{ marginBottom: 12 }}>
                        <Select
                            label="Filter by Group"
                            options={['All', 'SC', 'ST', 'Women', 'Children']}
                            value={group}
                            onChange={setGroup}
                        />
                    </div>

                    {/* Group count indicator */}
                    <div style={{ display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
                        {['SC', 'ST', 'Women', 'Children'].map(g => {
                            const cnt = scatterData.filter(s => s.group === g).length;
                            return (
                                <div key={g} style={{
                                    background: `${GC[g]}15`,
                                    border: `1px solid ${GC[g]}44`,
                                    borderRadius: 20, padding: '3px 10px',
                                    fontSize: 11, color: GC[g], fontWeight: 600,
                                }}>
                                    {g}: {cnt}
                                </div>
                            );
                        })}
                    </div>

                    <ResponsiveContainer width="100%" height={360}>
                        <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                type="number" dataKey="actual" name="Actual"
                                domain={['auto', 'auto']}
                                tick={{ fontSize: 10, fill: '#64748b' }}
                                label={{ value: 'Actual →', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }}
                            />
                            <YAxis
                                type="number" dataKey="predicted" name="Predicted"
                                domain={['auto', 'auto']}
                                tick={{ fontSize: 10, fill: '#64748b' }}
                                label={{ value: 'Predicted ↑', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                            />
                            <Tooltip content={({ active, payload }) => {
                                if (!active || !payload?.length) return null;
                                const d = payload[0]?.payload;
                                return (
                                    <div style={{
                                        background: '#111827', border: '1px solid rgba(255,255,255,0.08)',
                                        borderRadius: 8, padding: '8px 12px', fontSize: 12,
                                    }}>
                                        <div style={{ fontWeight: 700, color: GC[d.group] || '#a78bfa', marginBottom: 4 }}>
                                            {d.group} · {d.district}
                                        </div>
                                        <div style={{ color: '#94a3b8', marginBottom: 4 }}>{d.state}</div>
                                        <div>Actual: <b style={{ color: '#ef4444' }}>{d.actual}</b></div>
                                        <div>Predicted: <b style={{ color: '#a78bfa' }}>{d.predicted}</b></div>
                                    </div>
                                );
                            }} />
                            <ReferenceLine
                                segment={[{ x: 0, y: 0 }, { x: 800, y: 800 }]}
                                stroke="rgba(255,255,255,0.15)" strokeDasharray="6 3"
                            />
                            <Scatter data={scatterData} opacity={0.7}>
                                {scatterData.map((s, i) => (
                                    <Cell key={i} fill={GC[s.group] || '#a78bfa'} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>

                    <div style={{ display: 'flex', gap: 14, marginTop: 8, flexWrap: 'wrap' }}>
                        {['SC', 'ST', 'Women', 'Children'].map(g => (
                            <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <div style={{ width: 10, height: 10, borderRadius: '50%', background: GC[g] }} />
                                <span style={{ fontSize: 11, color: '#64748b' }}>{g}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            )}

            {/* ── Tab 3 : Records Table ─────────────────────────────────── */}
            {tab === 'Records Table' && (
                <Card>
                    <SectionTitle
                        title="Balanced Sample — All 4 Groups (2022)"
                        sub="25 highest-crime records per group · colored error: green <10%, yellow <30%, red >30%"
                    />
                    <div style={{ marginBottom: 12 }}>
                        <Select
                            label="Filter by Group"
                            options={['All', 'SC', 'ST', 'Women', 'Children']}
                            value={group}
                            onChange={setGroup}
                        />
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                            <thead>
                                <tr style={{ background: '#0b1120' }}>
                                    {['#', 'State', 'District', 'Group', 'Actual', 'Predicted', 'Error %'].map(col => (
                                        <th key={col} style={{
                                            padding: '9px 12px', textAlign: 'left',
                                            color: '#64748b', fontSize: 11,
                                            fontWeight: 600, textTransform: 'uppercase',
                                            letterSpacing: 0.4, whiteSpace: 'nowrap',
                                        }}>
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {filteredSamples.map((s, i) => {
                                    const err = s.actual > 0
                                        ? Math.abs(s.predicted - s.actual) / s.actual * 100
                                        : 0;
                                    const ec = err < 10 ? '#34d399' : err < 30 ? '#fbbf24' : '#f87171';
                                    return (
                                        <tr key={i} style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                                            <td style={{ padding: '8px 12px', color: '#475569' }}>{i + 1}</td>
                                            <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{s.state}</td>
                                            <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{s.district}</td>
                                            <td style={{ padding: '8px 12px' }}>
                                                <span style={{
                                                    background: `${GC[s.group]}1a`,
                                                    color: GC[s.group],
                                                    border: `1px solid ${GC[s.group]}44`,
                                                    borderRadius: 20, padding: '2px 8px',
                                                    fontSize: 11, fontWeight: 600,
                                                }}>
                                                    {s.group}
                                                </span>
                                            </td>
                                            <td style={{ padding: '8px 12px', fontWeight: 700, color: '#f1f5f9' }}>
                                                {s.actual}
                                            </td>
                                            <td style={{ padding: '8px 12px', fontWeight: 600, color: '#a78bfa' }}>
                                                {s.predicted}
                                            </td>
                                            <td style={{ padding: '8px 12px', fontWeight: 600, color: ec }}>
                                                {err.toFixed(1)}%
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </Card>
            )}
        </Page>
    );
}