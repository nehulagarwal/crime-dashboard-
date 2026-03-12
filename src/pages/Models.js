import React, { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, Cell, CartesianGrid,
    ScatterChart, Scatter, ReferenceLine
} from 'recharts';
import { Page, PageTitle, Card, SectionTitle, TabBar } from '../components/UI';

// ── Exact numbers from paper Table 4 ─────────────────────────────────
const MODELS = [
    {
        model: 'SARIMA', mae: 166.61, rmse: 220.34, r2: 0.000,
        fairness_gap: 20.85, fairness_ratio: 1.17, time: 0.07,
        type: 'Statistical', color: '#64B5F6',
    },
    {
        model: 'Prophet', mae: 135.46, rmse: 198.19, r2: 0.191,
        fairness_gap: 17.60, fairness_ratio: 1.22, time: 1.41,
        type: 'Statistical', color: '#60A5FA',
    },
    {
        model: 'Random Forest', mae: 2.14, rmse: 5.63, r2: 0.9993,
        fairness_gap: 1.08, fairness_ratio: 4.74, time: 0.12,
        type: 'Ensemble', color: '#FCD34D',
    },
    {
        model: 'XGBoost', mae: 1.83, rmse: 4.13, r2: 0.9996,
        fairness_gap: 0.81, fairness_ratio: 3.73, time: 0.12,
        type: 'Ensemble', color: '#F59E0B',
    },
    {
        model: 'CNN-LSTM', mae: 23.83, rmse: 57.33, r2: 0.9419,
        fairness_gap: null, fairness_ratio: null, time: 4.16,
        type: 'Deep Learning', color: '#34D399',
    },
    {
        model: 'Transformer', mae: 4.65, rmse: 9.12, r2: 0.9985,
        fairness_gap: null, fairness_ratio: null, time: 6.90,
        type: 'Deep Learning', color: '#6EE7B7',
    },
    {
        model: 'FC-MT-LSTM', mae: 6.54, rmse: 16.05, r2: 0.9922,
        fairness_gap: 12.61, fairness_ratio: 1.99, time: 8.50,
        type: 'Ours', color: '#A78BFA', is_ours: true,
    },
];

const TYPE_COLORS = {
    Statistical: '#64B5F6',
    Ensemble: '#FCD34D',
    'Deep Learning': '#34D399',
    Ours: '#A78BFA',
};

const METRICS = [
    { key: 'mae', label: 'MAE ↓', better: 'lower' },
    { key: 'rmse', label: 'RMSE ↓', better: 'lower' },
    { key: 'r2', label: 'R² ↑', better: 'higher' },
    { key: 'fairness_ratio', label: 'Fairness Ratio ↓', better: 'lower' },
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

export default function Models() {
    const [tab, setTab] = useState('Overview');
    const [metric, setMetric] = useState('mae');

    const mc = METRICS.find(x => x.key === metric);
    const validModels = MODELS.filter(m => m[metric] != null);
    const vals = validModels.map(m => m[metric]);
    const best = mc.better === 'lower' ? Math.min(...vals) : Math.max(...vals);

    return (
        <Page>
            <PageTitle
                title="⚖️ Model Comparison"
                sub="7 models from the paper · Accuracy vs Fairness · Paper Table 4"
            />

            {/* ── Hero finding banner ───────────────────────────────────── */}
            <div style={{
                background: 'linear-gradient(135deg, rgba(167,139,250,0.15), rgba(239,68,68,0.08))',
                border: '1px solid rgba(167,139,250,0.3)',
                borderRadius: 14, padding: 24, marginBottom: 24,
            }}>
                <div style={{
                    display: 'flex', alignItems: 'flex-start',
                    gap: 16, flexWrap: 'wrap',
                }}>
                    <div style={{ fontSize: 32 }}>🏆</div>
                    <div style={{ flex: 1 }}>
                        <div style={{
                            display: 'flex', alignItems: 'center',
                            gap: 10, marginBottom: 8, flexWrap: 'wrap',
                        }}>
                            <span style={{ fontWeight: 800, fontSize: 18, color: '#ddd6fe' }}>
                                FC-MT-LSTM
                            </span>
                            <span style={{
                                fontSize: 11,
                                background: 'rgba(167,139,250,0.2)',
                                color: '#c4b5fd',
                                borderRadius: 20, padding: '2px 10px', fontWeight: 600,
                            }}>
                                Our Proposed Model
                            </span>
                            <span style={{
                                fontSize: 11,
                                background: 'rgba(52,211,153,0.15)',
                                color: '#34d399',
                                borderRadius: 20, padding: '2px 10px', fontWeight: 600,
                            }}>
                                Best Fairness-Accuracy Balance
                            </span>
                        </div>

                        <p style={{
                            fontSize: 13, color: '#94a3b8',
                            lineHeight: 1.7, marginBottom: 16,
                        }}>
                            Ensemble models (RF, XGBoost) achieve the lowest MAE but show
                            <b style={{ color: '#fca5a5' }}> 3.7–4.7× fairness disparity</b> across groups.
                            FC-MT-LSTM nearly halves the fairness ratio to
                            <b style={{ color: '#a78bfa' }}> 1.99</b> via a
                            fairness-regularised multi-task loss, while maintaining
                            competitive R²=0.9922.
                        </p>

                        <div style={{
                            display: 'flex', gap: 10, flexWrap: 'wrap',
                        }}>
                            {[
                                ['MAE', '6.54', '#a78bfa'],
                                ['RMSE', '16.05', '#a78bfa'],
                                ['R²', '0.9922', '#34d399'],
                                ['F.Ratio', '1.99', '#34d399'],
                                ['vs RF', '4.74×', '#f87171'],
                            ].map(([l, v, c]) => (
                                <div key={l} style={{
                                    background: 'rgba(0,0,0,0.3)',
                                    borderRadius: 8, padding: '8px 14px', textAlign: 'center',
                                }}>
                                    <div style={{ fontSize: 18, fontWeight: 800, color: c }}>{v}</div>
                                    <div style={{ fontSize: 10, color: '#64748b' }}>{l}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            <TabBar
                tabs={['Overview', 'Bar Chart', 'Acc vs Fair', 'Full Table']}
                active={tab}
                onChange={setTab}
            />

            {/* ── Tab 1 : Overview cards ────────────────────────────────── */}
            {tab === 'Overview' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {MODELS.map(m => (
                        <Card key={m.model} style={{
                            borderLeft: `3px solid ${m.color}`,
                            background: m.is_ours ? 'rgba(167,139,250,0.07)' : '#111827',
                        }}>
                            <div style={{
                                display: 'flex', alignItems: 'center',
                                gap: 10, marginBottom: 12, flexWrap: 'wrap',
                            }}>
                                {m.is_ours && (
                                    <span style={{ fontSize: 16, color: '#a78bfa' }}>★</span>
                                )}
                                <span style={{
                                    fontWeight: 700, fontSize: 15,
                                    color: m.is_ours ? '#ddd6fe' : '#f1f5f9',
                                }}>
                                    {m.model}
                                </span>
                                <span style={{
                                    fontSize: 11,
                                    background: `${TYPE_COLORS[m.type]}18`,
                                    color: TYPE_COLORS[m.type],
                                    border: `1px solid ${TYPE_COLORS[m.type]}33`,
                                    borderRadius: 4, padding: '2px 7px', fontWeight: 600,
                                }}>
                                    {m.type}
                                </span>
                                <span style={{
                                    marginLeft: 'auto', fontSize: 11, color: '#475569',
                                }}>
                                    {m.time} min
                                </span>
                            </div>

                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))',
                                gap: 8,
                            }}>
                                {[
                                    ['MAE', m.mae, 'lower'],
                                    ['RMSE', m.rmse, 'lower'],
                                    ['R²', m.r2, 'higher'],
                                    ['F.Ratio', m.fairness_ratio, 'lower'],
                                ].map(([label, value, better]) => {
                                    const allVals = MODELS
                                        .filter(x => x[label.toLowerCase().replace('.', '_')] != null)
                                        .map(x => x[label.toLowerCase().replace('.', '_')]);
                                    return (
                                        <div key={label} style={{
                                            background: '#0b1120',
                                            borderRadius: 6, padding: '8px 10px', textAlign: 'center',
                                        }}>
                                            <div style={{
                                                fontSize: 14, fontWeight: 700,
                                                color: m.is_ours ? '#ddd6fe' : '#94a3b8',
                                            }}>
                                                {value == null ? '—' : typeof value === 'number' ? value : value}
                                            </div>
                                            <div style={{ fontSize: 9, color: '#64748b', marginTop: 1 }}>
                                                {label}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </Card>
                    ))}
                </div>
            )}

            {/* ── Tab 2 : Bar Chart ─────────────────────────────────────── */}
            {tab === 'Bar Chart' && (
                <Card>
                    {/* Metric selector */}
                    <div style={{
                        display: 'flex', gap: 8,
                        marginBottom: 16, flexWrap: 'wrap',
                    }}>
                        {METRICS.map(mt => (
                            <button
                                key={mt.key}
                                onClick={() => setMetric(mt.key)}
                                style={{
                                    padding: '5px 12px', borderRadius: 6,
                                    border: `1px solid ${metric === mt.key ? '#ef4444' : 'rgba(255,255,255,0.08)'}`,
                                    background: metric === mt.key ? 'rgba(239,68,68,0.15)' : 'transparent',
                                    color: metric === mt.key ? '#fca5a5' : '#64748b',
                                    fontSize: 12, fontWeight: 600,
                                }}
                            >
                                {mt.label}
                            </button>
                        ))}
                    </div>

                    <SectionTitle
                        title={`${mc.label} by Model`}
                        sub={`${mc.better === 'lower' ? 'Lower' : 'Higher'} is better · purple = our model · green = best`}
                    />

                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart
                            data={validModels}
                            margin={{ left: 0, right: 10, top: 10, bottom: 60 }}
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
                            <Bar dataKey={metric} radius={[4, 4, 0, 0]} name={mc.label}>
                                {validModels.map((m, i) => (
                                    <Cell
                                        key={i}
                                        fill={
                                            m.is_ours ? '#a78bfa' :
                                                m[metric] === best ? '#34d399' :
                                                    '#1f2937'
                                        }
                                        opacity={0.9}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </Card>
            )}

            {/* ── Tab 3 : Accuracy vs Fairness scatter ─────────────────── */}
            {tab === 'Acc vs Fair' && (
                <Card>
                    <SectionTitle
                        title="Accuracy vs Fairness Trade-off (Paper Fig. 2)"
                        sub="Bottom-left = ideal · FC-MT-LSTM sits in the balanced zone"
                    />
                    <ResponsiveContainer width="100%" height={380}>
                        <ScatterChart margin={{ left: 10, right: 30, top: 20, bottom: 30 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                type="number" dataKey="mae" name="MAE"
                                scale="log" domain={[1, 300]}
                                tick={{ fontSize: 10, fill: '#64748b' }}
                                label={{
                                    value: 'MAE (log scale) →',
                                    position: 'insideBottom', offset: -16,
                                    fill: '#64748b', fontSize: 11,
                                }}
                            />
                            <YAxis
                                type="number" dataKey="fairness_ratio" name="Fairness Ratio"
                                domain={[1, 5.5]}
                                tick={{ fontSize: 10, fill: '#64748b' }}
                                label={{
                                    value: 'Fairness Ratio →',
                                    angle: -90, position: 'insideLeft',
                                    fill: '#64748b', fontSize: 11,
                                }}
                            />
                            <Tooltip content={({ active, payload }) => {
                                if (!active || !payload?.length) return null;
                                const d = payload[0]?.payload;
                                return (
                                    <div style={{
                                        background: '#111827',
                                        border: '1px solid rgba(255,255,255,0.08)',
                                        borderRadius: 8, padding: '8px 12px', fontSize: 12,
                                    }}>
                                        <div style={{ fontWeight: 700, color: d.color, marginBottom: 4 }}>
                                            {d.model}
                                        </div>
                                        <div style={{ color: '#ef4444' }}>MAE: {d.mae}</div>
                                        <div style={{ color: '#a78bfa' }}>
                                            Fairness Ratio: {d.fairness_ratio}
                                        </div>
                                        <div style={{ color: '#64748b' }}>R²: {d.r2}</div>
                                    </div>
                                );
                            }} />
                            <ReferenceLine x={20} stroke="rgba(255,255,255,0.06)" strokeDasharray="6 3" />
                            <ReferenceLine y={2.5} stroke="rgba(255,255,255,0.06)" strokeDasharray="6 3" />
                            <Scatter
                                data={MODELS.filter(m => m.fairness_ratio != null)}
                                shape={(props) => {
                                    const { cx, cy, payload } = props;
                                    if (payload.is_ours) return (
                                        <g>
                                            <circle cx={cx} cy={cy} r={16}
                                                fill="rgba(167,139,250,0.2)" />
                                            <rect x={cx - 9} y={cy - 9}
                                                width={18} height={18} rx={3}
                                                fill="#a78bfa" />
                                            <text x={cx + 14} y={cy - 6}
                                                fontSize={10} fill="#ddd6fe"
                                                fontWeight="bold">
                                                {payload.model}
                                            </text>
                                        </g>
                                    );
                                    return (
                                        <g>
                                            <circle cx={cx} cy={cy} r={8}
                                                fill={payload.color} opacity={0.85} />
                                            <text x={cx + 11} y={cy + 4}
                                                fontSize={9} fill="#64748b">
                                                {payload.model.split(' ')[0]}
                                            </text>
                                        </g>
                                    );
                                }}
                            />
                        </ScatterChart>
                    </ResponsiveContainer>

                    <div style={{
                        marginTop: 12,
                        background: 'rgba(52,211,153,0.06)',
                        border: '1px solid rgba(52,211,153,0.2)',
                        borderRadius: 8, padding: 12,
                        fontSize: 12, color: '#94a3b8',
                    }}>
                        💡 <b style={{ color: '#34d399' }}>Optimal region</b> = bottom-left
                        (low MAE + low fairness ratio).
                        FC-MT-LSTM is positioned between accuracy-focused ensembles
                        and fair-but-inaccurate statistical models.
                    </div>
                </Card>
            )}

            {/* ── Tab 4 : Full Table ────────────────────────────────────── */}
            {tab === 'Full Table' && (
                <Card>
                    <SectionTitle
                        title="Complete Results — Paper Table 4"
                        sub="★ = our model · — = data not available · ↓ lower better · ↑ higher better"
                    />
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                            <thead>
                                <tr style={{ background: '#0b1120' }}>
                                    {['Model', 'Type', 'MAE ↓', 'RMSE ↓', 'R² ↑', 'F.Gap ↓', 'F.Ratio ↓', 'Time'].map(col => (
                                        <th key={col} style={{
                                            padding: '10px 12px', textAlign: 'left',
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
                                {MODELS.map((m, i) => (
                                    <tr key={m.model} style={{
                                        borderTop: '1px solid rgba(255,255,255,0.05)',
                                        background: m.is_ours
                                            ? 'rgba(167,139,250,0.07)'
                                            : 'transparent',
                                    }}>
                                        {/* Model name */}
                                        <td style={{ padding: '10px 12px' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                <div style={{
                                                    width: 3, height: 20,
                                                    borderRadius: 2, background: m.color,
                                                }} />
                                                <span style={{
                                                    fontWeight: m.is_ours ? 700 : 500,
                                                    color: m.is_ours ? '#ddd6fe' : '#f1f5f9',
                                                    whiteSpace: 'nowrap',
                                                }}>
                                                    {m.is_ours && '★ '}{m.model}
                                                </span>
                                            </div>
                                        </td>

                                        {/* Type badge */}
                                        <td style={{ padding: '10px 12px' }}>
                                            <span style={{
                                                background: `${TYPE_COLORS[m.type]}18`,
                                                color: TYPE_COLORS[m.type],
                                                border: `1px solid ${TYPE_COLORS[m.type]}33`,
                                                borderRadius: 4, padding: '2px 6px',
                                                fontSize: 10, fontWeight: 600,
                                            }}>
                                                {m.type}
                                            </span>
                                        </td>

                                        {/* Metrics */}
                                        {[m.mae, m.rmse, m.r2, m.fairness_gap, m.fairness_ratio].map((val, j) => (
                                            <td key={j} style={{
                                                padding: '10px 12px',
                                                color: m.is_ours ? '#ddd6fe' : '#94a3b8',
                                                fontWeight: m.is_ours ? 600 : 400,
                                            }}>
                                                {val == null
                                                    ? <span style={{ color: '#475569' }}>—</span>
                                                    : typeof val === 'number'
                                                        ? val.toFixed(j === 2 || j === 4 ? 4 : 2)
                                                        : val}
                                            </td>
                                        ))}

                                        {/* Time */}
                                        <td style={{ padding: '10px 12px', color: '#475569' }}>
                                            {m.time}m
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div style={{ marginTop: 10, fontSize: 11, color: '#475569' }}>
                        * CNN-LSTM and Transformer show — for fairness metrics
                        because per-group breakdown was not evaluated in the paper.
                    </div>
                </Card>
            )}

            {/* ── Model families explainer ──────────────────────────────── */}
            <div style={{ marginTop: 20 }}>
                <Card>
                    <SectionTitle
                        title="Why FC-MT-LSTM wins on fairness"
                        sub="Understanding the trade-off each model family makes"
                    />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {[
                            {
                                title: 'Statistical (SARIMA, Prophet)',
                                color: '#64B5F6',
                                content: 'Fair but inaccurate. R² near 0 means they barely explain variance. Useless for resource allocation despite low fairness ratio.',
                            },
                            {
                                title: 'Ensemble (RF, XGBoost)',
                                color: '#FCD34D',
                                content: 'Most accurate (MAE 1.8–2.1) but worst fairness (ratio 3.7–4.7×). The worst-served group has nearly 5× higher prediction error. Dangerous for policy decisions.',
                            },
                            {
                                title: 'Deep Learning (CNN-LSTM, Transformer)',
                                color: '#34D399',
                                content: 'Good accuracy but fairness was not evaluated. Cannot confirm equitable treatment without per-group metrics.',
                            },
                            {
                                title: 'FC-MT-LSTM — Our Model ★',
                                color: '#A78BFA',
                                content: 'Balances both. MAE=6.54, R²=0.9922, Fairness Ratio=1.99. The fairness-regularised loss directly penalises cross-group MAE differences during training.',
                            },
                        ].map(b => (
                            <div key={b.title} style={{
                                display: 'flex', gap: 14, padding: 14,
                                background: b.title.includes('★') ? 'rgba(167,139,250,0.07)' : '#0b1120',
                                borderRadius: 10, border: `1px solid ${b.color}22`,
                            }}>
                                <div style={{
                                    width: 3, borderRadius: 2,
                                    background: b.color, flexShrink: 0,
                                }} />
                                <div>
                                    <div style={{
                                        fontWeight: 700, color: b.color,
                                        marginBottom: 5, fontSize: 13,
                                    }}>
                                        {b.title}
                                    </div>
                                    <p style={{
                                        fontSize: 12, color: '#64748b', lineHeight: 1.6,
                                    }}>
                                        {b.content}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>

        </Page>
    );
}