import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, Cell, PieChart, Pie
} from 'recharts';
import { Page, PageTitle, Card, StatCard, SectionTitle } from '../components/UI';
import data from '../data/overview.json';

const { dataset, groups, top_states } = data;

// ── Custom tooltip for charts ─────────────────────────────────────────
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
                <div key={i} style={{ fontSize: 13, fontWeight: 600, color: p.color || '#f1f5f9' }}>
                    {p.name}: {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}
                </div>
            ))}
        </div>
    );
}

export default function Overview() {
    return (
        <Page>
            <PageTitle
                title="📊 Dataset Overview"
                sub="National Crime Records Bureau (NCRB) India · 2017–2022 · 4 protected groups"
            />

            {/* ── Key stats grid ────────────────────────────────────────── */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                gap: 12,
                marginBottom: 20,
            }}>
                <StatCard icon="📁" label="Total Records" value={dataset.total_records.toLocaleString()} color="#64B5F6" sub="Cleaned and validated" />
                <StatCard icon="🗺️" label="States / UTs" value={dataset.states} color="#81C784" sub="All of India" />
                <StatCard icon="🏘️" label="Districts" value={`${dataset.districts}+`} color="#FF7043" sub="District-wise data" />
                <StatCard icon="⚙️" label="Features" value={dataset.features} color="#FFB74D" sub="Engineered features" />
                <StatCard icon="🚂" label="Train Records" value={dataset.train_records.toLocaleString()} color="#a78bfa" sub="Years 2017–2021" />
                <StatCard icon="🧪" label="Test Records" value={dataset.test_records.toLocaleString()} color="#f472b6" sub="Year 2022 only" />
            </div>

            {/* ── Train / test split ────────────────────────────────────── */}
            <Card style={{ marginBottom: 20 }}>
                <SectionTitle
                    title="Train / Test Split"
                    sub="Model learns from 2017–2021 data then predicts 2022"
                />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>

                    <div style={{
                        background: 'rgba(129,199,132,0.08)',
                        border: '1px solid rgba(129,199,132,0.2)',
                        borderRadius: 10, padding: 16,
                    }}>
                        <div style={{
                            background: '#81C784', color: '#111',
                            fontSize: 10, fontWeight: 700,
                            borderRadius: 4, padding: '2px 8px',
                            display: 'inline-block', marginBottom: 10,
                        }}>
                            TRAIN
                        </div>
                        <div style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 4 }}>
                            2017 – 2021
                        </div>
                        <div style={{ fontSize: 13, color: '#64748b' }}>
                            {dataset.train_records.toLocaleString()} records
                        </div>
                        <div style={{ fontSize: 12, color: '#475569', marginTop: 6 }}>
                            Model learns patterns from these 5 years
                        </div>
                    </div>

                    <div style={{
                        background: 'rgba(255,112,67,0.08)',
                        border: '1px solid rgba(255,112,67,0.2)',
                        borderRadius: 10, padding: 16,
                    }}>
                        <div style={{
                            background: '#FF7043', color: '#111',
                            fontSize: 10, fontWeight: 700,
                            borderRadius: 4, padding: '2px 8px',
                            display: 'inline-block', marginBottom: 10,
                        }}>
                            TEST
                        </div>
                        <div style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 4 }}>
                            2022
                        </div>
                        <div style={{ fontSize: 13, color: '#64748b' }}>
                            {dataset.test_records.toLocaleString()} records
                        </div>
                        <div style={{ fontSize: 12, color: '#475569', marginTop: 6 }}>
                            We check predictions against real 2022 data
                        </div>
                    </div>

                </div>
            </Card>

            {/* ── Groups breakdown ──────────────────────────────────────── */}
            <Card style={{ marginBottom: 20 }}>
                <SectionTitle
                    title="Protected Group Distribution"
                    sub="Each group has its own crime categories and record count"
                />
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: 20,
                    alignItems: 'center',
                }}>

                    {/* Pie chart */}
                    <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                            <Pie
                                data={groups}
                                dataKey="records"
                                nameKey="name"
                                cx="50%" cy="50%"
                                outerRadius={80}
                                innerRadius={45}
                                paddingAngle={2}
                            >
                                {groups.map(g => (
                                    <Cell key={g.name} fill={g.color} />
                                ))}
                            </Pie>
                            <Tooltip
                                formatter={v => v.toLocaleString()}
                                contentStyle={{
                                    background: '#111827',
                                    border: '1px solid rgba(255,255,255,0.08)',
                                    borderRadius: 8,
                                }}
                            />
                        </PieChart>
                    </ResponsiveContainer>

                    {/* Group list */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                        {groups.map(g => (
                            <div key={g.name}>
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    marginBottom: 4,
                                }}>
                                    <span style={{ fontSize: 13, fontWeight: 600, color: g.color }}>
                                        {g.name}
                                    </span>
                                    <span style={{ fontSize: 11, color: '#64748b' }}>
                                        {g.records.toLocaleString()} · {g.categories} categories
                                    </span>
                                </div>
                                {/* Progress bar */}
                                <div style={{
                                    background: '#0b1120',
                                    borderRadius: 4, height: 6, overflow: 'hidden',
                                }}>
                                    <div style={{
                                        width: `${(g.records / 5400) * 100}%`,
                                        height: '100%',
                                        background: g.color,
                                        borderRadius: 4,
                                    }} />
                                </div>
                                <div style={{ fontSize: 11, color: '#475569', marginTop: 3 }}>
                                    {g.desc}
                                </div>
                            </div>
                        ))}
                    </div>

                </div>
            </Card>

            {/* ── Top states bar chart ──────────────────────────────────── */}
            <Card>
                <SectionTitle
                    title="Top 10 States by Average Crime Rate"
                    sub="Average total crimes per record across all years and groups"
                />
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                        data={top_states}
                        margin={{ left: 0, right: 10, top: 5, bottom: 70 }}
                    >
                        <XAxis
                            dataKey="state"
                            tick={{ fontSize: 10, fill: '#64748b' }}
                            angle={-35}
                            textAnchor="end"
                            interval={0}
                        />
                        <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Bar dataKey="avg" radius={[4, 4, 0, 0]} name="Avg Crimes">
                            {top_states.map((_, i) => (
                                <Cell key={i} fill={`hsl(${210 + i * 15}, 70%, ${55 + i * 2}%)`} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </Card>

        </Page>
    );
}