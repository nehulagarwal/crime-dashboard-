import React, { useState, useMemo } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, Cell, CartesianGrid
} from 'recharts';
import { Page, PageTitle, Card, SectionTitle, TabBar, Select } from '../components/UI';
import data from '../data/cities.json';

const GROUP_COLORS = {
    SC: '#64B5F6',
    ST: '#81C784',
    Women: '#FF7043',
    Children: '#FFB74D',
};

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
                    {p.name}: {p.value}
                </div>
            ))}
        </div>
    );
}

export default function Cities() {
    const [group, setGroup] = useState('Women');
    const [year, setYear] = useState('2022');
    const [tab, setTab] = useState('Bar Chart');

    // build chart data for selected group + year
    const chartData = useMemo(() => {
        return data.cities
            .map(c => ({
                city: c.city,
                value: c.groups[group]?.[year] || 0,
                y2021: c.groups[group]?.['2021'] || 0,
                y2022: c.groups[group]?.['2022'] || 0,
                y2023: c.groups[group]?.['2023'] || 0,
            }))
            .sort((a, b) => b.value - a.value);
    }, [group, year]);

    const maxValue = Math.max(...chartData.map(c => c.value), 1);
    const color = GROUP_COLORS[group];

    return (
        <Page>
            <PageTitle
                title="🏙️ Metro City Analysis"
                sub="34 major Indian cities · Crime data 2021–2023 · Source: NCRB"
            />

            {/* ── Filters ───────────────────────────────────────────────── */}
            <Card style={{ marginBottom: 16 }}>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: 14,
                }}>
                    <Select
                        label="Protected Group"
                        options={['Women', 'SC', 'ST', 'Children']}
                        value={group}
                        onChange={setGroup}
                    />
                    <Select
                        label="Year"
                        options={['2021', '2022', '2023']}
                        value={year}
                        onChange={setYear}
                    />
                </div>
            </Card>

            <TabBar
                tabs={['Bar Chart', 'Year Comparison', 'Ranked List']}
                active={tab}
                onChange={setTab}
            />

            {/* ── Tab 1 : Bar Chart ─────────────────────────────────────── */}
            {tab === 'Bar Chart' && (
                <Card>
                    <SectionTitle
                        title={`${group} Crimes · ${year} · Top 20 Cities`}
                        sub="Sorted highest to lowest"
                    />
                    <ResponsiveContainer width="100%" height={320}>
                        <BarChart
                            data={chartData.slice(0, 20)}
                            margin={{ left: 0, right: 10, bottom: 60, top: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="city"
                                tick={{ fontSize: 9, fill: '#64748b' }}
                                angle={-35}
                                textAnchor="end"
                                interval={0}
                            />
                            <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]} name={`${group} Crimes`}>
                                {chartData.slice(0, 20).map((_, i) => (
                                    <Cell key={i} fill={color} opacity={1 - i * 0.03} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </Card>
            )}

            {/* ── Tab 2 : Year Comparison ───────────────────────────────── */}
            {tab === 'Year Comparison' && (
                <Card>
                    <SectionTitle
                        title={`${group} Crimes · 2021 vs 2022 vs 2023 · Top 10 Cities`}
                        sub="See how each city changed over the years"
                    />
                    <ResponsiveContainer width="100%" height={320}>
                        <BarChart
                            data={[...chartData]
                                .sort((a, b) => (b.y2021 + b.y2022 + b.y2023) - (a.y2021 + a.y2022 + a.y2023))
                                .slice(0, 10)}
                            margin={{ left: 0, right: 10, bottom: 60, top: 5 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="city"
                                tick={{ fontSize: 9, fill: '#64748b' }}
                                angle={-35}
                                textAnchor="end"
                                interval={0}
                            />
                            <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="y2021" fill="#64B5F6" radius={[3, 3, 0, 0]} name="2021" />
                            <Bar dataKey="y2022" fill={color} radius={[3, 3, 0, 0]} name="2022" />
                            <Bar dataKey="y2023" fill="#81C784" radius={[3, 3, 0, 0]} name="2023" />
                        </BarChart>
                    </ResponsiveContainer>

                    {/* Legend */}
                    <div style={{ display: 'flex', gap: 16, marginTop: 12, justifyContent: 'center' }}>
                        {[['2021', '#64B5F6'], ['2022', color], ['2023', '#81C784']].map(([yr, c]) => (
                            <div key={yr} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />
                                <span style={{ fontSize: 12, color: '#64748b' }}>{yr}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            )}

            {/* ── Tab 3 : Ranked List ───────────────────────────────────── */}
            {tab === 'Ranked List' && (
                <Card>
                    <SectionTitle
                        title={`All 34 Cities Ranked · ${group} · ${year}`}
                        sub="Highest crimes at the top"
                    />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                        {chartData.map((c, i) => (
                            <div key={c.city} style={{
                                display: 'grid',
                                gridTemplateColumns: '28px 1fr auto',
                                gap: 10,
                                alignItems: 'center',
                            }}>
                                {/* Rank number */}
                                <span style={{
                                    fontSize: 11, color: '#475569',
                                    textAlign: 'right', fontWeight: 600,
                                }}>
                                    {i + 1}
                                </span>

                                {/* City name + bar */}
                                <div>
                                    <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 3 }}>
                                        {c.city}
                                    </div>
                                    <div style={{
                                        background: '#0b1120',
                                        borderRadius: 3, height: 5, overflow: 'hidden',
                                    }}>
                                        <div style={{
                                            width: `${(c.value / maxValue) * 100}%`,
                                            height: '100%',
                                            background: color,
                                            borderRadius: 3,
                                        }} />
                                    </div>
                                </div>

                                {/* Value */}
                                <span style={{
                                    fontSize: 13, fontWeight: 700,
                                    color: color, minWidth: 40, textAlign: 'right',
                                }}>
                                    {c.value}
                                </span>
                            </div>
                        ))}
                    </div>
                </Card>
            )}

        </Page>
    );
}