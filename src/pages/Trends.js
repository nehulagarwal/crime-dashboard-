import React, { useState } from 'react';
import {
    LineChart, Line, BarChart, Bar,
    XAxis, YAxis, Tooltip, Legend,
    ResponsiveContainer, CartesianGrid, Cell
} from 'recharts';
import { Page, PageTitle, Card, SectionTitle, TabBar, Select } from '../components/UI';
import data from '../data/trends.json';

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
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 600 }}>
                {label}
            </div>
            {payload.map((p, i) => (
                <div key={i} style={{ color: p.color, fontSize: 12, marginBottom: 2 }}>
                    {p.name}: <b>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</b>
                </div>
            ))}
        </div>
    );
}

export default function Trends() {
    const [tab, setTab] = useState('By Group');
    const [state, setState] = useState(data.top_states[0]);

    return (
        <Page>
            <PageTitle
                title="📈 Crime Trends 2017–2022"
                sub="Historical patterns across all protected groups and states"
            />

            <TabBar
                tabs={['By Group', 'By State', 'Year Table']}
                active={tab}
                onChange={setTab}
            />

            {/* ── Tab 1 : By Group ─────────────────────────────────────── */}
            {tab === 'By Group' && (
                <div>
                    <Card style={{ marginBottom: 16 }}>
                        <SectionTitle
                            title="Average Crimes per Record · by Protected Group"
                            sub="Each line = one group · across all districts and states"
                        />
                        <ResponsiveContainer width="100%" height={320}>
                            <LineChart
                                data={data.yearly}
                                margin={{ left: 0, right: 20, top: 10, bottom: 0 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="year" tick={{ fontSize: 11, fill: '#64748b' }} />
                                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                                <Tooltip content={<CustomTooltip />} />
                                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 16 }} />
                                {['SC', 'ST', 'Women', 'Children'].map(g => (
                                    <Line
                                        key={g}
                                        type="monotone"
                                        dataKey={g}
                                        stroke={GROUP_COLORS[g]}
                                        strokeWidth={2.5}
                                        dot={{ r: 4, fill: GROUP_COLORS[g] }}
                                        activeDot={{ r: 6 }}
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </Card>

                    {/* What the chart tells us */}
                    <Card>
                        <SectionTitle title="What this chart tells us" />
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                            gap: 12,
                        }}>
                            {[
                                { group: 'SC', color: '#64B5F6', note: 'Crimes against Scheduled Castes show a gradual upward trend across most states' },
                                { group: 'ST', color: '#81C784', note: 'Scheduled Tribe crimes remain lower overall due to smaller population base' },
                                { group: 'Women', color: '#FF7043', note: 'Women crimes are consistently the highest — a key concern across all years' },
                                { group: 'Children', color: '#FFB74D', note: 'Children crimes increased post-2019 partly due to better POCSO Act reporting' },
                            ].map(item => (
                                <div key={item.group} style={{
                                    borderLeft: `3px solid ${item.color}`,
                                    paddingLeft: 12, paddingTop: 4, paddingBottom: 4,
                                }}>
                                    <div style={{ fontWeight: 700, color: item.color, marginBottom: 4 }}>
                                        {item.group}
                                    </div>
                                    <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
                                        {item.note}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>
            )}

            {/* ── Tab 2 : By State ─────────────────────────────────────── */}
            {tab === 'By State' && (
                <div>
                    <Card style={{ marginBottom: 14 }}>
                        <Select
                            label="Select State"
                            options={data.all_states}
                            value={state}
                            onChange={setState}
                        />
                    </Card>

                    <Card style={{ marginBottom: 14 }}>
                        <SectionTitle
                            title={`Crime Trend · ${state}`}
                            sub="Average total crimes per record per year"
                        />
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart
                                data={data.state_trends[state] || []}
                                margin={{ left: 0, right: 20, top: 10 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="year" tick={{ fontSize: 11, fill: '#64748b' }} />
                                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                                <Tooltip content={<CustomTooltip />} />
                                <Line
                                    type="monotone"
                                    dataKey="avg"
                                    stroke="#ef4444"
                                    strokeWidth={2.5}
                                    dot={{ r: 5, fill: '#ef4444' }}
                                    activeDot={{ r: 7 }}
                                    name="Avg Crimes"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </Card>

                    {/* Top 10 states comparison bar chart */}
                    <Card>
                        <SectionTitle
                            title="Top 10 States · All Years Average"
                            sub="Compare high-crime states at a glance"
                        />
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart
                                data={data.top_states.map(s => ({
                                    state: s.length > 12 ? s.slice(0, 12) + '…' : s,
                                    avg: data.state_trends[s]
                                        ? Math.round(
                                            data.state_trends[s].reduce((sum, r) => sum + r.avg, 0) /
                                            data.state_trends[s].length
                                        )
                                        : 0,
                                }))}
                                margin={{ left: 0, bottom: 50 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis
                                    dataKey="state"
                                    tick={{ fontSize: 9, fill: '#64748b' }}
                                    angle={-35}
                                    textAnchor="end"
                                />
                                <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                                <Tooltip content={<CustomTooltip />} />
                                <Bar dataKey="avg" radius={[4, 4, 0, 0]} name="Avg Crimes">
                                    {data.top_states.map((_, i) => (
                                        <Cell key={i} fill="#ef4444" opacity={1 - i * 0.07} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </Card>
                </div>
            )}

            {/* ── Tab 3 : Year Table ───────────────────────────────────── */}
            {tab === 'Year Table' && (
                <Card>
                    <SectionTitle
                        title="Yearly Summary Table"
                        sub="Average crimes per record · broken down by group"
                    />
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                            <thead>
                                <tr style={{ background: '#0b1120' }}>
                                    {['Year', 'SC', 'ST', 'Women', 'Children'].map(col => (
                                        <th key={col} style={{
                                            padding: '10px 16px', textAlign: 'left',
                                            color: '#64748b', fontWeight: 600,
                                            fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5,
                                        }}>
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {data.yearly.map((row, i) => (
                                    <tr key={row.year} style={{
                                        borderTop: '1px solid rgba(255,255,255,0.05)',
                                        background: i % 2 ? 'rgba(255,255,255,0.01)' : 'transparent',
                                    }}>
                                        <td style={{
                                            padding: '10px 16px',
                                            fontWeight: 700, color: '#f1f5f9',
                                        }}>
                                            {row.year}
                                        </td>
                                        {['SC', 'ST', 'Women', 'Children'].map(g => (
                                            <td key={g} style={{
                                                padding: '10px 16px',
                                                color: GROUP_COLORS[g],
                                                fontWeight: 600,
                                            }}>
                                                {row[g]}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Card>
            )}

        </Page>
    );
}