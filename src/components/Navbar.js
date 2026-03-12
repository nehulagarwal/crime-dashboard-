import React from 'react';
import { NavLink } from 'react-router-dom';

const links = [
    { to: '/', label: 'Home' },
    { to: '/overview', label: 'Overview' },
    { to: '/trends', label: 'Trends' },
    { to: '/cities', label: 'Cities' },
    { to: '/predictions', label: 'Predictions' },
    { to: '/models', label: 'Models' },
    { to: '/fairness', label: 'Fairness' },
];

export default function Navbar() {
    return (
        <nav style={styles.nav}>

            <span style={styles.brand}>⚡ Crime Analysis</span>

            <div style={styles.links}>
                {links.map(link => (
                    <NavLink
                        key={link.to}
                        to={link.to}
                        end
                        style={({ isActive }) => ({
                            ...styles.link,
                            ...(isActive ? styles.active : {})
                        })}
                    >
                        {link.label}
                    </NavLink>
                ))}
            </div>

        </nav>
    );
}

const styles = {
    nav: {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: 56,
        background: 'rgba(2, 6, 23, 0.9)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '0 24px',
        zIndex: 100,
    },
    brand: {
        fontSize: 15,
        fontWeight: 800,
        background: 'linear-gradient(90deg, #ef4444, #f97316)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        marginRight: 16,
        whiteSpace: 'nowrap',
    },
    links: {
        display: 'flex',
        gap: 4,
    },
    link: {
        padding: '5px 12px',
        borderRadius: 6,
        fontSize: 13,
        fontWeight: 500,
        color: '#94a3b8',
        transition: 'all 0.15s',
    },
    active: {
        background: 'rgba(239, 68, 68, 0.12)',
        color: '#fca5a5',
    },
};