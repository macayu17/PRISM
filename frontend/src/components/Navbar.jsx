import { useLocation } from 'react-router-dom';
import PillNav from './PillNav';

export default function Navbar() {
    const location = useLocation();

    const items = [
        { label: 'Home', href: '/' },
        { label: 'Assessment', href: '/assessment' },
        { label: 'Twin', href: '/twin' },
        { label: 'About', href: '/about' },
        { label: 'Documents', href: '/documents' },
    ];

    return (
        <PillNav
            logo="/vite.svg"
            logoAlt="NeuroAssess"
            items={items}
            activeHref={location.pathname}
            baseColor="rgba(15, 15, 15, 0.75)"
            pillColor="rgba(255, 255, 255, 0.08)"
            pillTextColor="#94a3b8"
            hoveredPillTextColor="#ffffff"
            initialLoadAnimation={true}
        />
    );
}
