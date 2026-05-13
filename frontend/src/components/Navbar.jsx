import { NavLink } from 'react-router-dom';
import { Activity, Brain, FileText, Home, Orbit, Stethoscope } from 'lucide-react';

export default function Navbar() {
    const items = [
        { label: 'Home', href: '/', icon: Home },
        { label: 'Assessment', href: '/assessment', icon: Stethoscope },
        { label: 'Twin', href: '/twin', icon: Orbit },
        { label: 'About', href: '/about', icon: Activity },
        { label: 'Documents', href: '/documents', icon: FileText },
    ];

    return (
        <header className="fixed left-0 top-0 z-[1000] w-full border-b border-white/10 bg-[#080b0a]/90 backdrop-blur-xl">
            <nav className="mx-auto flex w-full max-w-7xl items-center gap-3 px-4 py-3 sm:px-6 lg:px-8" aria-label="Primary">
                <NavLink to="/" className="flex shrink-0 items-center gap-2 rounded-md border border-teal-300/20 bg-teal-300/10 px-3 py-2 text-sm font-semibold uppercase text-teal-100">
                    <Brain size={18} />
                    <span>NeuroAssess</span>
                </NavLink>

                <div className="flex min-w-0 flex-1 items-center gap-2 overflow-x-auto">
                    {items.map((item) => {
                        const IconComponent = item.icon;
                        return (
                            <NavLink
                                key={item.href}
                                to={item.href}
                                className={({ isActive }) =>
                                    `inline-flex shrink-0 items-center gap-2 rounded-md border px-3 py-2 text-sm font-semibold transition ${
                                        isActive
                                            ? 'border-emerald-300/40 bg-emerald-300/15 text-emerald-100'
                                            : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/20 hover:bg-white/[0.06] hover:text-white'
                                    }`
                                }
                            >
                                <IconComponent size={15} />
                                <span>{item.label}</span>
                            </NavLink>
                        );
                    })}
                </div>
            </nav>
        </header>
    );
}
