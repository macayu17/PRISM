import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { gsap } from 'gsap';

export type PillNavItem = {
    label: string;
    href: string;
    ariaLabel?: string;
};

export interface PillNavProps {
    logo: string;
    logoAlt?: string;
    items: PillNavItem[];
    activeHref?: string;
    className?: string;
    ease?: string;
    baseColor?: string;
    pillColor?: string;
    hoveredPillTextColor?: string;
    pillTextColor?: string;
    onMobileMenuClick?: () => void;
    initialLoadAnimation?: boolean;
}

const PillNav: React.FC<PillNavProps> = ({
    logo,
    logoAlt = 'Logo',
    items,
    activeHref,
    className = '',
    ease = 'power3.easeOut',
    baseColor = '#fff',
    pillColor = '#060010',
    hoveredPillTextColor = '#060010',
    pillTextColor,
    onMobileMenuClick,
    initialLoadAnimation = true
}) => {
    const resolvedPillTextColor = pillTextColor ?? baseColor;
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const circleRefs = useRef<Array<HTMLSpanElement | null>>([]);
    const tlRefs = useRef<Array<gsap.core.Timeline | null>>([]);
    const activeTweenRefs = useRef<Array<gsap.core.Tween | null>>([]);
    const logoImgRef = useRef<HTMLImageElement | null>(null);
    const logoTweenRef = useRef<gsap.core.Tween | null>(null);
    const hamburgerRef = useRef<HTMLButtonElement | null>(null);
    const mobileMenuRef = useRef<HTMLDivElement | null>(null);
    const navItemsRef = useRef<HTMLDivElement | null>(null);
    const logoRef = useRef<HTMLAnchorElement | HTMLElement | null>(null);

    useEffect(() => {
        const layout = () => {
            circleRefs.current.forEach(circle => {
                if (!circle?.parentElement) return;

                const pill = circle.parentElement as HTMLElement;
                const rect = pill.getBoundingClientRect();
                const { width: w, height: h } = rect;
                if (w === 0 || h === 0) return;
                const R = ((w * w) / 4 + h * h) / (2 * h);
                const D = Math.ceil(2 * R) + 2;
                const delta = Math.ceil(R - Math.sqrt(Math.max(0, R * R - (w * w) / 4))) + 1;
                const originY = D - delta;

                circle.style.width = `${D}px`;
                circle.style.height = `${D}px`;
                circle.style.bottom = `-${delta}px`;

                gsap.set(circle, {
                    xPercent: -50,
                    scale: 0,
                    transformOrigin: `50% ${originY}px`
                });

                const label = pill.querySelector<HTMLElement>('.pill-label');
                const white = pill.querySelector<HTMLElement>('.pill-label-hover');

                if (label) gsap.set(label, { y: 0 });
                if (white) gsap.set(white, { y: h + 12, opacity: 0 });

                const index = circleRefs.current.indexOf(circle);
                if (index === -1) return;

                tlRefs.current[index]?.kill();
                const tl = gsap.timeline({ paused: true });

                tl.to(circle, { scale: 1.2, xPercent: -50, duration: 2, ease, overwrite: 'auto' }, 0);

                if (label) {
                    tl.to(label, { y: -(h + 8), duration: 2, ease, overwrite: 'auto' }, 0);
                }

                if (white) {
                    gsap.set(white, { y: Math.ceil(h + 100), opacity: 0 });
                    tl.to(white, { y: 0, opacity: 1, duration: 2, ease, overwrite: 'auto' }, 0);
                }

                tlRefs.current[index] = tl;
            });
        };

        layout();

        const onResize = () => layout();
        window.addEventListener('resize', onResize);

        if (document.fonts) {
            document.fonts.ready.then(layout).catch(() => { });
        }

        const menu = mobileMenuRef.current;
        if (menu) {
            gsap.set(menu, { visibility: 'hidden', opacity: 0, scaleY: 1, y: 0 });
        }

        if (initialLoadAnimation) {
            const logoEl = logoRef.current;
            const navItems = navItemsRef.current;

            if (logoEl) {
                gsap.set(logoEl, { scale: 0 });
                gsap.to(logoEl, { scale: 1, duration: 0.6, ease });
            }

            if (navItems) {
                gsap.set(navItems, { width: 0, overflow: 'hidden' });
                gsap.to(navItems, { width: 'auto', duration: 0.6, ease });
            }
        }

        return () => window.removeEventListener('resize', onResize);
    }, [items, ease, initialLoadAnimation]);

    const handleEnter = (i: number) => {
        const tl = tlRefs.current[i];
        if (!tl) return;
        activeTweenRefs.current[i]?.kill();
        activeTweenRefs.current[i] = tl.tweenTo(tl.duration(), {
            duration: 0.3, ease, overwrite: 'auto'
        });
    };

    const handleLeave = (i: number) => {
        const tl = tlRefs.current[i];
        if (!tl) return;
        activeTweenRefs.current[i]?.kill();
        activeTweenRefs.current[i] = tl.tweenTo(0, {
            duration: 0.2, ease, overwrite: 'auto'
        });
    };

    const handleLogoEnter = () => {
        const img = logoImgRef.current;
        if (!img) return;
        logoTweenRef.current?.kill();
        gsap.set(img, { rotate: 0 });
        logoTweenRef.current = gsap.to(img, {
            rotate: 360, duration: 0.4, ease, overwrite: 'auto'
        });
    };

    const toggleMobileMenu = () => {
        const newState = !isMobileMenuOpen;
        setIsMobileMenuOpen(newState);

        const hamburger = hamburgerRef.current;
        const menu = mobileMenuRef.current;

        if (hamburger) {
            const lines = hamburger.querySelectorAll('.hamburger-line');
            if (newState) {
                gsap.to(lines[0], { rotation: 45, y: 3, duration: 0.3, ease });
                gsap.to(lines[1], { rotation: -45, y: -3, duration: 0.3, ease });
            } else {
                gsap.to(lines[0], { rotation: 0, y: 0, duration: 0.3, ease });
                gsap.to(lines[1], { rotation: 0, y: 0, duration: 0.3, ease });
            }
        }

        if (menu) {
            if (newState) {
                gsap.set(menu, { visibility: 'visible' });
                gsap.fromTo(menu,
                    { opacity: 0, y: -8 },
                    { opacity: 1, y: 0, duration: 0.3, ease, transformOrigin: 'top center' }
                );
            } else {
                gsap.to(menu, {
                    opacity: 0, y: -8, duration: 0.2, ease,
                    transformOrigin: 'top center',
                    onComplete: () => gsap.set(menu, { visibility: 'hidden' })
                });
            }
        }

        onMobileMenuClick?.();
    };

    const isExternalLink = (href: string) =>
        href.startsWith('http://') || href.startsWith('https://') ||
        href.startsWith('//') || href.startsWith('mailto:') ||
        href.startsWith('tel:') || href.startsWith('#');

    const isRouterLink = (href?: string) => href && !isExternalLink(href);

    const cssVars = {
        ['--base']: baseColor,
        ['--pill-bg']: pillColor,
        ['--hover-text']: hoveredPillTextColor,
        ['--pill-text']: resolvedPillTextColor,
        ['--nav-h']: '42px',
        ['--pill-pad-x']: '18px',
        ['--pill-gap']: '3px'
    } as React.CSSProperties;

    const pillStyle: React.CSSProperties = {
        background: 'var(--pill-bg)',
        color: 'var(--pill-text)',
    };

    return (
        <div
            className={`pointer-events-none fixed left-0 top-0 z-[1000] flex w-full justify-center px-4 pt-4 ${className}`}
            style={cssVars}
        >
            <nav className="pointer-events-auto flex w-full items-center justify-between md:w-auto md:justify-center md:gap-1.5" aria-label="Primary">

                {/* ---- Logo ---- */}
                {isRouterLink(items?.[0]?.href) ? (
                    <Link
                        to={items[0].href}
                        aria-label="Home"
                        onMouseEnter={handleLogoEnter}
                        ref={el => { logoRef.current = el; }}
                        className="inline-flex h-[var(--nav-h)] w-[var(--nav-h)] flex-none items-center justify-center overflow-hidden rounded-full bg-[var(--base)] p-[0.35rem] no-underline backdrop-blur-xl"
                    >
                        <img src={logo} alt={logoAlt} ref={logoImgRef} className="block h-[22px] w-[22px] object-contain" />
                    </Link>
                ) : (
                    <a
                        href={items?.[0]?.href || '#'}
                        aria-label="Home"
                        onMouseEnter={handleLogoEnter}
                        ref={el => { logoRef.current = el; }}
                        className="inline-flex h-[var(--nav-h)] w-[var(--nav-h)] flex-none items-center justify-center overflow-hidden rounded-full bg-[var(--base)] p-[0.35rem] no-underline backdrop-blur-xl"
                    >
                        <img src={logo} alt={logoAlt} ref={logoImgRef} className="block h-[22px] w-[22px] object-contain" />
                    </a>
                )}

                {/* ---- Desktop pill row ---- */}
                <div
                    ref={navItemsRef}
                    className="relative hidden h-[var(--nav-h)] items-center rounded-full bg-[var(--base)] backdrop-blur-xl md:flex"
                >
                    <ul role="menubar" className="m-0 flex h-full list-none items-stretch gap-[var(--pill-gap)] p-[3px]">
                        {items.map((item, i) => {
                            const isActive = activeHref === item.href;

                            const PillContent = (
                                <>
                                    <span
                                        className="hover-circle absolute bottom-0 left-1/2 z-[1] block rounded-full bg-[var(--base)] will-change-transform"
                                        aria-hidden="true"
                                        ref={el => { circleRefs.current[i] = el; }}
                                    />
                                    <span className="label-stack relative z-[2] inline-block leading-none">
                                        <span className="pill-label relative z-[2] inline-block leading-none will-change-transform">{item.label}</span>
                                        <span className="pill-label-hover absolute left-0 top-0 z-[3] inline-block text-[var(--hover-text)] will-change-[transform,opacity]" aria-hidden="true">
                                            {item.label}
                                        </span>
                                    </span>
                                    {isActive && (
                                        <span className="absolute -bottom-1.5 left-1/2 z-[4] h-1.5 w-1.5 -translate-x-1/2 rounded-full bg-[var(--base)]" aria-hidden="true" />
                                    )}
                                </>
                            );

                            return (
                                <li key={item.href} role="none" className="flex h-full">
                                    {isRouterLink(item.href) ? (
                                        <Link
                                            role="menuitem"
                                            to={item.href}
                                            className="relative inline-flex h-full cursor-pointer items-center justify-center overflow-hidden rounded-full px-[var(--pill-pad-x)] font-display text-base font-semibold uppercase leading-none tracking-[0.4px] whitespace-nowrap no-underline transition-colors"
                                            style={pillStyle}
                                            aria-label={item.ariaLabel || item.label}
                                            onMouseEnter={() => handleEnter(i)}
                                            onMouseLeave={() => handleLeave(i)}
                                        >
                                            {PillContent}
                                        </Link>
                                    ) : (
                                        <a
                                            role="menuitem"
                                            href={item.href}
                                            className="relative inline-flex h-full cursor-pointer items-center justify-center overflow-hidden rounded-full px-[var(--pill-pad-x)] font-display text-base font-semibold uppercase leading-none tracking-[0.4px] whitespace-nowrap no-underline transition-colors"
                                            style={pillStyle}
                                            aria-label={item.ariaLabel || item.label}
                                            onMouseEnter={() => handleEnter(i)}
                                            onMouseLeave={() => handleLeave(i)}
                                        >
                                            {PillContent}
                                        </a>
                                    )}
                                </li>
                            );
                        })}
                    </ul>
                </div>

                {/* ---- Hamburger (mobile) ---- */}
                <button
                    ref={hamburgerRef}
                    onClick={toggleMobileMenu}
                    type="button"
                    aria-label="Toggle menu"
                    aria-expanded={isMobileMenuOpen}
                    className="flex h-[var(--nav-h)] w-[var(--nav-h)] flex-col items-center justify-center gap-1 rounded-full bg-[var(--base)] p-0 backdrop-blur-xl md:hidden"
                >
                    <span className="hamburger-line h-0.5 w-4 rounded-full bg-[var(--pill-text)] transition-all duration-300 ease-out" />
                    <span className="hamburger-line h-0.5 w-4 rounded-full bg-[var(--pill-text)] transition-all duration-300 ease-out" />
                </button>
            </nav>

            {/* ---- Mobile dropdown ---- */}
            <div
                ref={mobileMenuRef}
                className="pointer-events-auto fixed left-4 right-4 top-[calc(1rem+var(--nav-h)+0.5rem)] z-[999] rounded-[20px] bg-[var(--base)] shadow-[0_8px_32px_rgba(0,0,0,0.25)] backdrop-blur-xl md:hidden"
                style={{ ...cssVars, background: 'var(--base)' }}
            >
                <ul className="m-0 flex list-none flex-col gap-[3px] p-1.5">
                    {items.map(item => {
                        const defaultStyle: React.CSSProperties = {
                            background: 'var(--pill-bg)',
                            color: 'var(--pill-text)'
                        };
                        const hoverIn = (e: React.MouseEvent<HTMLAnchorElement>) => {
                            e.currentTarget.style.background = 'var(--base)';
                            e.currentTarget.style.color = 'var(--hover-text)';
                        };
                        const hoverOut = (e: React.MouseEvent<HTMLAnchorElement>) => {
                            e.currentTarget.style.background = 'var(--pill-bg)';
                            e.currentTarget.style.color = 'var(--pill-text)';
                        };

                        return (
                            <li key={item.href}>
                                {isRouterLink(item.href) ? (
                                    <Link
                                        to={item.href}
                                        className="block rounded-[14px] px-5 py-3 text-[15px] font-medium no-underline transition-all duration-200 ease-out"
                                        style={defaultStyle}
                                        onMouseEnter={hoverIn}
                                        onMouseLeave={hoverOut}
                                        onClick={() => { setIsMobileMenuOpen(false); toggleMobileMenu(); }}
                                    >
                                        {item.label}
                                    </Link>
                                ) : (
                                    <a
                                        href={item.href}
                                        className="block rounded-[14px] px-5 py-3 text-[15px] font-medium no-underline transition-all duration-200 ease-out"
                                        style={defaultStyle}
                                        onMouseEnter={hoverIn}
                                        onMouseLeave={hoverOut}
                                        onClick={() => { setIsMobileMenuOpen(false); toggleMobileMenu(); }}
                                    >
                                        {item.label}
                                    </a>
                                )}
                            </li>
                        );
                    })}
                </ul>
            </div>
        </div>
    );
};

export default PillNav;
