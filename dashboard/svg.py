"""Small fixed-size SVG charts drawn on the server (spec section 5).

Pure: no database. Every embedded text is escaped.
"""
from html import escape

W, H = 480, 350
L, R, T, B = 60, 16, 44, 64        # margins: left, right, top, bottom
PW, PH = W - L - R, H - T - B        # plot width and height


def _svg(body: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">'
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
            f'<text x="{W / 2}" y="20" text-anchor="middle" font-size="13" '
            f'fill="#222">{escape(title)}</text>' + "".join(body) + "</svg>")


def _no_data(title: str) -> str:
    return _svg([f'<text x="{W / 2}" y="{H / 2}" text-anchor="middle" '
                 'fill="#888">no data</text>'], title)


def calibration_svg(reliability: list[dict], ece: float | None, title: str) -> str:
    """Predicted probability (x) against observed frequency (y)."""
    bins = [b for b in reliability if b.get("n")]
    if not bins:
        return _no_data(title)
    x = lambda p: L + p * PW
    y = lambda p: T + (1 - p) * PH
    biggest = max(b["n"] for b in bins)
    body = [
        f'<rect x="{L}" y="{T}" width="{PW}" height="{PH}" fill="none" stroke="#bbb"/>',
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#999" '
        'stroke-dasharray="4 4"/>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        body.append(f'<text x="{x(tick)}" y="{H - B + 14}" text-anchor="middle" '
                    f'fill="#444">{tick:.2f}</text>')
        body.append(f'<text x="{L - 6}" y="{y(tick) + 4}" text-anchor="end" '
                    f'fill="#444">{tick:.2f}</text>')
    body.append(f'<text x="{L + PW / 2}" y="{H - 8}" text-anchor="middle" '
                'fill="#444">predicted probability</text>')
    body.append(f'<text x="12" y="{T + PH / 2}" text-anchor="middle" fill="#444" '
                f'transform="rotate(-90 12 {T + PH / 2})">how often it happened</text>')
    for b in bins:
        r = 3 + 9 * (b["n"] / biggest) ** 0.5
        body.append(f'<circle cx="{x(b["mean_p"]):.1f}" cy="{y(b["freq"]):.1f}" '
                    f'r="{r:.1f}" fill="#2a6fdb" fill-opacity="0.7"/>')
    if ece is not None:
        body.append(f'<text x="{L + 8}" y="{T + 16}" fill="#222">ECE {ece:.3f}</text>')
    return _svg(body, title)


def skill_svg(per_fold: list[dict], skill: float, lo: float | None,
              hi: float | None, title: str) -> str:
    """One bar per test period: skill against the base rate."""
    if not per_fold:
        return _no_data(title)
    values = [f["skill"] for f in per_fold]
    top = max(0.005, max(values))
    bottom = min(-0.005, min(values))
    y = lambda v: T + (top - v) / (top - bottom) * PH
    zero = y(0.0)
    step = PW / len(per_fold)
    body = [f'<line class="zero" x1="{L}" y1="{zero:.1f}" x2="{L + PW}" '
            f'y2="{zero:.1f}" stroke="#444"/>']
    for v in (top, bottom):
        body.append(f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end" '
                    f'fill="#444">{v:+.1%}</text>')
    for i, f in enumerate(per_fold):
        v = f["skill"]
        top_y, bar_h = (y(v), zero - y(v)) if v >= 0 else (zero, y(v) - zero)
        colour = "#2e9d5b" if v >= 0 else "#d1453b"
        bx = L + i * step + step * 0.15
        body.append(f'<rect class="bar" x="{bx:.1f}" y="{top_y:.1f}" '
                    f'width="{step * 0.7:.1f}" height="{max(bar_h, 0.5):.1f}" '
                    f'fill="{colour}"/>')
        body.append(f'<text x="{bx + step * 0.35:.1f}" y="{H - B + 14}" '
                    f'text-anchor="end" fill="#444" font-size="9" '
                    f'transform="rotate(-45 {bx + step * 0.35:.1f} {H - B + 14})">'
                    f'{escape(f["test_start"][:7])}</text>')
    ci = "" if lo is None or hi is None else f" (95% CI {lo:+.1%} to {hi:+.1%})"
    body.append(f'<text x="{W / 2}" y="{T - 4}" text-anchor="middle" '
                f'fill="#444">pooled skill {skill:+.1%}{ci}</text>')
    return _svg(body, title)
