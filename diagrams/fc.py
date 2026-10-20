"""Flat-card diagram kit: the approved dark SVG style for the series.

Usage: d = Diagram(...); draw with helpers; d.save(outdir, name).
Fonts: Nunito (weights 400, 800, italic 400) is subset to the glyphs used and
embedded in every SVG as base64 WOFF, so PNGs render identically anywhere.
"""
import base64, io, os, re
from fontTools.ttLib import TTFont
from fontTools import subset
from PIL import ImageFont
from playwright.sync_api import sync_playwright

MARGIN = 20  # breathing room between the panel edge and the content
RADIUS = 28  # outer corner radius of the whole panel; pixels outside stay transparent
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Nunito, Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
BG, TEXT, MUTED, LINE, INK = '#2b2e34', '#e3e5e8', '#bcc2ca', '#868d97', '#2b2e34'
ACC = {  # accent, tint
    'neutral': ('#9aa0a8', '#353941'),
    'blue': ('#7f9bb5', '#3a424b'),
    'sage': ('#8fb39a', '#3d4646'),
    'teal': ('#5ba4ae', '#34434a'),
    'sand': ('#c9a96e', '#47443e'),
}
_FONTFILES = {(400, False): 'Nunito-400.ttf', (600, False): 'Nunito-600.ttf', (800, False): 'Nunito-800.ttf', (400, True): 'Nunito-400i.ttf'}
_pil = {}


def tw(text, size, weight=800, italic=False, ls=0.0):
    key = (weight, italic)
    if key not in _pil:
        _pil[key] = ImageFont.truetype(os.path.join(HERE, 'fonts', _FONTFILES[key]), 200)
    return _pil[key].getlength(text) * size / 200 + ls * len(text)


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def wrap(text, size, maxw, weight=400):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if tw(t, size, weight) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Diagram:
    def __init__(self, title, subtitle, desc, W=1280, bar='sand'):
        self.W, self.title, self.desc = W, title, desc
        self.o, self.used = [], {}
        self.sz = dict(title=32, sub=18, ctitle=26, cdesc=16, tag=17, chip=15, circ=17)
        self._head = (title, subtitle, bar)

    # ---- primitives -------------------------------------------------
    def raw(self, s):
        self.o.append(s)

    def t(self, x, y, s, size=16, weight=400, fill=TEXT, anchor='middle', italic=False, ls=0, opacity=None):
        if fill == MUTED and weight == 400 and not italic:
            weight = 600  # secondary text: a touch heavier than regular, lighter than the 800 primary text
        self.used.setdefault((weight, italic), set()).update(s)
        extra = (f' letter-spacing="{ls}"' if ls else '') + (f' fill-opacity="{opacity}"' if opacity is not None else '')
        if ls and anchor == 'middle':
            x += ls / 2
        st = ' font-style="italic"' if italic else ''
        w = f' font-weight="{weight}"' if weight != 400 else ''
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"{w}{st}{extra} fill="{fill}">{esc(s)}</text>')

    def card(self, x, y, w, h, key, r=26):
        a, tint = ACC[key]
        self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{tint}" stroke="{a}" stroke-opacity=".25"/>')

    def header(self, cx, cy, n, tag, key):
        a = ACC[key][0]
        z = self.sz
        r = z['circ']
        gw = 2 * r + 9 + tw(tag, z['tag'], 800, ls=2.2)
        gx = cx - gw / 2
        self.o.append(f'<circle cx="{gx+r:.1f}" cy="{cy}" r="{r}" fill="{a}"/>')
        self.t(gx + r, cy + r * .35, str(n), r + 2, 800, BG)
        self.t(gx + 2 * r + 9, cy + z['tag'] * .35, tag, z['tag'], 800, a, 'start', ls=2.2)

    def chip(self, x, y, w, h, key, text, style='outline', fs=None, a=None):
        fs = fs or self.sz['chip']
        a = a or ACC[key][0]
        ty = y + h / 2 + fs * .36
        if style == 'outline':
            self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{BG}" stroke="{a}" stroke-width="2.8"/>')
            self.t(x + w / 2, ty, text, fs, 800, a)
        elif style == 'ghost':
            self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="none" stroke="{a}" stroke-width="2.8" stroke-dasharray="7 6"/>')
            self.t(x + w / 2, ty, text, fs, 800, a)
        elif style == 'dim':
            self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{BG}" stroke="{a}" stroke-opacity=".5" stroke-width="2.8"/>')
            self.t(x + w / 2, ty, text, fs, 800, a, opacity=.6)
        else:  # solid
            self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{a}"/>')
            self.t(x + w / 2, ty, text, fs, 800, INK)

    def block(self, x, y, w, h, key, title, sub=None, style='outline', fs=16, subfs=13.5, r=14):
        """Dark block, accent border, bold accent title and optional muted sub (wrapped)."""
        a = ACC[key][0]
        if style == 'ghost':
            self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="none" stroke="{a}" stroke-width="2.8" stroke-dasharray="7 6"/>')
        else:
            self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{BG}" stroke="{a}" stroke-width="2.8"/>')
        lines = wrap(sub, subfs, w - 28) if sub else []
        lh = subfs * 1.3
        total = fs * 1.2 + (6 + lh * len(lines) if lines else 0)
        top = y + (h - total) / 2
        self.t(x + w / 2, top + fs * .95, title, fs, 800, a)
        for i, ln in enumerate(lines):
            self.t(x + w / 2, top + fs * 1.2 + 6 + lh * (i + .8), ln, subfs, 400, MUTED)

    def pill(self, cx, cy, text, fs=15, h=32, stroke=LINE):
        w = tw(text, fs, 800) + 32
        self.o.append(f'<rect x="{cx-w/2:.1f}" y="{cy-h/2}" width="{w:.1f}" height="{h}" rx="{h/2}" fill="{BG}" stroke="{stroke}" stroke-width="2.5"/>')
        self.t(cx, cy + fs * .36, text, fs, 800, TEXT)

    def arrow(self, pts, r=24, head=7, color=LINE, sw=3, dash='11 9'):
        """Dashed polyline with rounded corners; open chevron head with its tip on the last point."""
        import math
        (hx, hy), (px, py) = pts[-1], pts[-2]
        L = math.hypot(hx - px, hy - py)
        ux, uy = (hx - px) / L, (hy - py) / L
        end = (hx - ux * 3, hy - uy * 3)
        P = pts[:-1] + [end]
        d = f'M{P[0][0]:.1f} {P[0][1]:.1f}'
        for i in range(1, len(P) - 1):
            a, b, c = P[i - 1], P[i], P[i + 1]
            l1, l2 = math.hypot(b[0] - a[0], b[1] - a[1]), math.hypot(c[0] - b[0], c[1] - b[1])
            rr = min(r, l1 / 2, l2 / 2)
            p1 = (b[0] - (b[0] - a[0]) / l1 * rr, b[1] - (b[1] - a[1]) / l1 * rr)
            p2 = (b[0] + (c[0] - b[0]) / l2 * rr, b[1] + (c[1] - b[1]) / l2 * rr)
            d += f' L{p1[0]:.1f} {p1[1]:.1f} Q{b[0]:.1f} {b[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}'
        d += f' L{P[-1][0]:.1f} {P[-1][1]:.1f}'
        self.o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-dasharray="{dash}" stroke-linecap="round" stroke-linejoin="round"/>')
        bx, by = hx - ux * head, hy - uy * head
        nx, ny = -uy, ux
        self.o.append(f'<path d="M{bx+nx*head:.1f} {by+ny*head:.1f} L{hx:.1f} {hy:.1f} L{bx-nx*head:.1f} {by-ny*head:.1f}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')

    # ---- layouts ----------------------------------------------------
    def step_cards(self, steps, cw, gap, scenes, top=128, ch=410, with_q=True):
        """Row of tinted cards: header, scene, title, description, (divider, question). Returns (xs, top, bottom)."""
        n = len(steps)
        pad = (self.W - n * cw - (n - 1) * gap) / 2
        xs = [pad + i * (cw + gap) for i in range(n)]
        for i, (x, s) in enumerate(zip(xs, steps)):
            num, tag, title, desc, q, key = s
            cx = x + cw / 2
            self.card(x, top, cw, ch, key)
            self.header(cx, top + 36, num, tag, key)
            sy = top + (158 if with_q else 150)
            scenes[i](self, cx, sy, key)
            ty = top + (290 if with_q else 262)
            self.t(cx, ty, title, self.sz['ctitle'], 800, TEXT)
            self.t(cx, ty + self.sz['ctitle'] + 5, desc, self.sz['cdesc'], 400, MUTED)
            if with_q:
                self.o.append(f'<line x1="{x+28}" y1="{top+345}" x2="{x+cw-28}" y2="{top+345}" stroke="#fff" stroke-opacity=".12"/>')
                self.t(cx, top + 378, q, 16, 400, ACC[key][0], italic=True)
            if i < n - 1:
                self.arrow([(x + cw + 1, top + sy_off(with_q)), (x + cw + gap - 1, top + sy_off(with_q))])
        return xs, top, top + ch

    # ---- output -----------------------------------------------------
    def font_css(self):
        css = ''
        for (wt, it), chars in sorted(self.used.items()):
            chars = set(chars) | {' ', ' '}
            f = TTFont(os.path.join(HERE, 'fonts', _FONTFILES[(wt, it)]))
            opts = subset.Options()
            opts.flavor = 'woff'
            opts.layout_features = ['kern', 'liga', 'calt', 'ccmp', 'locl', 'mark', 'mkmk']
            opts.name_IDs = [1, 2]
            opts.notdef_outline = True
            sub = subset.Subsetter(opts)
            sub.populate(text=''.join(sorted(chars)))
            sub.subset(f)
            buf = io.BytesIO()
            f.flavor = 'woff'
            f.save(buf)
            b64 = base64.b64encode(buf.getvalue()).decode()
            css += (f"@font-face{{font-family:'Nunito';font-weight:{wt};font-style:{'italic' if it else 'normal'};"
                    f"src:url(data:font/woff;base64,{b64}) format('woff')}}")
        return css

    def _draw_head(self):
        title, subtitle, bar = self._head
        z = self.sz
        k = z['title'] / 32
        saved, self.o = self.o, []
        self.t(32, 52 * k, title, z['title'], 800, TEXT, 'start')
        self.t(33, 52 * k + 30 * k, subtitle, z['sub'], 400, MUTED, 'start')
        self.o.append(f'<rect x="33" y="{52*k+42*k:.1f}" width="54" height="5" rx="2.5" fill="{ACC[bar][0]}"/>')
        self.o = self.o + saved

    def save(self, outdir, name, height, png=True):
        self._draw_head()
        os.makedirs(outdir, exist_ok=True)
        W, H = self.W + 2 * MARGIN, height + 2 * MARGIN
        body = ''.join(self.o)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
               f'font-family="{FONT}" role="img"><title>{esc(self.title)}</title><desc>{esc(self.desc)}</desc>'
               f'<style>{self.font_css()}</style><rect width="{W}" height="{H}" rx="{RADIUS}" fill="{BG}"/><g transform="translate({MARGIN},{MARGIN})">{body}</g></svg>')
        path = os.path.join(outdir, name + '.svg')
        open(path, 'w', encoding='utf-8').write(svg)
        import xml.dom.minidom
        xml.dom.minidom.parse(path)
        if png:
            with sync_playwright() as p:
                b = p.chromium.launch()
                pg = b.new_page(viewport={'width': W, 'height': H}, device_scale_factor=1.5)
                pg.goto('file://' + os.path.abspath(path))
                pg.evaluate('document.fonts.ready')
                pg.screenshot(path=os.path.join(outdir, name + '.png'), clip={'x': 0, 'y': 0, 'width': W, 'height': H}, omit_background=True)
                b.close()
        return path


def sy_off(with_q):
    return 158 if with_q else 150
