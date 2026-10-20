"""Builds the series diagrams (flat dark card style) into each article's images/ folder.

    python3 diagrams/build.py            # all ten
    python3 diagrams/build.py a1_07      # one
Needs: pip install fonttools pillow playwright (and playwright install chromium).
Fonts: Nunito (SIL OFL) static instances live in diagrams/fonts/ and are subset + embedded in each SVG.
Captions are NOT drawn in the image: they are collected in CAPTIONS and written to images/captions.md
(paste them under the image on Medium, and keep the matching italic line under the image in the README).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fc import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A1 = os.path.join(ROOT, 'article-01-how-transformer-llms-work', 'images')
A2 = os.path.join(ROOT, 'article-02-prompting-fundamentals-for-llms', 'images')
CAPTIONS = {}



def slabs(d, cx, sy, key, labels, w=230, h=36, g=15, fs=15):
    n = len(labels)
    for k, l in enumerate(labels):
        y = sy - (n * h + (n - 1) * g) / 2 + k * (h + g)
        d.chip(cx - w / 2, y, w, h, key, l, 'outline', fs=fs)


def tokens(d, cx, sy, key, labels, styles, w=70, h=54, g=10, dy=0):
    n = len(labels)
    x0 = cx - (n * w + (n - 1) * g) / 2
    for k, (l, s) in enumerate(zip(labels, styles)):
        d.chip(x0 + k * (w + g), sy - h / 2 + dy, w, h, key, l, s)
    return x0, sy - h / 2 + dy, w, g


# ------------------------------------------------------------------ Article 1
def a1_01():
    d = Diagram('From text to next token', 'Five stages, run once for every token the model writes.',
                'Text becomes tokens, tokens become embeddings, embeddings flow through a stack of transformer blocks, and the model outputs a next token.',
                bar='blue')

    def s_text(d, cx, sy, k): d.chip(cx - 88, sy - 23, 176, 46, k, 'Self-attention', 'outline')

    def s_tok(d, cx, sy, k): tokens(d, cx, sy, k, ['15352', '12', '6207'], ['outline'] * 3, w=58, h=46, g=6)

    def s_vec(d, cx, sy, k):
        a = ACC[k][0]
        for j in range(3):
            x = cx - 84 + j * 62
            d.o.append(f'<rect x="{x}" y="{sy-50}" width="50" height="100" rx="13" fill="{BG}" stroke="{a}" stroke-width="2.8"/>')
            for r, wd in enumerate([[30, 18, 26, 12], [14, 30, 20, 26], [26, 12, 30, 18]][j]):
                d.o.append(f'<rect x="{x+25-wd/2}" y="{sy-34+r*20}" width="{wd}" height="8" rx="4" fill="{a}" fill-opacity=".75"/>')

    def s_blocks(d, cx, sy, k): slabs(d, cx, sy, k, ['Block N', 'Block 2', 'Block 1'], w=176, h=34, g=12)

    def s_out(d, cx, sy, k): d.chip(cx - 40, sy - 27, 80, 54, k, 'is', 'solid', fs=18)

    steps = [(1, 'TEXT', 'Text', 'what you type', '', 'neutral'),
             (2, 'TOKENS', 'Tokens', 'integer IDs', '', 'blue'),
             (3, 'VECTORS', 'Embeddings', 'a vector per token', '', 'sage'),
             (4, 'MODEL', 'Transformer', '×N blocks', '', 'teal'),
             (5, 'OUTPUT', 'Next token', 'sampled', '', 'sand')]
    xs, top, bot = d.step_cards(steps, 214, 40, [s_text, s_tok, s_vec, s_blocks, s_out], ch=340, with_q=False)
    CAPTIONS['a1/01-pipeline-overview'] = 'Run this once per output token, feeding each new token back in, and you get generation.'
    return d.save(A1, '01-pipeline-overview', bot + 24)


def a1_02():
    d = Diagram('Encoder blocks versus decoder blocks', 'GPT-style LLMs keep only the decoder side.',
                "A side-by-side comparison. The encoder block uses self-attention that sees the whole sequence, then a feed-forward network. The decoder block uses masked self-attention that only sees earlier tokens, a cross-attention layer that attends to the encoder's output (dropped in decoder-only models like GPT), and a feed-forward network. An arrow shows the encoder's output feeding the decoder's cross-attention layer.",
                bar='teal')
    cw, ch, top, gap = 520, 470, 128, 160
    x1 = (1280 - 2 * cw - gap) / 2
    x2 = x1 + cw + gap
    d.card(x1, top, cw, ch, 'blue')
    d.card(x2, top, cw, ch, 'teal')
    d.t(x1 + cw / 2, top + 52, 'Encoder block', 26, 800, TEXT)
    d.t(x1 + cw / 2, top + 82, 'reads the whole input at once', 16, 400, MUTED)
    d.t(x2 + cw / 2, top + 52, 'Decoder block', 26, 800, TEXT)
    d.t(x2 + cw / 2, top + 82, 'writes one token at a time', 16, 400, MUTED)
    bw, bh, bg = 400, 74, 40
    y0 = top + 124
    ys = [y0 + i * (bh + bg) for i in range(3)]
    ex, dx = x1 + (cw - bw) / 2, x2 + (cw - bw) / 2
    d.block(ex, ys[0], bw, bh, 'blue', 'Self-attention', 'sees the whole sequence')
    d.block(ex, ys[1], bw, bh, 'blue', 'Feed-forward network', "same as the decoder's")
    d.block(dx, ys[0], bw, bh, 'teal', 'Masked self-attention', 'only sees earlier tokens')
    d.block(dx, ys[1], bw, bh, 'teal', 'Cross-attention', 'dropped in decoder-only LLMs', style='ghost')
    d.block(dx, ys[2], bw, bh, 'teal', 'Feed-forward network', "same as the encoder's")
    for x in (ex, dx):
        for i in range(2 if x == dx else 1):
            d.arrow([(x + bw / 2, ys[i] + bh + 1), (x + bw / 2, ys[i + 1] - 1)], head=6)
    d.arrow([(ex + bw + 1, ys[1] + bh / 2), (dx - 1, ys[1] + bh / 2)])
    d.pill((ex + bw + dx) / 2, ys[1] + bh / 2 - 30, 'encoder output', 14, 30)
    CAPTIONS['a1/02-encoder-vs-decoder'] = "GPT-style LLMs stack only decoder blocks, minus the dashed cross-attention layer: there's no encoder output left to attend to."
    return d.save(A1, '02-encoder-vs-decoder', top + ch + 24)


def a1_03():
    d = Diagram('How a sentence becomes token IDs', 'A sentence becomes subword tokens, and each token becomes one integer ID.',
                'The sentence "Self-attention is powerful" is split into six subword tokens, then each token is mapped to an integer token ID.',
                bar='sage')
    toks = ['Self', '-', 'attention', 'is', 'power', 'ful']
    ids = ['15352', '12', '6207', '318', '1176', '913']
    w = [max(tw(t, 18, 800), tw(i, 18, 800)) + 40 for t, i in zip(toks, ids)]
    gap = 16
    total = sum(w) + gap * 5
    left = 330
    area = 1232 - left - 24
    x0 = 24 + left + (area - total) / 2
    top, rh, rg = 128, 124, 44
    rows = [('blue', 1, 'TEXT', 'Raw sentence', None), ('sage', 2, 'SPLIT', 'Subword tokens', toks), ('teal', 3, 'LOOK UP', 'Integer IDs', ids)]
    for r, (key, n, tag, title, items) in enumerate(rows):
        y = top + r * (rh + rg)
        d.card(24, y, 1232, rh, key)
        d.header(24 + 165, y + 38, n, tag, key)
        d.t(24 + 165, y + 92, title, 22, 800, TEXT)
        cy = y + rh / 2
        if items is None:
            d.chip(24 + left + (area - 440) / 2, cy - 30, 440, 60, key, 'Self-attention is powerful', 'outline', fs=20)
        else:
            x = x0
            for k, it in enumerate(items):
                d.chip(x, cy - 28, w[k], 56, key, it, 'outline' if r == 1 else 'solid', fs=18)
                x += w[k] + gap
        if r < 2:
            d.arrow([(640, y + rh + 1), (640, y + rh + rg - 1)], head=7)
    CAPTIONS['a1/03-tokenization-example'] = 'The model never sees the string, only this list of integer IDs.'
    return d.save(A1, '03-tokenization-example', top + 3 * rh + 2 * rg + 24)


def a1_04():
    d = Diagram('Embeddings place tokens in a semantic space', 'A simplified two-dimensional view of where tokens land.',
                'A simplified two-dimensional view of word embeddings. Man, woman, king and queen sit along one row, with the man-to-woman and king-to-queen arrows pointing the same way. Cat and dog form a separate cluster elsewhere.',
                bar='teal')
    top, ch = 128, 470
    d.card(24, top, 1232, ch, 'neutral')
    px0, py0, px1, py1 = 110, top + 40, 1216, top + ch - 70
    d.arrow([(px0, py1), (px1, py1)], head=7, color='#6b727c')
    d.arrow([(px0, py1), (px0, py0)], head=7, color='#6b727c')
    d.t(px1, py1 + 38, 'royalty & gender', 16, 800, MUTED, 'end')
    d.o.append(f'<text transform="translate({px0-34},{py0+2}) rotate(-90)" text-anchor="end" font-size="16" font-weight="800" fill="{MUTED}">animals</text>')
    d.used.setdefault((800, False), set()).update('animals')
    row = py1 - 90
    pts = {'man': (250, row, 'blue'), 'woman': (480, row, 'blue'), 'king': (760, row, 'sage'), 'queen': (990, row, 'sage'),
           'cat': (480, py0 + 80, 'sand'), 'dog': (690, py0 + 80, 'sand')}
    for name, (x, y, k) in pts.items():
        w = tw(name, 18, 800) + 44
        d.chip(x - w / 2, y - 25, w, 50, k, name, 'outline', fs=18)
    for (a, b) in (('man', 'woman'), ('king', 'queen')):
        xa, ya, _ = pts[a]
        xb, yb, _ = pts[b]
        wa, wb = tw(a, 18, 800) + 44, tw(b, 18, 800) + 44
        d.arrow([(xa + wa / 2 + 1, ya), (xb - wb / 2 - 1, yb)], head=7)
        d.pill((xa + wa / 2 + xb - wb / 2) / 2, ya - 34, 'male → female', 14, 28)
    CAPTIONS['a1/04-embeddings-2d-space'] = 'Similar meanings sit close together, and consistent relationships (male → female) point the same way.'
    return d.save(A1, '04-embeddings-2d-space', top + ch + 24)


def a1_05():
    d = Diagram('Self-attention weights from the token “it”', 'Line thickness shows how much attention “it” pays to each word.',
                'The sentence "The cat sat because it was tired" with the word "it" attending to every other word. The line to "cat" is thickest, with 62% of the attention weight, which is how the model resolves what "it" refers to.',
                bar='sand')
    words = ['The', 'cat', 'sat', 'because', 'it', 'was', 'tired']
    wts = {'The': 3, 'cat': 62, 'sat': 8, 'because': 5, 'was': 7, 'tired': 15}
    top, ch = 128, 380
    d.card(24, top, d.W - 48, ch, 'neutral')
    ws = [tw(w, 20, 800) + 44 for w in words]
    gap = 84
    total = sum(ws) + gap * 6
    x = d.W / 2 - total / 2
    cy = top + 262
    cx_of = {}
    for w, wd in zip(words, ws):
        cx_of[w] = x + wd / 2
        x += wd + gap
    qx = cx_of['it']
    for w in words:
        if w == 'it':
            continue
        p = wts[w]
        sw = 2 + p / 62 * 12
        hi = w == 'cat'
        col = ACC['sage'][0] if hi else ACC['blue'][0]
        far = abs(cx_of[w] - qx)
        side = 1 if cx_of[w] > qx else -1
        sx = qx + side * 10
        ctrl = cy - 26 - (46 + far * .5)
        d.o.append(f'<path d="M{sx:.1f} {cy-27} Q{(sx+cx_of[w])/2:.1f} {ctrl:.1f} {cx_of[w]:.1f} {cy-27}" fill="none" stroke="{col}" stroke-width="{sw:.1f}" stroke-opacity="{.95 if hi else .6}" stroke-linecap="round"/>')
    for w, wd in zip(words, ws):
        c = cx_of[w]
        if w == 'it':
            d.chip(c - wd / 2, cy - 26, wd, 52, 'sand', w, 'solid', fs=20)
            d.t(c, cy + 58, 'query', 15, 400, ACC['sand'][0], italic=True)
        elif w == 'cat':
            d.chip(c - wd / 2, cy - 26, wd, 52, 'sage', w, 'outline', fs=20)
            d.t(c, cy + 58, f'{wts[w]}%', 18, 800, ACC['sage'][0])
        else:
            d.chip(c - wd / 2, cy - 26, wd, 52, 'blue', w, 'dim', fs=20)
            d.t(c, cy + 58, f'{wts[w]}%', 16, 800, MUTED)
    CAPTIONS['a1/05-self-attention-example'] = 'Most of it lands on “cat”, and that’s how the model resolves the pronoun.'
    return d.save(A1, '05-self-attention-example', top + ch + 24)


def a1_06():
    d = Diagram('Inside a transformer block', 'One block, repeated N times. Dashed lines are residual connections.',
                'One transformer block: token and position embeddings flow into multi-head self-attention, then add and normalize, then a feed-forward network, then add and normalize again, producing the block output. Residual connections skip around both sub-layers. A bracket shows the block repeats N times.',
                bar='teal')
    top = 128
    bw, bh, gp = 520, 66, 38
    names = [('blue', 'Token + position embeddings', None), ('teal', 'Multi-head self-attention', 'context from every other token'),
             ('sage', 'Add & normalize', 'residual + layer norm'), ('sand', 'Feed-forward network', 'same network, every position'),
             ('sage', 'Add & normalize', 'residual + layer norm'), ('blue', 'Block output', 'into the next block')]
    n = len(names)
    ch = n * bh + (n - 1) * gp + 100
    d.card(24, top, d.W - 48, ch, 'neutral')
    bx = d.W / 2 - bw / 2 - 40
    y0 = top + 50
    ys = [y0 + i * (bh + gp) for i in range(n)]
    for (k, t, s), y in zip(names, ys):
        d.block(bx, y, bw, bh, k, t, s)
    for i in range(n - 1):
        d.arrow([(bx + bw / 2, ys[i] + bh + 1), (bx + bw / 2, ys[i + 1] - 1)], head=6)
    # residual loops (right side)
    rx = bx + bw + 120
    d.arrow([(bx + bw + 1, ys[0] + bh / 2), (rx, ys[0] + bh / 2), (rx, ys[2] + bh / 2 - 12), (bx + bw + 1, ys[2] + bh / 2 - 12)], r=20)
    d.arrow([(bx + bw + 1, ys[2] + bh / 2 + 12), (rx + 100, ys[2] + bh / 2 + 12), (rx + 100, ys[4] + bh / 2), (bx + bw + 1, ys[4] + bh / 2)], r=20)
    d.pill(rx, (ys[0] + ys[2] + bh) / 2 - 6, 'residual', 14, 28)
    d.pill(rx + 100, (ys[2] + ys[4] + bh) / 2 + 6, 'residual', 14, 28)
    # xN bracket (left)
    lx = bx - 80
    d.o.append(f'<path d="M{lx+20} {ys[1]-8} Q{lx} {ys[1]-8} {lx} {ys[1]+12} V{ys[4]+bh-12} Q{lx} {ys[4]+bh+8} {lx+20} {ys[4]+bh+8}" fill="none" stroke="{LINE}" stroke-width="3" stroke-linecap="round"/>')
    d.pill(lx, (ys[1] + ys[4] + bh) / 2, '×N layers', 15, 32)
    CAPTIONS['a1/06-transformer-block'] = 'Dashed lines are residual connections: they carry the input forward unchanged, so a block only has to learn what to add, not the whole representation from scratch.'
    return d.save(A1, '06-transformer-block', top + ch + 24)


def a1_07():
    d = Diagram('How an LLM writes: one token at a time', 'Four steps, repeated until the model emits a stop token.',
                'Text generation as a loop: the tokens so far go through the decoder stack, which scores every token in the vocabulary; one token is sampled, appended to the input, and the whole process repeats.',
                bar='sand')

    def s1(d, cx, sy, k):
        tokens(d, cx, sy, k, ['The', 'cat', '?'], ['outline', 'outline', 'ghost'], w=70, h=54, g=10)

    def s2(d, cx, sy, k): slabs(d, cx, sy, k, ['Block N', 'Block 2', 'Block 1'], w=230, h=36, g=15)

    def s3(d, cx, sy, k):
        hs = [46, 74, 122, 58, 46]
        lb = ['a', 'the', 'sat', 'on', '…']
        w, g = 42, 8
        x0 = cx - (5 * w + 4 * g) / 2
        base = sy + 62
        a = ACC[k][0]
        for j, (hh, l) in enumerate(zip(hs, lb)):
            bx = x0 + j * (w + g)
            if j == 2:
                d.o.append(f'<rect x="{bx}" y="{base-hh}" width="{w}" height="{hh}" rx="10" fill="{a}"/>')
                d.t(bx + w / 2, base - hh / 2 + 5.4, l, 15, 800, INK)
            else:
                d.o.append(f'<rect x="{bx}" y="{base-hh}" width="{w}" height="{hh}" rx="10" fill="{BG}" stroke="{a}" stroke-opacity=".7" stroke-width="2.6"/>')
                d.t(bx + w / 2, base - hh / 2 + 5.4, l, 15, 800, a)

    def s4(d, cx, sy, k):
        x0, y0, w, g = tokens(d, cx, sy, k, ['The', 'cat', 'sat'], ['dim', 'dim', 'solid'], w=70, h=54, g=10, dy=14)
        ax = x0 + 2 * (w + g) + w / 2
        d.arrow([(ax, y0 - 44), (ax, y0 - 1)], head=7)

    steps = [(1, 'INPUT', 'Tokens so far', 'the text generated so far', '“Here is everything so far.”', 'blue'),
             (2, 'THINK', 'Decoder stack', '×N attention + MLP blocks', '“Which earlier tokens matter?”', 'sage'),
             (3, 'SCORE', 'Vocab scores', 'one probability per token', '“How likely is each word next?”', 'teal'),
             (4, 'PICK', 'Sample a token', 'pick one, favouring high scores', '“Pick one, add it to the text.”', 'sand')]
    xs, top, bot = d.step_cards(steps, 275, 44, [s1, s2, s3, s4])
    x1, x4 = xs[0] + 137.5, xs[3] + 137.5
    yb = bot + 64
    d.arrow([(x4, bot + 1), (x4, yb), (x1, yb), (x1, bot + 3)], r=26)
    d.pill((x1 + x4) / 2, yb, 'append the token, repeat', 20, 44)
    CAPTIONS['a1/07-generation-loop'] = 'Every loop produces exactly one new token, using the whole sequence generated so far. Repeat until an end-of-sequence token appears or a length limit is hit.'
    return d.save(A1, '07-generation-loop', bot + 110)


# ------------------------------------------------------------------ Article 2
def a2_01():
    d = Diagram('Four decisions behind a good prompt', 'Make each one on purpose, and the answer improves.',
                'Left to right: pick the source (memory, web, or research), give context (only what is relevant), get honest critique (rubric and a second model), and match the tool (search, code, or files).',
                bar='sand')

    def s1(d, cx, sy, k): slabs(d, cx, sy, k, ['Memory', 'Web', 'Research'], w=190, h=40, g=14)

    def s2(d, cx, sy, k):
        w, h, g = 190, 40, 14
        ys = [sy - (3 * h + 2 * g) / 2 + i * (h + g) for i in range(3)]
        d.chip(cx - w / 2, ys[0], w, h, k, 'Your question', 'solid')
        d.chip(cx - w / 2, ys[1], w, h, k, 'Relevant context', 'outline')
        d.chip(cx - w / 2, ys[2], w, h, k, 'Everything else', 'dim')

    def s3(d, cx, sy, k): slabs(d, cx, sy, k, ['Rubric', 'Second model'], w=190, h=44, g=16)

    def s4(d, cx, sy, k): slabs(d, cx, sy, k, ['Search', 'Code', 'Files'], w=190, h=40, g=14)

    steps = [(1, 'SOURCE', 'Pick the source', 'memory, web, or research', '“Where should this come from?”', 'sand'),
             (2, 'CONTEXT', 'Give context', 'only what is relevant', '“What does it need to know?”', 'blue'),
             (3, 'CRITIQUE', 'Get honest critique', 'rubric, second model', '“What would a skeptic say?”', 'teal'),
             (4, 'TOOL', 'Match the tool', 'search, code, files', '“Which tool fits the task?”', 'sage')]
    xs, top, bot = d.step_cards(steps, 275, 44, [s1, s2, s3, s4])
    return d.save(A2, '01-prompting-overview', bot + 24)


def a2_02():
    d = Diagram('Choosing an information source', 'Two questions decide where the answer should come from.',
                'Flowchart: start with your question. If it needs recent, local, or niche information, go on; otherwise use pretrained knowledge. If it is also complex and multi-part, use deep research; otherwise use web search.',
                bar='sand')
    top = 150
    bh = 104
    ymain = top
    qx, qw = 24, 170
    x1, w1 = 278, 280
    x2, w2 = 642, 280
    x3, w3 = 1006, 250
    d.block(qx, ymain + 20, qw, 64, 'neutral', 'Your question', None, r=32)
    d.block(x1, ymain, w1, bh, 'sand', 'Is it recent, local, or niche?', 'after the training cutoff, near me, rarely discussed')
    d.block(x2, ymain, w2, bh, 'sand', 'Is it complex and multi-part?', 'needs many sources compared and combined')
    cy = ymain + bh / 2
    d.arrow([(qx + qw + 1, cy), (x1 - 1, cy)])
    d.arrow([(x1 + w1 + 1, cy), (x2 - 1, cy)])
    d.arrow([(x2 + w2 + 1, cy), (x3 - 1, cy)])
    d.pill((x1 + w1 + x2) / 2, cy - 30, 'Yes', 14, 28)
    d.pill((x2 + w2 + x3) / 2, cy - 30, 'Yes', 14, 28)
    ylow = ymain + bh + 110
    for (x, w, key, t, s) in ((x1, w1, 'blue', 'Pretrained knowledge', 'facts, definitions, summaries'),):
        d.card(x, ylow, w, bh, key)
        d.t(x + w / 2, ylow + 44, t, 22, 800, TEXT)
        d.t(x + w / 2, ylow + 74, s, 16, 400, MUTED)
        d.arrow([(x + w / 2, ymain + bh + 1), (x + w / 2, ylow - 1)])
        d.pill(x + w / 2 + 44, ymain + bh + 55, 'No', 14, 28)
    d.card(x2, ylow, w2, bh, 'sage')
    d.t(x2 + w2 / 2, ylow + 44, 'Web search', 22, 800, TEXT)
    d.t(x2 + w2 / 2, ylow + 74, 'seconds, a few sources', 16, 400, MUTED)
    d.arrow([(x2 + w2 / 2, ymain + bh + 1), (x2 + w2 / 2, ylow - 1)])
    d.pill(x2 + w2 / 2 + 44, ymain + bh + 55, 'No', 14, 28)
    d.card(x3, ymain, w3, bh, 'teal')
    d.t(x3 + w3 / 2, ymain + 44, 'Deep research', 22, 800, TEXT)
    d.t(x3 + w3 / 2, ymain + 74, 'minutes, many sources, cited', 16, 400, MUTED)
    CAPTIONS['a2/02-choosing-an-information-source'] = 'Not sure whether the model will search? Ask for it explicitly.'
    return d.save(A2, '02-choosing-an-information-source', ylow + bh + 24)


def a2_03():
    d = Diagram('Choosing the right tool for the task', 'Ask the questions in order and stop at the first yes.',
                'Flowchart: if the task needs fresh external information, search the web. Otherwise, if it needs exact math or a chart, run code. Otherwise, if it needs a file read or written, use file tools. Otherwise, answer directly.',
                bar='sage')
    top = 130
    bh, gp = 96, 46
    xl, wl = 24, 500
    xr, wr = 756, 500
    rows = [('Needs fresh outside info?', 'recent, local, or niche', 'Web search', 'e.g. a highly rated gym near me', 'sage'),
            ('Needs exact math or a chart?', 'numbers, trends, calculations', 'Run code', 'e.g. what are my sales trends?', 'teal'),
            ('Needs a file read or written?', 'your documents, a new report', 'Read or write a file', 'e.g. summarize the attached report', 'blue')]
    ys = [top + i * (bh + gp) for i in range(4)]
    for i, (q, qs, o, os_, k) in enumerate(rows):
        d.block(xl, ys[i], wl, bh, 'sand', q, qs)
        d.card(xr, ys[i], wr, bh, k)
        d.t(xr + wr / 2, ys[i] + 42, o, 22, 800, TEXT)
        d.t(xr + wr / 2, ys[i] + 72, os_, 16, 400, MUTED)
        d.arrow([(xl + wl + 1, ys[i] + bh / 2), (xr - 1, ys[i] + bh / 2)])
        d.pill((xl + wl + xr) / 2, ys[i] + bh / 2 - 30, 'Yes', 14, 28)
        d.arrow([(xl + wl / 2, ys[i] + bh + 1), (xl + wl / 2, ys[i + 1] - 1)])
        d.pill(xl + wl / 2 + 44, ys[i] + bh + gp / 2, 'No', 14, 28)
    d.card(xl, ys[3], wl, bh, 'neutral')
    d.t(xl + wl / 2, ys[3] + 42, 'Answer directly', 22, 800, TEXT)
    d.t(xl + wl / 2, ys[3] + 72, 'e.g. why do cats stare at walls?', 16, 400, MUTED)
    CAPTIONS['a2/03-choosing-a-tool'] = 'Complex, multi-source questions can combine these steps, which is what deep research automates.'
    return d.save(A2, '03-choosing-a-tool', ys[3] + bh + 24)


ALL = {'a1_01': a1_01, 'a1_02': a1_02, 'a1_03': a1_03, 'a1_04': a1_04, 'a1_05': a1_05, 'a1_06': a1_06, 'a1_07': a1_07,
       'a2_01': a2_01, 'a2_02': a2_02, 'a2_03': a2_03}
def write_captions():
    for key, folder in (('a1', A1), ('a2', A2)):
        rows = sorted((k, v) for k, v in CAPTIONS.items() if k.startswith(key + '/'))
        if rows:
            with open(os.path.join(folder, 'captions.md'), 'w', encoding='utf-8') as f:
                f.write('# Image captions (paste under each image on Medium)\n\n')
                for k, v in rows:
                    f.write(f'- `{k.split("/")[1]}.svg`: {v}\n')


if __name__ == '__main__':
    for k in (sys.argv[1:] or ALL):
        print(k, ALL[k]())
    write_captions()
