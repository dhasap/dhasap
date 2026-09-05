#!/usr/bin/env python3
"""generate_snake.py — v6: organic snake pathfinding.

The snake does NOT follow a straight row-major line anymore. It behaves like a
classic Snake game: it hunts the nearest "food" (a contribution cell), taking
irregular turns — so the route looks natural / random, not like a train track.

Algorithm
---------
1. Build the 7 x N contribution grid from the public GitHub calendar page.
2. Simulate a snake: at every step pick the adjacent cell that is on a shortest
   path (BFS) toward the nearest unvisited food cell; ties broken by a
   deterministic but scattered heuristic so turns are irregular.
3. When no food remains, walk a short wandering tail so the animation ends
   gracefully, then loop.

Rendering (SMIL, self-contained SVG)
------------------------------------
- grid: rounded squares, 5 levels (#101737 empty -> #54e08a max)
- body: tail segments follow the head along the same motion path (mpath ref)
- head: chibi anime snake head, rotate=auto, chomping mouth + tongue flick
- eat effect: 4-point sparkle burst on each food cell as the head arrives

Usage:
  python3 generate_snake.py --user dhasap --out snake.svg
  python3 generate_snake.py --dummy --out test.svg   # offline test
"""
import argparse
import math
import random
import re
import sys
import urllib.request
from collections import deque
from datetime import date, timedelta

# ---------- palette ----------
BG = "#060a1f"
STAR = "#fdf6e9"
CORAL = "#e2524a"
HAIR = "#5468d4"
HAIR_HI = "#8aa0f5"
SKIN = "#ffe0d1"
BLUSH = "#ff8fab"
EYE = "#1d2d50"
BODY_DARK = "#3b4a8f"
BODY_MID = "#5b74d9"
BODY_LIGHT = "#8aa0f5"
BELLY = "#c7d4ff"
LEVELS = ["#101737", "#0e5a3a", "#1a8a4f", "#2fbf67", "#54e08a"]

UAS = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")


# ---------- data ----------
def fetch_contributions(user: str) -> dict:
    """Parse github.com/users/<user>/contributions (public, no token)."""
    url = f"https://github.com/users/{user}/contributions"
    req = urllib.request.Request(url, headers={"User-Agent": UAS})
    with urllib.request.urlopen(req, timeout=30) as r:
        page = r.read().decode("utf-8", errors="replace")
    cells = re.findall(r'data-date="(\d{4}-\d{2}-\d{2})"[^>]*data-level="(\d+)"', page)
    if not cells:
        raise RuntimeError("no contribution cells found")
    return {d: int(l) for d, l in cells}


def dummy_contributions(weeks=52) -> dict:
    out, today = {}, date.today()
    rnd = random.Random(11)
    for i in range(weeks * 7):
        d = today - timedelta(days=(weeks * 7 - 1) - i)
        out[d.isoformat()] = rnd.choice([0, 0, 1, 1, 2, 3, 4, 2, 0, 1])
    return out


class Grid:
    def __init__(self, contribs: dict):
        days = sorted(contribs)
        first = date.fromisoformat(days[0])
        start = first - timedelta(days=first.weekday())
        last = date.fromisoformat(days[-1])
        self.cols, d = [], start
        while d <= last:
            self.cols.append([
                contribs.get((d + timedelta(days=i)).isoformat(), 0) for i in range(7)
            ])
            d += timedelta(days=7)
        self.nrows, self.ncols = 7, len(self.cols)
        mx = max(max(c) for c in self.cols) or 1
        self.level = [[0] * self.nrows for _ in range(self.ncols)]
        for ci, col in enumerate(self.cols):
            for ri, c in enumerate(col):
                if c > 0:
                    self.level[ci][ri] = 1 + min(3, int(c / mx * 3.999))

    def inb(self, c, r):
        return 0 <= c < self.ncols and 0 <= r < self.nrows

    def food(self):
        return {(c, r) for c in range(self.ncols) for r in range(self.nrows)
                if self.level[c][r] > 0}


# ---------- organic path (snake hunts nearest food) ----------
def bfs_next_step(grid: Grid, src, dst, blocked):
    """First move (from src) along a shortest path to dst, avoiding blocked."""
    if src == dst:
        return None
    q = deque([src])
    prev = {src: None}
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        c, r = cur
        nbrs = [(c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)]
        # scattered tie-breaking so turns are irregular, not a straight line
        random.shuffle(nbrs)
        for n in nbrs:
            if n not in prev and grid.inb(*n) and n not in blocked:
                prev[n] = cur
                q.append(n)
    if dst not in prev:
        return None
    cur = dst
    while prev[cur] is not None and prev[cur] != src:
        cur = prev[cur]
    return cur


def build_organic_path(grid: Grid, rng: random.Random):
    """Simulate a snake that eats every food cell in a scattered order."""
    food = grid.food()
    if not food:
        # fallback: wander
        return [(c, r) for c in range(grid.ncols) for r in range(grid.nrows)]

    start = min(food)  # oldest contribution (leftmost, then topmost)
    path = [start]
    current = start
    remaining = set(food) - {start}
    visited = {start}
    # short body so the snake can manoeuvre (visual only; path stays simple)
    trail = deque([start], maxlen=12)

    while remaining:
        # nearest food by manhattan distance
        dst = min(remaining, key=lambda p: abs(p[0] - current[0]) + abs(p[1] - current[1]))
        blocked = set(trail) - {current}
        nxt = bfs_next_step(grid, current, dst, blocked)
        if nxt is None:
            # unreachable (boxed in) — drop that target and try another
            remaining.discard(dst)
            continue
        path.append(nxt)
        trail.append(nxt)
        visited.add(nxt)
        current = nxt
        remaining.discard(nxt)

    return path


# ---------- smoothing ----------
def smooth(pts, samples=3):
    """Catmull-Rom through points for a fluid slither."""
    if len(pts) < 3:
        return pts
    out, p = [], [pts[0]] + pts + [pts[-1]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for k in range(samples):
            t = k / samples
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(pts[-1])
    return out


# ---------- render ----------
def render(grid: Grid, path, cell=13, gap=3, speed=22, out="snake.svg"):
    cols, nrows = grid.ncols, grid.nrows
    W = cols * (cell + gap) + 56
    H = nrows * (cell + gap) + 96
    ox, oy = 28, 68

    cells_svg = "\n  ".join(
        f'<rect x="{ox + ci * (cell + gap)}" y="{oy + ri * (cell + gap)}" width="{cell}" '
        f'height="{cell}" rx="3.5" fill="{LEVELS[grid.level[ci][ri]]}"/>'
        for ci in range(cols) for ri in range(nrows)
    )

    centers = [(ox + c * (cell + gap) + cell / 2,
                oy + r * (cell + gap) + cell / 2) for c, r in path]
    rng = random.Random(7)
    sm = smooth(centers, samples=3)
    simp = [sm[0]]
    for p in sm[1:]:
        if math.dist(p, simp[-1]) > 2.0:
            simp.append(p)
    sm = simp
    dur = max(8.0, len(sm) / (speed * 3))
    mpath = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in sm)

    # ---- eat sparkles: timed at each food cell's visit ----
    visit_idx = {}
    seen = set()
    for i, pos in enumerate(path):
        if pos not in seen:
            visit_idx[pos] = i
            seen.add(pos)
    sparkles = []
    for (ci, ri) in sorted(grid.food()):
        i = visit_idx.get((ci, ri))
        if i is None:
            continue
        frac = i / max(1, len(path) - 1)
        x = ox + ci * (cell + gap) + cell / 2
        y = oy + ri * (cell + gap) + cell / 2
        begin = f"{frac * dur:.2f}s"
        sparkles.append(
            f'<g transform="translate({x:.1f},{y:.1f})" opacity="0">'
            f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.05;0.18;0.3" '
            f'dur="{dur:.2f}s" begin="{begin}" repeatCount="indefinite"/>'
            f'<path d="M0 -6 L1.6 -1.6 L6 0 L1.6 1.6 L0 6 L-1.6 1.6 L-6 0 L-1.6 -1.6 Z" fill="{STAR}"/>'
            f'<path d="M0 -3.4 L1 -1 L3.4 0 L1 1 L0 3.4 L-1 1 L-3.4 0 L-1 -1 Z" fill="{CORAL}" opacity="0.9"/>'
            f"</g>"
        )
    if len(sparkles) > 240:
        sparkles = sparkles[::2]
    sparkles_svg = "\n  ".join(sparkles)

    rnd = random.Random(42)
    stars_svg = "\n  ".join(
        f'<circle cx="{rnd.uniform(8, W - 8):.1f}" cy="{rnd.uniform(8, H - 8):.1f}" '
        f'r="{rnd.choice([0.9, 1.2, 1.6])}" fill="{STAR}" opacity="{rnd.choice([0.3, 0.5, 0.75])}"/>'
        for _ in range(44)
    )

    # ---- body: tail segments trailing the head (12% of the path) ----
    body_len = 42
    segs, belly = [], []
    for s in range(body_len):
        f = 1 - (s + 1) * 0.12 / body_len
        t = s / body_len
        r = 7.4 * (1 - t) ** 0.7 + 1.6
        segs.append(
            f'<circle r="{r:.2f}" fill="{BODY_MID}">'
            f'<animateMotion dur="{dur:.2f}s" repeatCount="indefinite" begin="-{f * dur:.2f}s" '
            f'keyPoints="1;0" keyTimes="0;1" calcMode="linear"><mpath href="#snakepath"/></animateMotion>'
            f"</circle>"
        )
    for s in range(0, body_len, 2):
        f = 1 - (s + 1) * 0.12 / body_len
        t = s / body_len
        r = 2.9 * (1 - t) ** 0.8 + 0.35
        belly.append(
            f'<circle r="{r:.2f}" fill="{BELLY}" opacity="0.5">'
            f'<animateMotion dur="{dur:.2f}s" repeatCount="indefinite" begin="-{f * dur:.2f}s" '
            f'keyPoints="1;0" keyTimes="0;1" calcMode="linear"><mpath href="#snakepath"/></animateMotion>'
            f"</circle>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="anime contribution snake">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{BG}"/>
    <stop offset="1" stop-color="#0b1030"/>
  </linearGradient>
  <path id="snakepath" d="{mpath}" fill="none"/>
</defs>
<rect width="{W}" height="{H}" rx="14" fill="url(#bg)"/>
<g>
  {stars_svg}
</g>
<text x="{W / 2}" y="30" text-anchor="middle" font-family="Segoe UI,sans-serif" font-size="14" font-weight="700" fill="{HAIR_HI}">contribution snake — anime edition (◕‿◕)</text>
<g>
  {cells_svg}
</g>
<g>
  {sparkles_svg}
</g>
<g>
  {''.join(segs)}
  {''.join(belly)}
  <g>
    <animateMotion dur="{dur:.2f}s" repeatCount="indefinite" calcMode="linear" keyPoints="1;0" keyTimes="0;1" rotate="auto"><mpath href="#snakepath"/></animateMotion>
    <g transform="scale(1.7)">
      <path d="M 2 0 Q 2 -8 -6 -8.6 Q -16 -8.6 -16 0 Q -16 8.6 -6 8.6 Q 2 8 2 0 Z" fill="{HAIR}"/>
      <path d="M -14 -6 Q -10 -13 -4 -12 Q -8 -9 -6 -6.5 Z" fill="{HAIR_HI}"/>
      <path d="M -6 -8.4 Q 0 -14 5 -11 Q 0 -9.5 -2 -7 Z" fill="{HAIR}"/>
      <ellipse cx="-7" cy="-1.6" rx="2.5" ry="3.1" fill="{EYE}"/>
      <ellipse cx="-0.5" cy="-1.6" rx="2.5" ry="3.1" fill="{EYE}"/>
      <circle cx="-6.3" cy="-2.6" r="0.85" fill="#fff"/>
      <circle cx="0.2" cy="-2.6" r="0.85" fill="#fff"/>
      <ellipse cx="-11.5" cy="3" rx="2.6" ry="1.5" fill="{BLUSH}" opacity="0.85"/>
      <ellipse cx="3" cy="3" rx="2.6" ry="1.5" fill="{BLUSH}" opacity="0.85"/>
      <g>
        <animateTransform attributeName="transform" type="scale" values="1 0.55;1 1;1 0.55" dur="0.9s" repeatCount="indefinite" additive="sum"/>
        <ellipse cx="-1.5" cy="4.6" rx="3.6" ry="2.4" fill="{CORAL}"/>
        <path d="M -4.6 4.2 Q -1.5 8 1.6 4.2 Z" fill="{CORAL}"/>
      </g>
      <path d="M -1 7 Q 1 10 4 9" stroke="{CORAL}" stroke-width="1.2" fill="none" stroke-linecap="round" opacity="0">
        <animate attributeName="opacity" values="0;1;0" keyTimes="0;0.5;1" dur="2.4s" repeatCount="indefinite"/>
      </path>
    </g>
  </g>
</g>
</svg>
"""
    with open(out, "w") as f:
        f.write(svg)
    import os
    print(f"written {out}: {os.path.getsize(out)} bytes, {len(sm)} pts, "
          f"{len(path)} eats, {dur:.1f}s loop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="dhasap")
    ap.add_argument("--out", default="snake.svg")
    ap.add_argument("--cell", type=int, default=13)
    ap.add_argument("--speed", type=float, default=22)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()

    try:
        contribs = dummy_contributions() if a.dummy else fetch_contributions(a.user)
    except Exception as e:
        print(f"fetch failed ({e}); dummy fallback", file=sys.stderr)
        contribs = dummy_contributions()

    random.seed(a.seed)
    grid = Grid(contribs)
    rng = random.Random(a.seed)
    path = build_organic_path(grid, rng)
    render(grid, path, cell=a.cell, speed=a.speed, out=a.out)


if __name__ == "__main__":
    main()
