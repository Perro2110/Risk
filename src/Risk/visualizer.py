"""
Risk/visualizer.py

Visual display of a Risk GameState using matplotlib, styled to evoke the
classic physical board game — parchment tones, muted paints, serif typography,
and ink-like print rendering throughout.

Usage:
    from Risk.visualizer import RiskVisualizer
    viz = RiskVisualizer(game_state)
    viz.show()              # blocking display
    viz.save("board.png")   # save to file
    viz.update(game_state)  # redraw with updated state
    plt.pause(0.1)          # non-blocking refresh inside a simulation loop
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import (
    FancyBboxPatch, Circle, FancyArrowPatch, Polygon
)
from matplotlib.colors import to_rgba
from scipy.spatial import ConvexHull          # ships with scipy (matplotlib dep)
from typing import TYPE_CHECKING
from shapely.geometry import Polygon as ShapelyPolygon

if TYPE_CHECKING:
    from Risk.game_state import GameState


# =============================================================================
#  PALETTE — parchment / antique map
# =============================================================================

# Board surfaces
PARCHMENT      = "#F5EDD6"   # main background / panel surface
PARCHMENT_DARK = "#E8D9B8"   # slightly deeper parchment for inset areas
OCEAN          = "#8FABBF"   # desaturated blue-green sea
OCEAN_DEEP     = "#6E95A8"   # slightly deeper water for vignette
INK            = "#2B1D0E"   # near-black ink (text, outlines)
INK_FAINT      = "#6B5A47"   # faint ink for minor labels / rules
INK_MID        = "#4A3728"   # medium ink for sub-headings

# Continent land tints (muted earth tones, semi-transparent overlays)
CONTINENT_TINTS: dict[str, str] = {
    "North America": "#C4B078",   # dusty gold
    "South America": "#C47878",   # sage green
    "Europe":        "#A0B8C8",   # slate blue
    "Africa":        "#C49A6C",   # raw sienna
    "Asia":          "#A8C48A",   # terra cotta pink
    "Australia":     "#7E86BF",   # teal-mint
}

# Player colours — muted, board-paint inspired
PLAYER_COLOURS = [
    "#B03030",   # brick red
    "#2A5F8F",   # navy blue
    "#3A7A3A",   # forest green
    "#B07830",   # ochre / burnt orange
    "#6A3A8A",   # deep violet
    "#8A7A30",   # olive gold
    "#30787A",   # dark teal
    "#A04060",   # dusty rose
]
NEUTRAL_COLOUR = "#9A8A78"   # warm grey-brown

# Accent / chrome
GOLD_BORDER    = "#A8883A"   # antique gold for active highlights
SEPARATOR_COL  = "#C8B890"   # rule lines in panel

# Typography
# We request serif fonts in a priority chain; matplotlib falls back gracefully.
SERIF_FAMILY   = ["Palatino Linotype", "Palatino", "Book Antiqua",
                  "Times New Roman", "DejaVu Serif", "serif"]
TITLE_FAMILY   = ["Palatino Linotype", "Palatino", "Book Antiqua",
                  "DejaVu Serif", "serif"]


# =============================================================================
#  TERRITORY POSITIONS  (x: 0–100 west→east,  y: 0–100 south→north)
# =============================================================================
TERRITORY_POS: dict[str, tuple[float, float]] = {
    "Alaska":                (7.4,  83.9),
    "Northwest Territory":   (17.5, 84.6),
    "Greenland":             (34.9, 88.5),
    "Alberta":               (15.8, 76.5),
    "Ontario":               (22.2, 74.8),
    "Quebec":                (28.9, 74.5),
    "Western United States": (16.1, 65.8),
    "Eastern United States": (23.9, 63.5),
    "Central America":       (17.6, 53.0),
    "Venezuela":             (24.0, 45.5),
    "Peru":                  (23.8, 33.5),
    "Brazil":                (32.0, 37.5),
    "Argentina":             (25.0, 23.0),
    "Iceland":               (42.9, 79.6),
    "Great Britain":         (40.5, 69.0),
    "Northern Europe":       (50.3, 66.6),
    "Scandinavia":           (50.3, 80.4),
    "Ukraine":               (58.7, 71.6),
    "Western Europe":        (42.5, 55.0),
    "Southern Europe":       (51.0, 58.0),
    "North Africa":          (46.7, 39.9),
    "Egypt":                 (54.3, 44.0),
    "East Africa":           (59.3, 33.8),
    "Congo":                 (53.8, 28.1),
    "South Africa":          (55.0, 15.0),
    "Madagascar":            (63.3, 14.9),
    "Ural":                  (68.6, 76.5),
    "Siberia":               (73.9, 82.2),
    "Yakutsk":               (81.2, 85.7),
    "Kamchatka":             (89.0, 84.5),
    "Irkutsk":               (80.0, 75.0),
    "Mongolia":              (81.2, 65.7),
    "Japan":                 (91.5, 65.8),
    "Afghanistan":           (67.3, 63.8),
    "China":                 (79.5, 55.9),
    "Middle East":           (61.4, 49.4),
    "India":                 (72.7, 48.0),
    "Siam":                  (80.7, 44.5),
    "Indonesia":             (81.2, 29.1),
    "New Guinea":            (89.4, 33.2),
    "Western Australia":     (86.3, 16.0),
    "Eastern Australia":     (95.0, 19.2),
}

# Classic Risk adjacency list
ADJACENCY: list[tuple[str, str]] = [
    ("Alaska", "Northwest Territory"), ("Alaska", "Alberta"),
    ("Northwest Territory", "Alberta"), ("Northwest Territory", "Ontario"),
    ("Northwest Territory", "Greenland"), ("Greenland", "Ontario"),
    ("Greenland", "Quebec"), ("Alberta", "Ontario"),
    ("Alberta", "Western United States"), ("Ontario", "Quebec"),
    ("Ontario", "Western United States"), ("Ontario", "Eastern United States"),
    ("Quebec", "Eastern United States"),
    ("Western United States", "Eastern United States"),
    ("Western United States", "Central America"),
    ("Eastern United States", "Central America"),
    ("Central America", "Venezuela"),
    ("Venezuela", "Peru"), ("Venezuela", "Brazil"),
    ("Peru", "Brazil"), ("Peru", "Argentina"), ("Brazil", "Argentina"),
    ("Brazil", "North Africa"),
    ("Iceland", "Great Britain"), ("Iceland", "Scandinavia"),
    ("Great Britain", "Northern Europe"), ("Great Britain", "Scandinavia"),
    ("Great Britain", "Western Europe"),
    ("Northern Europe", "Scandinavia"), ("Northern Europe", "Ukraine"),
    ("Northern Europe", "Western Europe"), ("Northern Europe", "Southern Europe"),
    ("Scandinavia", "Ukraine"), ("Ukraine", "Southern Europe"),
    ("Ukraine", "Middle East"), ("Ukraine", "Afghanistan"),
    ("Ukraine", "Ural"), ("Western Europe", "Southern Europe"),
    ("Western Europe", "North Africa"), ("Southern Europe", "North Africa"),
    ("Southern Europe", "Egypt"), ("Southern Europe", "Middle East"),
    ("North Africa", "Egypt"), ("North Africa", "East Africa"),
    ("North Africa", "Congo"), ("Egypt", "East Africa"),
    ("Egypt", "Middle East"), ("East Africa", "Congo"),
    ("East Africa", "South Africa"), ("East Africa", "Madagascar"),
    ("East Africa", "Middle East"), ("Congo", "South Africa"),
    ("South Africa", "Madagascar"),
    ("Ural", "Afghanistan"), ("Ural", "Siberia"), ("Ural", "China"),
    ("Siberia", "Yakutsk"), ("Siberia", "Irkutsk"),
    ("Siberia", "Mongolia"), ("Siberia", "China"),
    ("Yakutsk", "Kamchatka"), ("Yakutsk", "Irkutsk"),
    ("Kamchatka", "Irkutsk"), ("Kamchatka", "Mongolia"),
    ("Kamchatka", "Japan"), ("Irkutsk", "Mongolia"),
    ("Mongolia", "Japan"), ("Mongolia", "China"),
    ("Afghanistan", "China"), ("Afghanistan", "India"),
    ("Afghanistan", "Middle East"), ("China", "India"),
    ("China", "Siam"), ("India", "Middle East"), ("India", "Siam"),
    ("Siam", "Indonesia"),
    ("Indonesia", "New Guinea"), ("Indonesia", "Western Australia"),
    ("New Guinea", "Eastern Australia"),
    ("Western Australia", "Eastern Australia"),
    ("Alaska", "Kamchatka"),
    ("Greenland", "Iceland"),
]

# Sea / wrap connections that get a dashed line treatment
SEA_CONNECTIONS: set[frozenset[str]] = {
    frozenset({"Alaska", "Kamchatka"}),
    frozenset({"Greenland", "Iceland"}),
    frozenset({"Brazil", "North Africa"}),
}

# Continent membership
CONTINENT_MEMBERS: dict[str, list[str]] = {
    "North America": [
        "Alaska", "Northwest Territory", "Greenland", "Alberta", "Ontario",
        "Quebec", "Western United States", "Eastern United States",
        "Central America",
    ],
    "South America": ["Venezuela", "Peru", "Brazil", "Argentina"],
    "Europe": [
        "Iceland", "Great Britain", "Northern Europe", "Scandinavia",
        "Ukraine", "Western Europe", "Southern Europe",
    ],
    "Africa": [
        "North Africa", "Egypt", "East Africa", "Congo",
        "South Africa", "Madagascar",
    ],
    "Asia": [
        "Ural", "Siberia", "Yakutsk", "Kamchatka", "Irkutsk", "Mongolia",
        "Japan", "Afghanistan", "China", "Middle East", "India", "Siam",
    ],
    "Australia": [
        "Indonesia", "New Guinea", "Western Australia", "Eastern Australia",
    ],
}

SHORT_LABEL: dict[str, str] = {
    "Northwest Territory":   "NW Terr.",
    "Western United States": "W. USA",
    "Eastern United States": "E. USA",
    "Central America":       "C. Amer.",
    "North Africa":          "N. Africa",
    "East Africa":           "E. Africa",
    "South Africa":          "S. Africa",
    "Western Europe":        "W. Eur.",
    "Northern Europe":       "N. Eur.",
    "Southern Europe":       "S. Eur.",
    "Western Australia":     "W. Aus.",
    "Eastern Australia":     "E. Aus.",
    "Middle East":           "Mid. East",
    "Afghanistan":           "Afghan.",
}


# =============================================================================
#  HELPERS
# =============================================================================

def _short(name: str) -> str:
    return SHORT_LABEL.get(name, name)


def _serif(size: float, bold: bool = False, italic: bool = False) -> dict:
    """Return a dict of text kwargs using the serif font stack."""
    return dict(
        fontfamily=SERIF_FAMILY,
        fontsize=size,
        fontweight="bold" if bold else "normal",
        fontstyle="italic" if italic else "normal",
    )


def _convex_hull_polygon(points: list[tuple[float, float]],
                         pad: float = 2.5) -> np.ndarray | None:
    """Return padded convex-hull vertices for a set of (x,y) points."""
    if len(points) < 3:
        return None
    pts = np.array(points)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    verts = pts[hull.vertices]
    # Pad outward from centroid
    cx, cy = verts.mean(axis=0)
    normed = verts - np.array([cx, cy])
    norms = np.linalg.norm(normed, axis=1, keepdims=True)
    norms[norms == 0] = 1
    padded = verts + (normed / norms) * pad
    return padded


def _parchment_texture_overlay(ax: plt.Axes) -> None:
    """
    Draw a faint noise grain over the map to simulate a printed board texture.
    Uses a small random RGBA image stretched over the axes.
    """
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1, (80, 80))
    rgba = np.zeros((80, 80, 4), dtype=float)
    rgba[..., 0] = 0.72   # warm cream R
    rgba[..., 1] = 0.58   # G
    rgba[..., 2] = 0.38   # B
    rgba[..., 3] = noise * 0.065   # very faint alpha
    ax.imshow(rgba, extent=(0, 100, 0, 100),
              aspect="auto", zorder=10, interpolation="bicubic")


def _vignette_overlay(ax: plt.Axes) -> None:
    """Subtle radial vignette around map edges using a gradient fill polygon."""
    # We approximate the vignette with 4 edge gradient strips
    for (x0, y0, x1, y1) in [
        (0, 0, 8, 100), (92, 0, 100, 100),   # left, right
        (0, 0, 100, 8), (0, 92, 100, 100),   # bottom, top
    ]:
        alpha = 0.18
        ax.fill(
            [x0, x1, x1, x0],
            [y0, y0, y1, y1],
            color=OCEAN_DEEP,
            alpha=alpha, zorder=9, linewidth=0,
        )


def _draw_compass(ax: plt.Axes,
                  cx: float = 5.5, cy: float = 8.0,
                  r: float = 3.5) -> None:
    """Draw a simple decorative compass rose."""
    dirs = [
        (0,  r,     "N", 8.5),
        (0,  -r,    "S", 7.0),
        (r,  0,     "E", 7.0),
        (-r, 0,     "W", 7.0),
    ]
    for dx, dy, lbl, fs in dirs:
        ax.annotate(
            "", xy=(cx + dx * 0.65, cy + dy * 0.65),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=INK_FAINT,
                lw=0.8,
                mutation_scale=7,
            ),
            zorder=12,
        )
        ax.text(cx + dx, cy + dy, lbl,
                color=INK_FAINT, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=12) # """fontfamily=SERIF_FAMILY,"""
    # inner circle
    ax.add_patch(Circle((cx, cy), r * 0.18,
                         color=PARCHMENT_DARK, ec=INK_FAINT, lw=0.7,
                         zorder=13))


# =============================================================================
#  MAIN CLASS
# =============================================================================

class RiskVisualizer:
    """
    Renders a Risk GameState as a geographically-positioned classic board map,
    styled to evoke the physical board game's parchment-and-paint aesthetic.

    Parameters
    ----------
    game_state : GameState
        Initial game state to visualise.
    figsize : tuple[int, int]
        Matplotlib figure dimensions in inches.
    """

    def __init__(
        self,
        game_state: GameState,
        figsize: tuple[int, int] = (22, 13),
    ) -> None:
        # matplotlib.rcParams["font.family"] = SERIF_FAMILY
        matplotlib.rcParams["axes.facecolor"] = PARCHMENT

        self.fig = plt.figure(figsize=figsize, facecolor=PARCHMENT)

        # Layout: map takes most of the canvas; thin panel on the right;
        # narrow turn-tracker strip along the bottom.
        self.ax_map   = self.fig.add_axes((0.00, 0.08, 0.77, 0.92))
        self.ax_panel = self.fig.add_axes((0.78, 0.08, 0.21, 0.92))
        self.ax_bar   = self.fig.add_axes((0.00, 0.00, 1.00, 0.075))

        self._style_axes()
        self.draw(game_state)

    # -------------------------------------------------------------------------
    #  Public API
    # -------------------------------------------------------------------------

    def show(self, blocking: bool = True) -> None:
        """Display the board (blocking by default)."""
        plt.show(block=blocking)

    def save(self, path: str, dpi: int = 180) -> None:
        """Save board image to *path*."""
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight",
                         facecolor=PARCHMENT)

    def update(self, game_state: GameState) -> None:
        """Redraw all axes with a new game state (use inside an animation loop)."""
        for ax in (self.ax_map, self.ax_panel, self.ax_bar):
            ax.cla()
        self._style_axes()
        self.draw(game_state)
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------------------------
    #  Orchestration
    # -------------------------------------------------------------------------

    def draw(self, gs: GameState) -> None:
        self._draw_map(gs)
        self._draw_panel(gs)
        self._draw_status_bar(gs)

    # -------------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------------

    def _style_axes(self) -> None:
        """Apply base styling to all three axes."""
        for ax in (self.ax_map, self.ax_panel, self.ax_bar):
            ax.set_facecolor(PARCHMENT)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    @staticmethod
    def _colour_map(gs: GameState) -> dict[str, str]:
        return {
            p.color: PLAYER_COLOURS[i % len(PLAYER_COLOURS)]
            for i, p in enumerate(gs.get_players())
        }

    # -------------------------------------------------------------------------
    #  MAP
    # -------------------------------------------------------------------------

    def _draw_map(self, gs: GameState) -> None:  # noqa: C901
        ax = self.ax_map
        ax.set_xlim(-1, 101)
        ax.set_ylim(-1, 101)
        ax.set_facecolor(OCEAN)

        # --- board image (if available) -------------------------------------
        try:
            ax.imshow(plt.imread("data/Risk_board.png"),
                      extent=(0, 100, 0, 100), zorder=0)
        except Exception:
            pass   # no base image — colour overlays will carry the rendering

        # --- continent tint overlays ----------------------------------------
        gs_countries = {
            c.get_name(): c for c in gs.get_game_map().get_countries()
        }
        for cont, members in CONTINENT_MEMBERS.items():
            pts = [TERRITORY_POS[t] for t in members if t in TERRITORY_POS]
            poly = _convex_hull_polygon(pts, pad=2.0)
            if poly is None:
                continue
            tint = CONTINENT_TINTS.get(cont, PARCHMENT)
            shapely_poly = ShapelyPolygon(poly)
            rounded = shapely_poly.buffer(3, join_style=1)
            rounded_coords = list(rounded.exterior.coords)

            ax.add_patch(Polygon(
                rounded_coords, closed=True,
                facecolor=to_rgba(tint, 0.38),
                edgecolor=to_rgba(INK_FAINT, 0.5),
                linewidth=0.6, linestyle="--",
                zorder=1,
            ))

        # --- parchment texture + vignette -----------------------------------
        _parchment_texture_overlay(ax)
        _vignette_overlay(ax)

        # --- adjacency lines ------------------------------------------------
        cmap = self._colour_map(gs)

        # for a, b in ADJACENCY:
        #     if a not in TERRITORY_POS or b not in TERRITORY_POS:
        #         continue
        #     x0, y0 = TERRITORY_POS[a]
        #     x1, y1 = TERRITORY_POS[b]
        #     is_sea = frozenset({a, b}) in SEA_CONNECTIONS
        #     ax.plot(
        #         [x0, x1], [y0, y1],
        #         color=INK_FAINT,
        #         linewidth=0.55 if not is_sea else 0.45,
        #         linestyle=(0, (4, 4)) if is_sea else "solid",
        #         alpha=0.55 if not is_sea else 0.40,
        #         zorder=2,
        #         solid_capstyle="round",
        #     )

        # --- territory nodes ------------------------------------------------
        R = 1.55   # token radius in data coordinates
        current_player = gs.get_current_player()

        for name, (x, y) in TERRITORY_POS.items():
            c = gs_countries.get(name)
            if c is None:
                continue

            owner  = c.get_owner()
            armies = c.get_army_size() if c else 0
            fill   = cmap.get(owner.color, NEUTRAL_COLOUR)   # type: ignore
            is_active = (owner == current_player and owner != "")

            # Drop-shadow (offset dark circle)
            ax.add_patch(Circle(
                (x + 0.35, y - 0.35), R,
                color=INK, alpha=0.22, zorder=3,
            ))

            # Active-player highlight ring (antique gold)
            if is_active:
                ax.add_patch(Circle(
                    (x, y), R + 0.72,
                    color=GOLD_BORDER, alpha=0.55, zorder=3,
                ))
                ax.add_patch(Circle(
                    (x, y), R + 0.72,
                    fill=False,
                    edgecolor=GOLD_BORDER, linewidth=1.0,
                    alpha=0.9, zorder=3,
                ))

            # Main token circle
            ax.add_patch(Circle(
                (x, y), R,
                color=fill, zorder=4,
                ec=GOLD_BORDER if is_active else _darken(fill, 0.35),
                linewidth=1.4 if is_active else 0.9,
            ))

            # Inner ring for a 3-D token look
            ax.add_patch(Circle(
                (x, y), R * 0.72,
                fill=False, zorder=5,
                edgecolor=_lighten(fill, 0.4),
                linewidth=0.6, alpha=0.7,
            ))

            # Army count — bold serif
            ax.text(
                x, y + 0.15, str(armies),
                color="white", ha="center", va="center",
                fontsize=6.8, fontweight="bold",
                #fontfamily=SERIF_FAMILY,
                zorder=6,
                path_effects=[
                    pe.withStroke(linewidth=1.6,
                                  foreground=_darken(fill, 0.5))
                ],
            )

            # Territory name label
            ax.text(
                x, y - R - 0.45, _short(name),
                color=INK,
                ha="center", va="top",
                fontsize=4.6,
                #fontfamily=SERIF_FAMILY,
                zorder=6,
                bbox=dict(
                    facecolor=to_rgba(PARCHMENT, 0.72),
                    edgecolor="none",
                    pad=0.5,
                    boxstyle="round,pad=0.25",
                ),
            )

        # --- decorative elements --------------------------------------------
        _draw_compass(ax, cx=5.5, cy=7.5, r=4.0)

        # Decorative title banner
        ax.text(
            50, 95.5,
            "RISK  -  THE GAME OF GLOBAL DOMINATION",
            color=INK, ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            # fontfamily=TITLE_FAMILY,
            fontstyle="italic",
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground=PARCHMENT)
            ],
            clip_on=False,
        )
        # Thin decorative rule beneath title
        ax.axhline(99.8, color=INK_FAINT, linewidth=0.7,
                   xmin=0.05, xmax=0.95, zorder=7)

    # -------------------------------------------------------------------------
    #  SIDE PANEL  (scoreboard)
    # -------------------------------------------------------------------------

    def _draw_panel(self, gs: GameState) -> None:
        ax = self.ax_panel
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Panel background — slightly deeper parchment with a ruled border
        ax.add_patch(FancyBboxPatch(
            (0.03, 0.01), 0.94, 0.97,
            boxstyle="round,pad=0.01",
            facecolor=PARCHMENT_DARK,
            edgecolor=INK_FAINT,
            linewidth=1.0,
            transform=ax.transAxes,
        ))

        game_map = gs.get_game_map()
        players  = gs.get_players()
        cmap     = self._colour_map(gs)
        total    = len(game_map.get_countries())

        # Panel heading
        ax.text(0.5, 0.965, "SCOREBOARD",
                color=INK, ha="center", va="top",
                fontsize=9.5, fontweight="bold",
                # fontfamily=TITLE_FAMILY,
                fontstyle="italic",
                transform=ax.transAxes)
        _rule(ax, 0.940)

        # --- Player rows ----------------------------------------------------
        n = len(players)
        card_h = min(0.155, 0.72 / max(n, 1))
        gap    = 0.010

        for i, player in enumerate(players):
            y_top = 0.925 - i * (card_h + gap)
            col   = cmap[player.color]            # type: ignore
            owned = game_map.get_owned_countries(player)
            n_c   = len(owned)
            n_a   = sum(c.get_army_size() for c in owned)
            pct   = n_c / total if total else 0
            is_current = (player == gs.get_current_player())

            y_bot = y_top - card_h

            # Row background — subtle tint matching player colour
            ax.add_patch(FancyBboxPatch(
                (0.06, y_bot), 0.88, card_h,
                boxstyle="square,pad=0.0",
                facecolor=to_rgba(col, 0.10),
                edgecolor=col if is_current else to_rgba(INK_FAINT, 0.4),
                linewidth=1.5 if is_current else 0.5,
                transform=ax.transAxes,
            ))

            mid_y = y_bot + card_h / 2

            # Active-turn marker (small triangle)
            if is_current:
                ax.text(0.07, mid_y + 0.006, "▶",
                        color=INK, ha="center", va="center",
                        fontsize=5.5, fontweight="bold",
                        transform=ax.transAxes, zorder=4)

            # Player name
            ax.text(0.22, mid_y + 0.018, str(player),  # type: ignore
                    color=INK, ha="left", va="center",
                    fontsize=7.0, fontweight="bold",
                    # fontfamily=SERIF_FAMILY,
                    transform=ax.transAxes)

            # Territory / army counts
            ax.text(0.22, mid_y - 0.018,
                    f"{n_c} territories   {n_a} armies",
                    color=INK_MID, ha="left", va="center",
                    fontsize=5.2, # fontfamily=SERIF_FAMILY,
                    transform=ax.transAxes)

            # Dominance bar (simple horizontal ink-fill)
            bar_y  = y_bot + 0.010
            bar_h  = 0.014
            bar_x0 = 0.08
            bar_w  = 0.84
            # background track
            ax.add_patch(FancyBboxPatch(
                (bar_x0, bar_y), bar_w, bar_h,
                boxstyle="round,pad=0.002",
                facecolor=PARCHMENT,
                edgecolor=to_rgba(INK_FAINT, 0.5),
                linewidth=0.4,
                transform=ax.transAxes,
            ))
            # filled portion
            if pct > 0:
                ax.add_patch(FancyBboxPatch(
                    (bar_x0, bar_y), bar_w * pct, bar_h,
                    boxstyle="round,pad=0.002",
                    facecolor=col,
                    edgecolor="none",
                    alpha=0.80,
                    transform=ax.transAxes,
                ))
            ax.text(bar_x0 + bar_w + 0.015, bar_y + bar_h / 2,
                    f"{pct:.0%}",
                    color=INK_MID, va="center", ha="left",
                    fontsize=4.8, # fontfamily=SERIF_FAMILY,
                    transform=ax.transAxes)

        # --- Continent legend -----------------------------------------------
        sep_y = 0.925 - n * (card_h + gap) - 0.018
        _rule(ax, sep_y)
        ax.text(0.5, sep_y - 0.010, "Continents",
                color=INK, ha="center", va="top",
                fontsize=7.0, fontweight="bold", fontstyle="italic",
                # fontfamily=TITLE_FAMILY,
                transform=ax.transAxes)

        gs_ctrs = {c.get_name(): c for c in game_map.get_countries()}

        cy = sep_y - 0.055
        for cont_name, members in CONTINENT_MEMBERS.items():
            c_objs = [gs_ctrs[t] for t in members if t in gs_ctrs]
            owners = {c.get_owner() for c in c_objs} if c_objs else {""}

            reward = "?"
            for cont in game_map.get_continents():
                if {c.get_name() for c in cont.get_countries()} & set(members):
                    reward = cont.get_reward()   # type: ignore
                    break

            if len(owners) == 1 and "" not in owners:
                owner_obj = next(iter(owners))
                col = cmap.get(owner_obj, NEUTRAL_COLOUR)
                owner_label = str(owner_obj)
                text_col = col
            else:
                col = NEUTRAL_COLOUR
                owner_label = "contested"
                text_col = INK_FAINT

            # Tint swatch
            ax.add_patch(FancyBboxPatch(
                (0.07, cy - 0.018), 0.014, 0.025,
                boxstyle="square,pad=0.0",
                facecolor=CONTINENT_TINTS.get(cont_name, PARCHMENT),
                edgecolor=INK_FAINT, linewidth=0.4,
                transform=ax.transAxes,
            ))

            ax.text(0.10, cy + 0.005,
                    f"{cont_name[:13]}  +{reward}",
                    color=text_col, fontsize=7.4,
                    # fontfamily=SERIF_FAMILY,
                    va="top",
                    transform=ax.transAxes)
            ax.text(0.10, cy - 0.01,
                    f"  {owner_label}",
                    color=text_col, fontsize=6.8, fontstyle="italic",
                    # fontfamily=SERIF_FAMILY,
                    va="top", alpha=0.80,
                    transform=ax.transAxes)
            cy -= 0.048
            if cy < 0.02:
                break

    # -------------------------------------------------------------------------
    #  STATUS BAR (turn tracker)
    # -------------------------------------------------------------------------

    def _draw_status_bar(self, gs: GameState) -> None:
        ax = self.ax_bar
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Bar background
        ax.add_patch(FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="square,pad=0.0",
            facecolor=PARCHMENT_DARK,
            edgecolor=INK_FAINT, linewidth=0.8,
            transform=ax.transAxes,
        ))
        # Top rule
        ax.axhline(0.92, color=INK_FAINT, linewidth=0.6,
                   xmin=0.01, xmax=0.99)

        phases = ["Place Armies", "Attack", "Fortify"]
        cur    = gs.get_phase()
        player = gs.get_current_player()
        cmap   = self._colour_map(gs)
        p_col  = cmap.get(player.color, NEUTRAL_COLOUR)   # type: ignore

        # Player label
        ax.text(0.012, 0.52,
                f"Turn:  {player.color}",           # type: ignore
                color=p_col, va="center",
                fontsize=9.5, fontweight="bold",
                # fontfamily=SERIF_FAMILY,
                transform=ax.transAxes)

        # Phase indicators — classic underline style, no glowing pills
        phase_x_start = 0.26
        phase_spacing = 0.13

        for i, label in enumerate(phases):
            x  = phase_x_start + i * phase_spacing
            active = (i == cur)
            txt_col = INK if active else INK_FAINT

            ax.text(x, 0.62, label,
                    color=txt_col,
                    ha="left", va="center",
                    fontsize=8.0,
                    fontweight="bold" if active else "normal",
                    # fontfamily=SERIF_FAMILY,
                    transform=ax.transAxes)

            # Underline active phase with an antique-gold rule
            if active:
                # Approximate text width
                ax.annotate(
                    "", xy=(x + 0.105, 0.26), xytext=(x - 0.003, 0.26),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-",
                                    color=GOLD_BORDER,
                                    lw=2.0),
                )

            # Bullet separator
            if i < len(phases) - 1:
                ax.text(x + 0.108, 0.62, "·",
                        color=INK_FAINT, ha="left", va="center",
                        fontsize=9, transform=ax.transAxes)

        # Phase counter (right-aligned)
        ax.text(0.988, 0.52,
                f"Phase {cur + 1} of {len(phases)}",
                color=INK_FAINT, va="center", ha="right",
                fontsize=7.0, # fontfamily=SERIF_FAMILY,
                transform=ax.transAxes)


# =============================================================================
#  SMALL UTILITY FUNCTIONS
# =============================================================================

def _rule(ax: plt.Axes, y: float, alpha: float = 0.7) -> None:
    """Draw a full-width horizontal rule at normalised y in ax.transAxes."""
    ax.axhline(y, color=SEPARATOR_COL, linewidth=0.8,
               xmin=0.05, xmax=0.95, alpha=alpha)


def _darken(hex_col: str, amount: float = 0.3) -> str:
    """Return a darkened version of *hex_col* (amount in [0,1])."""
    r, g, b, a = to_rgba(hex_col)
    r = max(0.0, r * (1 - amount))
    g = max(0.0, g * (1 - amount))
    b = max(0.0, b * (1 - amount))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _lighten(hex_col: str, amount: float = 0.3) -> str:
    """Return a lightened version of *hex_col* (amount in [0,1])."""
    r, g, b, a = to_rgba(hex_col)
    r = min(1.0, r + (1 - r) * amount)
    g = min(1.0, g + (1 - g) * amount)
    b = min(1.0, b + (1 - b) * amount)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"