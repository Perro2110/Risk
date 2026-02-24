"""
Risk/visualizer.py

Visual display of a Risk GameState using matplotlib, with territory positions
matching the actual classic Risk board geography (all 42 territories).

Usage:
    from Risk.visualizer import RiskVisualizer
    viz = RiskVisualizer(game_state)
    viz.show()              # blocking display
    viz.save("board.png")   # save to file
    viz.update(game_state)  # redraw with updated state
    plt.pause(0.1)          # non-blocking refresh inside a simulation loop
"""

from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Risk.game_state import GameState

# -----------------------------------------------------------------------------
#  COLOUR PALETTE
# -----------------------------------------------------------------------------
PLAYER_COLOURS = [
    "#E63946",  # red
    "#457B9D",  # steel blue
    "#2DC653",  # green
    "#F4A261",  # orange
    "#9B5DE5",  # purple
    "#F7B731",  # yellow
    "#00B4D8",  # cyan
    "#FF6B6B",  # coral
]
NEUTRAL_COLOUR = "#666677"

# Continent background tints
CONTINENT_COLOURS = {
    "North America": "#0D2B1A",
    "South America": "#1A0D2B",
    "Europe":        "#0D272B",
    "Africa":        "#2B1A0D",
    "Asia":          "#2B0D0D",
    "Australia":     "#0D2B2B",
}

BG_COLOUR = "#0B1520"
MAP_BG = "#719299"
PANEL_BG = "#0E1A28"
TEXT_LIGHT = "#E8E8E8"
TEXT_DIM = "#778899"
EDGE_COL = "#2A3A4A"
GOLD = "#FFD700"


# -----------------------------------------------------------------------------
#  GEOGRAPHIC POSITIONS  (x: 0–100 = west→east,  y: 0–100 = south→north)
#  Hand-tuned to match the classic Risk board layout.
# -----------------------------------------------------------------------------
TERRITORY_POS: dict[str, tuple[float, float]] = {
    # -- North America ------------------------------------------------------
    "Alaska":                (7.4,  83.9),
    "Northwest Territory":   (17.5, 84.6),
    "Greenland":             (34.9, 88.5),
    "Alberta":               (15.8, 76.5),
    "Ontario":               (22.2, 74.8),
    "Quebec":                (28.9, 74.5),
    "Western United States": (16.1, 65.8),
    "Eastern United States": (23.9, 63.5),
    "Central America":       (17.6, 53.0),

    # -- South America ------------------------------------------------------
    "Venezuela":             (24.0, 45.5),
    "Peru":                  (23.8, 33.5),
    "Brazil":                (32.0, 37.5),
    "Argentina":             (25.0, 23.0),

    # -- Europe -------------------------------------------------------------
    "Iceland":               (42.9, 79.6),
    "Great Britain":         (40.5, 69.0),
    "Northern Europe":       (50.3, 66.6),
    "Scandinavia":           (50.3, 80.4),
    "Ukraine":               (58.7, 71.6),
    "Western Europe":        (42.5, 55.0),
    "Southern Europe":       (51.0, 58.0),

    # -- Africa -------------------------------------------------------------
    "North Africa":          (46.7, 39.9),
    "Egypt":                 (54.3, 44.0),
    "East Africa":           (59.3, 33.8),
    "Congo":                 (53.8, 28.1),
    "South Africa":          (55.0, 15.0),
    "Madagascar":            (63.3, 14.9),

    # -- Asia ---------------------------------------------------------------
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

    # -- Australia ----------------------------------------------------------
    "Indonesia":             (81.2, 29.1),
    "New Guinea":            (89.4, 33.2),
    "Western Australia":     (86.3, 16.0),
    "Eastern Australia":     (95.0, 19.2),
}

# -- Classic Risk adjacency ---------------------------------------------------
ADJACENCY: list[tuple[str, str]] = [
    # North America internal
    ("Alaska", "Northwest Territory"),
    ("Alaska", "Alberta"),
    ("Northwest Territory", "Alberta"),
    ("Northwest Territory", "Ontario"),
    ("Northwest Territory", "Greenland"),
    ("Greenland", "Ontario"),
    ("Greenland", "Quebec"),
    ("Alberta", "Ontario"),
    ("Alberta", "Western United States"),
    ("Ontario", "Quebec"),
    ("Ontario", "Western United States"),
    ("Ontario", "Eastern United States"),
    ("Quebec", "Eastern United States"),
    ("Western United States", "Eastern United States"),
    ("Western United States", "Central America"),
    ("Eastern United States", "Central America"),
    # North America → South America
    ("Central America", "Venezuela"),
    # South America internal
    ("Venezuela", "Peru"),
    ("Venezuela", "Brazil"),
    ("Peru", "Brazil"),
    ("Peru", "Argentina"),
    ("Brazil", "Argentina"),
    # South America → Africa (trans-Atlantic)
    ("Brazil", "North Africa"),
    # Europe internal
    ("Iceland", "Great Britain"),
    ("Iceland", "Scandinavia"),
    ("Great Britain", "Northern Europe"),
    ("Great Britain", "Scandinavia"),
    ("Great Britain", "Western Europe"),
    ("Northern Europe", "Scandinavia"),
    ("Northern Europe", "Ukraine"),
    ("Northern Europe", "Western Europe"),
    ("Northern Europe", "Southern Europe"),
    ("Scandinavia", "Ukraine"),
    ("Ukraine", "Southern Europe"),
    ("Ukraine", "Middle East"),
    ("Ukraine", "Afghanistan"),
    ("Ukraine", "Ural"),
    ("Western Europe", "Southern Europe"),
    ("Western Europe", "North Africa"),
    ("Southern Europe", "North Africa"),
    ("Southern Europe", "Egypt"),
    ("Southern Europe", "Middle East"),
    # Africa internal
    ("North Africa", "Egypt"),
    ("North Africa", "East Africa"),
    ("North Africa", "Congo"),
    ("Egypt", "East Africa"),
    ("Egypt", "Middle East"),
    ("East Africa", "Congo"),
    ("East Africa", "South Africa"),
    ("East Africa", "Madagascar"),
    ("East Africa", "Middle East"),
    ("Congo", "South Africa"),
    ("South Africa", "Madagascar"),
    # Asia internal
    ("Ural", "Afghanistan"),
    ("Ural", "Siberia"),
    ("Ural", "China"),
    ("Siberia", "Yakutsk"),
    ("Siberia", "Irkutsk"),
    ("Siberia", "Mongolia"),
    ("Siberia", "China"),
    ("Yakutsk", "Kamchatka"),
    ("Yakutsk", "Irkutsk"),
    ("Kamchatka", "Irkutsk"),
    ("Kamchatka", "Mongolia"),
    ("Kamchatka", "Japan"),
    ("Irkutsk", "Mongolia"),
    ("Mongolia", "Japan"),
    ("Mongolia", "China"),
    ("Afghanistan", "China"),
    ("Afghanistan", "India"),
    ("Afghanistan", "Middle East"),
    ("China", "India"),
    ("China", "Siam"),
    ("India", "Middle East"),
    ("India", "Siam"),
    # Asia → Australia
    ("Siam", "Indonesia"),
    # Australia internal
    ("Indonesia", "New Guinea"),
    ("Indonesia", "Western Australia"),
    ("New Guinea", "Eastern Australia"),
    ("Western Australia", "Eastern Australia"),
    # Trans-Pacific: Alaska ↔ Kamchatka
    ("Alaska", "Kamchatka"),
    # Greenland ↔ Iceland (trans-Atlantic)
    ("Greenland", "Iceland"),
]

# Continent membership (for hull drawing & panel)
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
        "North Africa", "Egypt", "East Africa", "Congo", "South Africa",
        "Madagascar",
    ],
    "Asia": [
        "Ural", "Siberia", "Yakutsk", "Kamchatka", "Irkutsk", "Mongolia",
        "Japan", "Afghanistan", "China", "Middle East", "India", "Siam",
    ],
    "Australia": [
        "Indonesia", "New Guinea", "Western Australia",
        "Eastern Australia"
    ],
}

# Short labels to prevent node text overflow
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


def _short(name: str) -> str:
    return SHORT_LABEL.get(name, name)


# ------------------------------------------------------------------------------
#  MAIN CLASS
# ------------------------------------------------------------------------------
class RiskVisualizer:
    """
    Renders a Risk GameState as a geographically-positioned classic board map.

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
    ):
        matplotlib.rcParams["font.family"] = "monospace"

        self.fig = plt.figure(figsize=figsize, facecolor=BG_COLOUR)

        self.ax_map = self.fig.add_axes((0.00, 0.07, 0.78, 0.91))
        self.ax_panel = self.fig.add_axes((0.79, 0.07, 0.20, 0.91))
        self.ax_bar = self.fig.add_axes((0.00, 0.00, 1.00, 0.065))

        for ax in (self.ax_map, self.ax_panel, self.ax_bar):
            ax.set_facecolor(PANEL_BG if ax is not self.ax_map else MAP_BG)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#1A3050")

        self.draw(game_state)

    # -- public API -----------------------------------------------------------

    def show(self, blocking: bool = True) -> None:
        """Display the board (blocking)."""
        plt.show(block=blocking)

    def save(self, path: str, dpi: int = 180) -> None:
        """Save board image to *path*."""
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight",
                         facecolor=BG_COLOUR)

    def update(self, game_state: GameState) -> None:
        """Redraw all axes with a new game state (use inside a loop)."""
        for ax in (self.ax_map, self.ax_panel, self.ax_bar):
            ax.cla()
            ax.set_facecolor(PANEL_BG if ax is not self.ax_map else MAP_BG)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#1A3050")
        self.draw(game_state)
        self.fig.canvas.draw_idle()

    # -- orchestration --------------------------------------------------------

    def draw(self, gs: GameState) -> None:
        self._draw_map(gs)
        self._draw_panel(gs)
        self._draw_status_bar(gs)

    # -- MAP ------------------------------------------------------------------

    def _draw_map(self, gs: GameState) -> None:
        ax = self.ax_map
        ax.set_xlim(-1, 101)
        ax.set_ylim(-1, 101)
        ax.imshow(plt.imread('data/Risk_board.png'), extent=(0, 100, 0, 100))

        colour_map: dict[str, str] = {
            p: PLAYER_COLOURS[i % len(PLAYER_COLOURS)]
            for i, p in enumerate(gs.player_names)
        }
        gs_countries = {
            c.get_name(): c for c in gs.get_game_map().get_countries()
        }
        current_player = gs.get_current_player()

        # -- edges ------------------------------------------------------------
        # drawn: set[frozenset] = set()
        # for t1, t2 in ADJACENCY:
        #     key = frozenset((t1, t2))
        #     if key in drawn:
        #         continue
        #     drawn.add(key)
        #     if t1 not in TERRITORY_POS or t2 not in TERRITORY_POS:
        #         continue
        #     x1, y1 = TERRITORY_POS[t1]
        #     x2, y2 = TERRITORY_POS[t2]
        #     # Mark trans-oceanic routes with dashes
        #     dx = abs(x1 - x2)
        #     ls = (0, (5, 4)) if dx > 28 else "solid"
        #     ax.plot([x1, x2], [y1, y2],
        #             color=EDGE_COL, linewidth=0.9,
        #             linestyle=ls, zorder=2, alpha=0.75)

        # -- nodes ------------------------------------------------------------
        R = 1.3  # node radius in data coords

        for name, (x, y) in TERRITORY_POS.items():
            c = gs_countries.get(name)

            if c is None:
                continue

            owner = c.get_owner()
            armies = c.get_army_size() if c else 0

            fill = colour_map.get(owner.color, NEUTRAL_COLOUR)
            is_active = (owner == current_player and owner != "")

            # gold glow ring for current-player territories
            if is_active:
                ax.add_patch(Circle(
                    (x, y), R + 0.9,
                    color=GOLD, alpha=0.35, zorder=3,
                ))

            # main circle
            ax.add_patch(Circle(
                (x, y), R,
                color=fill, zorder=4,
                ec=GOLD if is_active else "#0B1520",
                linewidth=1.6 if is_active else 0.8,
            ))

            # army count
            ax.text(x, y + 0.25, str(armies),
                    color="white", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", zorder=5)

            # territory name tag
            ax.text(x, y - R - 0.4, _short(name),
                    color=TEXT_LIGHT, ha="center", va="top",
                    fontsize=4.8, zorder=5,
                    bbox=dict(facecolor=BG_COLOUR, edgecolor="none",
                              alpha=0.55, pad=0.4,
                              boxstyle="round,pad=0.2"))

        ax.set_title(
            "◈  RISK — CLASSIC BOARD  ◈",
            color=TEXT_LIGHT, fontsize=12, fontweight="bold",
            pad=6, loc="center", fontfamily="monospace",
        )

    # -- SIDE PANEL -----------------------------------------------------------

    def _draw_panel(self, gs: GameState) -> None:
        ax = self.ax_panel
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        game_map = gs.get_game_map()
        players = gs.get_players()
        colour_map = {
            p.color: PLAYER_COLOURS[i % len(PLAYER_COLOURS)]
            for i, p in enumerate(players)
        }
        total = len(game_map.get_countries())

        ax.text(0.5, 0.975, "PLAYERS",
                color=TEXT_LIGHT, ha="center", va="top",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.axhline(0.945, color="#1A3050", linewidth=1,
                   xmin=0.05, xmax=0.95)

        n = len(players)
        card_h = min(0.17, 0.80 / max(n, 1))
        gap = 0.012

        for i, player in enumerate(players):
            y_top = 0.935 - i * (card_h + gap)
            col = colour_map[player.color]
            owned = game_map.get_owned_countries(player)
            n_c = len(owned)
            n_a = sum(c.get_army_size() for c in owned)
            pct = n_c / total if total else 0
            is_current = (player == gs.get_current_player())

            ax.add_patch(FancyBboxPatch(
                (0.04, y_top - card_h), 0.92, card_h,
                boxstyle="round,pad=0.01",
                facecolor=col + "25",
                edgecolor=col if is_current else "#1A3050",
                linewidth=2.0 if is_current else 0.6,
                transform=ax.transAxes,
            ))

            ax.add_patch(Circle(
                (0.13, y_top - card_h / 2), 0.046,
                color=col, transform=ax.transAxes, zorder=3,
            ))
            if is_current:
                ax.text(0.13, y_top - card_h / 2, "▶",
                        color="white", ha="center", va="center",
                        fontsize=5.5, fontweight="bold",
                        transform=ax.transAxes, zorder=4)

            mid = y_top - card_h / 2
            ax.text(0.25, mid + 0.014, player,
                    color=TEXT_LIGHT, ha="left", va="center",
                    fontsize=7.5, fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.25, mid - 0.022,
                    f"{n_c} terr.   * {n_a} armies",
                    color=TEXT_DIM, ha="left", va="center",
                    fontsize=5.5, transform=ax.transAxes)

            # progress bar
            bar_y = y_top - card_h + 0.011
            ax.add_patch(FancyBboxPatch(
                (0.07, bar_y), 0.84, 0.018,
                boxstyle="round,pad=0.003",
                facecolor="#0A1520", edgecolor="none",
                transform=ax.transAxes))
            if pct > 0:
                ax.add_patch(FancyBboxPatch(
                    (0.07, bar_y), 0.84 * pct, 0.018,
                    boxstyle="round,pad=0.003",
                    facecolor=col, edgecolor="none",
                    transform=ax.transAxes))
            ax.text(0.93, bar_y + 0.009, f"{pct:.0%}",
                    color=col, va="center", ha="left",
                    fontsize=5.2, transform=ax.transAxes)

        # -- Continent summary ------------------------------------------------
        sep_y = 0.935 - n * (card_h + gap) - 0.015
        ax.axhline(sep_y, color="#1A3050", linewidth=0.8,
                   xmin=0.05, xmax=0.95)
        ax.text(0.5, sep_y - 0.008, "CONTINENTS",
                color=TEXT_LIGHT, ha="center", va="top",
                fontsize=7, fontweight="bold", transform=ax.transAxes)

        gs_countries = {c.get_name(): c for c in game_map.get_countries()}

        cy = sep_y - 0.055
        for cont_name, members in CONTINENT_MEMBERS.items():
            c_objs = [gs_countries.get(t) for t in members
                      if gs_countries.get(t) is not None]
            owners = {c.get_owner() for c in c_objs if c is not None} \
                if c_objs else {""}
            # look up reward from the game map
            reward = "?"
            for cont in game_map.get_continents():
                cont_c_names = {c.get_name() for c in cont.get_countries()}
                if cont_c_names & set(members):
                    reward = cont.get_reward()  # type: ignore
                    break
            if len(owners) == 1 and "" not in owners:
                owner = next(iter(owners))
                col = colour_map.get(owner, NEUTRAL_COLOUR)
                lbl = f"● {cont_name[:12]}  +{reward}"
                sub = f"  → {owner}"
            else:
                col = TEXT_DIM
                lbl = f"○ {cont_name[:12]}  +{reward}"
                sub = "  contested"
            ax.text(0.06, cy, lbl, color=col, fontsize=5.5, va="top",
                    transform=ax.transAxes)
            ax.text(0.06, cy - 0.032, sub, color=col, fontsize=5.0, va="top",
                    alpha=0.75, transform=ax.transAxes)
            cy -= 0.072
            if cy < 0.01:
                break

    # -- STATUS BAR -----------------------------------------------------------

    def _draw_status_bar(self, gs: GameState) -> None:
        ax = self.ax_bar
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        phases = [
            ("⚑", "PLACE ARMY", "#2DC653"),
            ("⚔", "ATTACK",     "#E63946"),
            ("↺", "FORTIFY",    "#F4A261"),
        ]
        cur = gs.get_phase()
        player = gs.get_current_player()
        p_col = PLAYER_COLOURS[
            gs.player_names.index(player.color) % len(PLAYER_COLOURS)
        ]

        ax.text(0.01, 0.5, f"  TURN ▶  {player.color}",
                color=p_col, va="center",
                fontsize=10, fontweight="bold",
                transform=ax.transAxes)

        pill_w = 0.05
        for i, (icon, label, col) in enumerate(phases):
            x = 0.28 + i * (pill_w + 0.04)
            active = (i == cur)
            ax.add_patch(FancyBboxPatch(
                (x, 0.12), pill_w, 0.76,
                boxstyle="round,pad=0.02",
                facecolor=col if active else col + "18",
                edgecolor=col,
                linewidth=1.8 if active else 0.8,
                transform=ax.transAxes,
            ))
            ax.text(x + pill_w / 2, 0.5,
                    f"{icon} {label}",
                    color="white" if active else col,
                    ha="center", va="center",
                    fontsize=8,
                    fontweight="bold" if active else "normal",
                    transform=ax.transAxes)

        ax.text(0.99, 0.5, f"Phase {cur + 1}/3",
                color=TEXT_DIM, va="center", ha="right",
                fontsize=7.5, transform=ax.transAxes)
