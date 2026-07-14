"""
xg_visualization.py — Visualizzazioni per il modello xG
========================================================

Uso:
    from xg_visualization import XGVisualizer

    viz = XGVisualizer(df)
    viz.shot_map()
    viz.xg_distribution()
    viz.goal_rate_heatmap()
    viz.calibration_plot(y_pred, y_true)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


class XGVisualizer:
    """Visualizzazioni per analisi xG su dati StatsBomb."""

    # Costanti del campo StatsBomb (condivise da tutti i metodi)
    PITCH_LENGTH = 120
    PITCH_WIDTH = 80
    GOAL_CENTER_Y = 40

    def __init__(self, df):
        """
        Parametri:
            df: DataFrame con colonne shot_x, shot_y, is_goal
                (e opzionalmente xg_pred, shot_statsbomb_xg)
        """
        # Validazione: il DataFrame ha le colonne minime necessarie?
        required = ["shot_x", "shot_y", "is_goal"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Colonne mancanti nel DataFrame: {missing}")

        self.df = df

    def _draw_pitch(self, ax, half=False):
        """
        Disegna il campo sul matplotlib axis dato.
        Metodo privato (prefisso _) perché è un helper interno,
        non qualcosa che l'utente chiama direttamente.

        Parametri:
            ax: matplotlib axis
            half: se True, disegna solo la metà d'attacco (x >= 60)
        """
        x_min = 60 if half else 0

        # Perimetro
        ax.plot(
            [x_min, 120, 120, x_min, x_min],
            [0, 0, 80, 80, 0],
            color="black", linewidth=1.5
        )

        if not half:
            # Linea di metà campo
            ax.plot([60, 60], [0, 80], color="black")
            # Cerchio di centrocampo
            ax.add_patch(patches.Circle((60, 40), 10, fill=False, color="black"))
            # Area di rigore sinistra
            ax.plot([0, 18, 18, 0], [18, 18, 62, 62], color="black")
            ax.plot([0, 6, 6, 0], [30, 30, 50, 50], color="black")
            ax.plot(12, 40, marker="o", markersize=3, color="black")

        # Area di rigore destra (quella verso cui si tira)
        ax.plot([120, 102, 102, 120], [18, 18, 62, 62], color="black")
        # Area piccola destra
        ax.plot([120, 114, 114, 120], [30, 30, 50, 50], color="black")
        # Dischetto del rigore
        ax.plot(108, 40, marker="o", markersize=3, color="black")

        ax.set_xlim(x_min - 2, 122)
        ax.set_ylim(-2, 82)
        ax.set_aspect("equal")
        ax.axis("off")

    def shot_map(self, save_path=None):
        """
        Scatter plot di tutti i tiri, gol in rosso, non-gol in grigio.

        Parametri:
            save_path: se fornito, salva il plot come immagine
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        self._draw_pitch(ax, half=True)

        goals = self.df[self.df["is_goal"] == 1]
        no_goals = self.df[self.df["is_goal"] == 0]

        ax.scatter(
            no_goals["shot_x"], no_goals["shot_y"],
            c="lightgrey", s=15, alpha=0.5, label="No goal"
        )
        ax.scatter(
            goals["shot_x"], goals["shot_y"],
            c="red", s=40, alpha=0.8, label="Goal"
        )

        ax.set_title("Shot Map — Serie A 2015/16")
        ax.legend(loc="upper left")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def shot_map_xg(self, xg_column="xg_pred", save_path=None):
        """
        Scatter plot con colore proporzionale all'xG.

        Parametri:
            xg_column: nome della colonna xG da usare per il colore
            save_path: se fornito, salva il plot come immagine
        """
        if xg_column not in self.df.columns:
            raise ValueError(f"Colonna '{xg_column}' non trovata nel DataFrame")

        fig, ax = plt.subplots(figsize=(10, 7))
        self._draw_pitch(ax, half=True)

        scatter = ax.scatter(
            self.df["shot_x"], self.df["shot_y"],
            c=self.df[xg_column],
            cmap="Reds", s=30, alpha=0.7,
            edgecolors="black", linewidths=0.3
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label("xG")

        ax.set_title(f"Shot Map per xG ({xg_column})")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def xg_distribution(self, xg_column="xg_pred", save_path=None):
        """Istogramma della distribuzione delle predizioni xG."""
        if xg_column not in self.df.columns:
            raise ValueError(f"Colonna '{xg_column}' non trovata nel DataFrame")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            self.df[xg_column], bins=50,
            color="steelblue", edgecolor="black"
        )
        ax.set_xlabel("xG predetto")
        ax.set_ylabel("Numero di tiri")
        ax.set_title("Distribuzione delle predizioni xG")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def goal_rate_heatmap(self, save_path=None):
        """Heatmap empirica: % gol per zona del campo."""
        fig, ax = plt.subplots(figsize=(10, 7))
        self._draw_pitch(ax, half=True)

        x_bins = np.linspace(60, 120, 13)
        y_bins = np.linspace(0, 80, 9)

        H, xedges, yedges = np.histogram2d(
            self.df["shot_x"], self.df["shot_y"],
            bins=[x_bins, y_bins],
            weights=self.df["is_goal"]
        )
        counts, _, _ = np.histogram2d(
            self.df["shot_x"], self.df["shot_y"],
            bins=[x_bins, y_bins]
        )
        goal_rate = np.divide(
            H, counts,
            out=np.zeros_like(H),
            where=counts > 0
        )

        im = ax.imshow(
            goal_rate.T,
            extent=[60, 120, 0, 80],
            origin="lower",
            cmap="Reds",
            alpha=0.7,
            aspect="auto"
        )
        plt.colorbar(im, ax=ax, shrink=0.6, label="% gol")
        ax.set_title("Heatmap: percentuale gol per zona")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def calibration_plot(self, y_pred, y_true, n_bins=10, save_path=None):
        """
        Calibration plot: predizioni vs realtà.

        Parametri:
            y_pred: array delle predizioni xG
            y_true: array dei risultati reali (0/1)
            n_bins: numero di bin (default 10)
        """
        temp = pd.DataFrame({"pred": y_pred, "actual": y_true})
        temp["bin"] = pd.qcut(temp["pred"], q=n_bins, duplicates="drop")

        cal = temp.groupby("bin", observed=True).agg(
            predicted=("pred", "mean"),
            actual=("actual", "mean"),
            count=("actual", "count")
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.5, label="Calibrazione perfetta")
        ax.scatter(
            cal["predicted"], cal["actual"],
            s=cal["count"] / 3,
            zorder=5, label="Modello"
        )

        ax.set_xlabel("xG predetto (media del bin)")
        ax.set_ylabel("% gol effettivi")
        ax.set_title("Calibration Plot")
        ax.legend()
        ax.set_xlim(0, 0.45)
        ax.set_ylim(0, 0.45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()