from dataclasses import dataclass

import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_PALETTE: dict[str, str] = {
    "primary": "#4fc3f7",
    "secondary": "#ff6b6b",
    "success": "#2ca02c",
    "danger": "#d62728",
    "neutral": "#808080",
    "background": "#1f1f1f",
    "grid": "#222222",
    "text": "#ffffff",
    "color0": "#4fc3f7",
    "color1": "#00D7F3",
    "color2": "#00E7DB",
    "color3": "#66F3B5",
    "color4": "#B1F98D",
    "color5": "#F9F871",
}


@dataclass(frozen=True)
class PlotStyle:
    paper_bgcolor: str = DEFAULT_PALETTE["background"]
    plot_bgcolor: str = DEFAULT_PALETTE["background"]
    font_color: str = DEFAULT_PALETTE["text"]
    accent1: str = DEFAULT_PALETTE["primary"]
    accent2: str = DEFAULT_PALETTE["secondary"]
    accent3: str = DEFAULT_PALETTE["success"]
    accent4: str = DEFAULT_PALETTE["danger"]
    accent5: str = DEFAULT_PALETTE["neutral"]

    color0: str = DEFAULT_PALETTE["color0"]
    color1: str = DEFAULT_PALETTE["color1"]
    color2: str = DEFAULT_PALETTE["color2"]
    color3: str = DEFAULT_PALETTE["color3"]
    color4: str = DEFAULT_PALETTE["color4"]
    color5: str = DEFAULT_PALETTE["color5"]

    accent6: str = "#FFBE7D"
    grid: str = DEFAULT_PALETTE["grid"]
    line: str = DEFAULT_PALETTE["neutral"]
    muted: str = DEFAULT_PALETTE["neutral"]
    font_size: int = 10
    title_size: int = 12

    def apply(self, fig: go.Figure) -> go.Figure:
        """Apply style to Plotly figure."""
        fig.update_layout(
            paper_bgcolor=self.paper_bgcolor,
            plot_bgcolor=self.plot_bgcolor,
            font=dict(color=self.font_color, size=12),
            margin=dict(l=60, r=30, t=40, b=50),
            legend=dict(
                bgcolor=self.plot_bgcolor,
                bordercolor=self.line,
                borderwidth=1,
            ),
            xaxis=dict(
                gridcolor=self.grid,
                zeroline=False,
                linecolor=self.line,
                tickfont=dict(color=self.font_color),
                title=dict(font=dict(color=self.font_color)),
            ),
            yaxis=dict(
                gridcolor=self.grid,
                zeroline=False,
                linecolor=self.line,
                tickfont=dict(color=self.font_color),
                title=dict(font=dict(color=self.font_color)),
            ),
            title=dict(font=dict(color=self.font_color, size=14)),
        )
        return fig

    def apply_mpl(self, fig: Figure | None = None, ax: Axes | None = None) -> Figure:
        """Apply style to Matplotlib figure/axes."""
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.gcf()

        # Set figure background
        fig.patch.set_facecolor(self.paper_bgcolor)

        # Style all axes if no specific ax provided
        axes = [ax] if ax is not None else fig.get_axes()

        for ax in axes:
            # Background
            ax.set_facecolor(self.plot_bgcolor)

            # Grid
            ax.grid(True, alpha=0.3, color=self.grid, linestyle="-", linewidth=0.5)

            # Spines
            for spine in ax.spines.values():
                spine.set_edgecolor(self.line)
                spine.set_linewidth(0.8)

            # Tick labels
            ax.tick_params(colors=self.font_color, labelsize=self.font_size)

            # Axis labels
            ax.xaxis.label.set_color(self.font_color)
            ax.yaxis.label.set_color(self.font_color)
            ax.xaxis.label.set_fontsize(self.font_size)
            ax.yaxis.label.set_fontsize(self.font_size)

            # Title
            if ax.get_title():
                ax.title.set_color(self.font_color)
                ax.title.set_fontsize(self.title_size)

            # Legend (if exists)
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor(self.plot_bgcolor)
                legend.get_frame().set_edgecolor(self.line)
                for text in legend.get_texts():
                    text.set_color(self.font_color)

        return fig


DEFAULT_STYLE: PlotStyle = PlotStyle()
