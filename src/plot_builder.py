from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SUPPORTED_PLOT_KINDS = {"scatter", "line", "bar", "histogram"}


@dataclass
class PlotSeries:
    x: list[float | int | str] = field(default_factory=list)
    y: list[float | int] = field(default_factory=list)
    label: str = ""
    color: str | None = None


@dataclass
class PlotSpec:
    kind: str
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    series: list[PlotSeries] = field(default_factory=list)
    bins: int = 10
    alpha: float = 0.8
    figsize: tuple[float, float] = (8.0, 4.5)
    grid: bool = True


@dataclass
class PlotBatchItem:
    filename: str
    plot: PlotSpec


def create_plot_from_spec(spec: PlotSpec | dict[str, Any], output_path: str | Path) -> Path:
    plot_spec = spec if isinstance(spec, PlotSpec) else plot_spec_from_dict(spec)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if plot_spec.kind not in SUPPORTED_PLOT_KINDS:
        raise ValueError(f"Unsupported plot kind: {plot_spec.kind}")

    figure, axis = plt.subplots(figsize=plot_spec.figsize)
    try:
        if plot_spec.kind == "scatter":
            _draw_scatter(axis, plot_spec)
        elif plot_spec.kind == "line":
            _draw_line(axis, plot_spec)
        elif plot_spec.kind == "bar":
            _draw_bar(axis, plot_spec)
        elif plot_spec.kind == "histogram":
            _draw_histogram(axis, plot_spec)

        _apply_common_style(axis, plot_spec)
        figure.tight_layout()
        figure.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)

    return output


def create_plots_from_batch(
    batch: list[PlotBatchItem] | list[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    items = batch if batch and isinstance(batch[0], PlotBatchItem) else plot_batch_from_list(batch)  # type: ignore[index]
    generated_paths = []
    for item in items:
        generated_paths.append(create_plot_from_spec(item.plot, output_root / item.filename))
    return generated_paths


def create_plots_from_manifest(manifest_path: str | Path, output_dir: str | Path) -> list[Path]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The plot manifest must be a JSON array.")
    return create_plots_from_batch(payload, output_dir)


def create_scatter_plot(
    x: list[float | int],
    y: list[float | int],
    output_path: str | Path,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str | None = None,
) -> Path:
    return create_plot_from_spec(
        PlotSpec(
            kind="scatter",
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=[PlotSeries(x=x, y=y, color=color)],
        ),
        output_path,
    )


def create_line_plot(
    x: list[float | int | str],
    y: list[float | int],
    output_path: str | Path,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str | None = None,
) -> Path:
    return create_plot_from_spec(
        PlotSpec(
            kind="line",
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=[PlotSeries(x=x, y=y, color=color)],
        ),
        output_path,
    )


def create_bar_chart(
    categories: list[str],
    values: list[float | int],
    output_path: str | Path,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str | None = None,
) -> Path:
    return create_plot_from_spec(
        PlotSpec(
            kind="bar",
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=[PlotSeries(x=categories, y=values, color=color)],
        ),
        output_path,
    )


def create_histogram(
    values: list[float | int],
    output_path: str | Path,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    bins: int = 10,
    color: str | None = None,
) -> Path:
    return create_plot_from_spec(
        PlotSpec(
            kind="histogram",
            title=title,
            x_label=x_label,
            y_label=y_label,
            bins=bins,
            series=[PlotSeries(x=values, color=color)],
        ),
        output_path,
    )


def plot_spec_from_dict(payload: dict[str, Any]) -> PlotSpec:
    series_payload = payload.get("series") or []
    series = [
        PlotSeries(
            x=list(item.get("x") or []),
            y=list(item.get("y") or []),
            label=str(item.get("label", "")),
            color=item.get("color"),
        )
        for item in series_payload
    ]
    return PlotSpec(
        kind=str(payload.get("kind", "")).strip().lower(),
        title=str(payload.get("title", "")),
        x_label=str(payload.get("x_label", "")),
        y_label=str(payload.get("y_label", "")),
        series=series,
        bins=int(payload.get("bins", 10)),
        alpha=float(payload.get("alpha", 0.8)),
        figsize=tuple(payload.get("figsize", (8.0, 4.5))),
        grid=bool(payload.get("grid", True)),
    )


def plot_batch_from_list(payload: list[dict[str, Any]]) -> list[PlotBatchItem]:
    items = []
    for index, item in enumerate(payload, start=1):
        filename = str(item.get("filename", "")).strip() or f"plot_{index}.png"
        plot_payload = item.get("plot") or item
        items.append(PlotBatchItem(filename=filename, plot=plot_spec_from_dict(plot_payload)))
    return items


def build_plot_generation_instructions() -> str:
    return """
Return a JSON object describing a simple plot to generate in Python.

Supported kinds:
- scatter
- line
- bar
- histogram

JSON schema:
{
  "kind": "scatter | line | bar | histogram",
  "title": "string",
  "x_label": "string",
  "y_label": "string",
  "bins": 10,
  "alpha": 0.8,
  "figsize": [8.0, 4.5],
  "grid": true,
  "series": [
    {
      "x": [1, 2, 3],
      "y": [4, 5, 6],
      "label": "optional legend label",
      "color": "optional matplotlib color"
    }
  ]
}

Rules:
- For a histogram, put the numeric values in series[0].x and omit y.
- For a scatter plot, x and y must have the same length.
- For a bar chart, x should be category labels and y the numeric values.
- Prefer short readable titles and axis labels.
- Do not return markdown, only JSON.
""".strip()


def build_plot_batch_generation_instructions() -> str:
    return """
Return a JSON array. Each item describes one plot file to generate in Python.

Expected format:
[
  {
    "filename": "scatter_revenue_vs_margin.png",
    "plot": {
      "kind": "scatter",
      "title": "Revenue vs Margin",
      "x_label": "Revenue",
      "y_label": "Margin",
      "series": [
        {
          "x": [100, 120, 150],
          "y": [12, 15, 18],
          "label": "Products A-C",
          "color": "#4C78A8"
        }
      ]
    }
  }
]

Supported kinds:
- scatter
- line
- bar
- histogram

Rules:
- Return only raw JSON, never markdown.
- Each item must include a filename ending in .png.
- Put the exact numeric or categorical values to plot in the JSON.
- For histogram, put numeric values in plot.series[0].x and omit y.
- For scatter and line, x and y must have the same length.
- For bar, x must contain category labels and y the numeric values.
- Keep titles and axis labels short and presentation-ready.
""".strip()


def _draw_scatter(axis, spec: PlotSpec) -> None:
    _require_series(spec, allowed_multiple=True)
    for series in spec.series:
        _validate_xy_lengths(series)
        axis.scatter(series.x, series.y, label=series.label or None, color=series.color, alpha=spec.alpha)


def _draw_line(axis, spec: PlotSpec) -> None:
    _require_series(spec, allowed_multiple=True)
    for series in spec.series:
        _validate_xy_lengths(series)
        axis.plot(series.x, series.y, label=series.label or None, color=series.color, alpha=spec.alpha)


def _draw_bar(axis, spec: PlotSpec) -> None:
    _require_series(spec, allowed_multiple=False)
    series = spec.series[0]
    _validate_xy_lengths(series)
    axis.bar(series.x, series.y, color=series.color, alpha=spec.alpha, label=series.label or None)


def _draw_histogram(axis, spec: PlotSpec) -> None:
    _require_series(spec, allowed_multiple=False)
    series = spec.series[0]
    if not series.x:
        raise ValueError("Histogram requires numeric values in series[0].x")
    axis.hist(series.x, bins=spec.bins, color=series.color, alpha=spec.alpha, label=series.label or None)


def _apply_common_style(axis, spec: PlotSpec) -> None:
    axis.set_title(spec.title)
    axis.set_xlabel(spec.x_label)
    axis.set_ylabel(spec.y_label)
    if spec.grid:
        axis.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axis.get_legend_handles_labels()
    if handles and any(labels):
        axis.legend(frameon=False)


def _require_series(spec: PlotSpec, allowed_multiple: bool) -> None:
    if not spec.series:
        raise ValueError("Plot spec requires at least one series")
    if not allowed_multiple and len(spec.series) != 1:
        raise ValueError(f"{spec.kind} plot expects exactly one series")


def _validate_xy_lengths(series: PlotSeries) -> None:
    if len(series.x) != len(series.y):
        raise ValueError("Series x and y must have the same length")
