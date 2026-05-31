from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class VisualAsset:
    path: Path
    source: str
    description: str


@dataclass
class PipelineData:
    pdf_path: Path
    raw_text_path: Path
    page_images: List[Path]
    cropped_visuals: List[VisualAsset]
    generated_plot_paths: List[Path]
    thesis_summary: str
    slide_plan_path: Path
    output_pptx: Path
