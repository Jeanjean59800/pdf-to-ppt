from pathlib import Path
import json
import re

import cv2
import fitz

from src.types import PipelineData, VisualAsset


def extract_pdf_text_to_folder(pdf_path: Path, output_base_dir: Path) -> Path:
    pdf_name = pdf_path.stem
    output_dir = output_base_dir / f"{pdf_name}_extracted_text"
    output_dir.mkdir(parents=True, exist_ok=True)

    text_output_path = output_dir / f"{pdf_name}_text.txt"
    doc = fitz.open(str(pdf_path))

    all_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        all_text += f"\n--- Page {page_num + 1} ---\n"
        all_text += page.get_text("text")

    text_output_path.write_text(all_text, encoding="utf-8")
    doc.close()
    return text_output_path


def render_pdf_pages(pdf_path: Path, output_base_dir: Path) -> list[Path]:
    pdf_name = pdf_path.stem
    output_dir = output_base_dir / f"{pdf_name}_rendered_pages"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        image_path = output_dir / f"page_{page_num + 1}.png"
        pix.save(str(image_path))
        image_paths.append(image_path)

    doc.close()
    return image_paths


def crop_candidate_visuals(page_images: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_paths = []

    for image_path in page_images:
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cropped_img = img[y : y + h, x : x + w]
            output_path = output_dir / f"{image_path.stem}_crop_{i}.png"
            cv2.imwrite(str(output_path), cropped_img)
            cropped_paths.append(output_path)

        print(f"Processed {image_path.name}")

    return cropped_paths


def delete_too_small_files(folder_path: Path, size_threshold_kb: int) -> int:
    size_threshold_bytes = size_threshold_kb * 1024
    deleted_count = 0

    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.stat().st_size < size_threshold_bytes:
            file_path.unlink()
            deleted_count += 1

    print(f"Cleanup complete. Removed {deleted_count} files smaller than {size_threshold_kb} KB.")
    return deleted_count


def build_visual_inventory(cropped_folder: Path) -> list[VisualAsset]:
    visuals = []
    for path in sorted(cropped_folder.glob("*.png")):
        visuals.append(
            VisualAsset(
                path=path,
                source=guess_source_page(path.name),
                description=describe_visual_placeholder(path),
            )
        )
    return visuals


def guess_source_page(filename: str) -> str:
    match = re.search(r"page_(\d+)", filename)
    if match:
        return f"page {match.group(1)}"
    return "unknown page"


def describe_visual_placeholder(path: Path) -> str:
    size_kb = path.stat().st_size / 1024
    return f"Candidate visual from {guess_source_page(path.name)} ({size_kb:.1f} KB). Needs semantic labeling."


def format_visual_inventory_for_prompt(visuals: list[VisualAsset], max_items: int = 12) -> str:
    if not visuals:
        return "- No reliable cropped visual found yet."
    return "\n".join(f"- {asset.path.name}: {asset.description}" for asset in visuals[:max_items])


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_text_from_pdf(pdf_path: Path, artifacts_dir: Path) -> Path:
    return extract_pdf_text_to_folder(pdf_path, artifacts_dir / "text")


def extract_images_and_graphs_from_pdf(
    pdf_path: Path,
    artifacts_dir: Path,
    size_threshold_kb: int,
) -> tuple[list[Path], list[VisualAsset]]:
    rendered_pages_dir = artifacts_dir / "rendered_pages"
    candidate_visuals_dir = artifacts_dir / "candidate_visuals" / f"{pdf_path.stem}_crops"

    page_images = render_pdf_pages(pdf_path, rendered_pages_dir)
    crop_candidate_visuals(page_images, candidate_visuals_dir)
    delete_too_small_files(candidate_visuals_dir, size_threshold_kb)

    cropped_visuals = build_visual_inventory(candidate_visuals_dir)
    return page_images, cropped_visuals


def write_debug_manifest(data: PipelineData, artifacts_dir: Path) -> Path:
    manifest_path = artifacts_dir / "manifest.json"
    manifest = {
        "pdf": str(data.pdf_path),
        "raw_text": str(data.raw_text_path),
        "rendered_pages": [str(path) for path in data.page_images],
        "candidate_visuals": [
            {"path": str(asset.path), "source": asset.source, "description": asset.description}
            for asset in data.cropped_visuals
        ],
        "generated_plots": [str(path) for path in data.generated_plot_paths],
        "slide_plan": str(data.slide_plan_path),
        "output_pptx": str(data.output_pptx),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
