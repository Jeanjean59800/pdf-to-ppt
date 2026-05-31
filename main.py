from pathlib import Path

from dotenv import load_dotenv

from src.document_analysis import (
    analyze_document,
    analyze_document_with_local_placeholder,
    annotate_visuals_with_gemini,
    parse_slide_plan,
    save_slide_plan,
)
from src.pdf_pipeline import (
    extract_images_and_graphs_from_pdf,
    extract_text_from_pdf,
    read_text_file,
    write_debug_manifest,
)
from src.plot_builder import create_plots_from_manifest
from src.ppt_builder import build_editable_pptx
from src.run_metrics import run_metrics
from src.types import PipelineData


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
PPTX_DIR = OUTPUT_DIR / "pptx"
GENERATED_PLOTS_DIR = OUTPUT_DIR / "generated_plots"

PDF_PATH = DATA_DIR / "mna_test_pdf.pdf"
# Alternative test PDF:
# PDF_PATH = DATA_DIR / "Projet_IA_ROY_VANHUYSSE.pdf"

OUTPUT_PPTX = PPTX_DIR / "generated_presentation_skeleton.pptx"
PLOT_MANIFEST_PATH = PROJECT_DIR / "plot_manifest.json"
SIZE_THRESHOLD_KB = 5
MAX_TEXT_FOR_GEMINI = 50000
MAX_SLIDES = 12
ENABLE_FAL_IMAGE_GENERATION = False
ALLOW_ANALYSIS_FALLBACK = False
ENABLE_GEMINI_VISUAL_ANNOTATION = False


def _print_run_metrics(metrics_path: Path) -> None:
    snapshot = run_metrics.snapshot()
    print("Run metrics")
    print(f"- Metrics JSON: {metrics_path}")

    counters = snapshot["counters"]
    print("- Usage counters")
    for name in [
        "api.gemini.generate_text.calls",
        "api.gemini.describe_visual.calls",
        "api.gemini.generate_pdf_transcript.calls",
        "api.gemini.upload_image.calls",
        "api.gemini.upload_pdf.calls",
        "rag.index_build.calls",
        "rag.search.calls",
        "rag.embedding.calls",
        "rag.embedding.texts_total",
        "rag.chunks.created",
        "analysis.section_plan.calls",
        "analysis.section_plan.success",
        "analysis.section_slides.calls",
        "analysis.section_slides.success",
        "analysis.section_slides.deterministic_fallback",
        "analysis.coverage.tags_total",
        "analysis.coverage.tags_missing",
        "analysis.coverage.repairs",
        "pipeline.sections.contexts_built",
        "local_llm.loads",
        "local_llm.reused",
        "local_llm.inference.calls",
        "local_llm.fallback_placeholder.used",
        "pipeline.visuals.total",
        "pipeline.visuals.annotated",
        "pipeline.visuals.annotation_failures",
    ]:
        print(f"  {name}: {counters.get(name, 0)}")

    timings = snapshot["timings"]
    print("- Timings")
    for name in [
        "stage.extract_text.seconds",
        "stage.extract_visuals.seconds",
        "stage.annotate_visuals.seconds",
        "stage.analyze_document.seconds",
        "stage.rag_build.seconds",
        "stage.generate_plots.seconds",
        "stage.build_pptx.seconds",
        "api.gemini.generate_text.seconds",
        "api.gemini.describe_visual.seconds",
        "api.gemini.generate_pdf_transcript.seconds",
        "api.gemini.upload_image.seconds",
        "api.gemini.upload_pdf.seconds",
        "rag.index_build.seconds",
        "rag.search.seconds",
        "rag.embedding.seconds",
        "local_llm.load.seconds",
        "local_llm.inference.seconds",
    ]:
        entry = timings.get(name, {"count": 0, "total_seconds": 0.0, "avg_seconds": 0.0})
        print(
            f"  {name}: count={entry['count']} total={entry['total_seconds']}s avg={entry['avg_seconds']}s"
        )


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PPTX_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/6 Extracting text")
    with run_metrics.timed("stage.extract_text.seconds"):
        raw_text_path = extract_text_from_pdf(PDF_PATH, ARTIFACTS_DIR)
        document_text = read_text_file(raw_text_path)

    print("2/6 Extracting pages and candidate visuals")
    with run_metrics.timed("stage.extract_visuals.seconds"):
        page_images, cropped_visuals = extract_images_and_graphs_from_pdf(
            pdf_path=PDF_PATH,
            artifacts_dir=ARTIFACTS_DIR,
            size_threshold_kb=SIZE_THRESHOLD_KB,
        )
    if ENABLE_GEMINI_VISUAL_ANNOTATION:
        with run_metrics.timed("stage.annotate_visuals.seconds"):
            cropped_visuals = annotate_visuals_with_gemini(cropped_visuals)

    print("3/6 Analyzing document")
    try:
        with run_metrics.timed("stage.analyze_document.seconds"):
            slide_plan_text = analyze_document(
                document_text=document_text,
                visuals=cropped_visuals,
                max_text_chars=MAX_TEXT_FOR_GEMINI,
            )
    except Exception as exc:
        if not ALLOW_ANALYSIS_FALLBACK:
            raise RuntimeError(f"Document analysis failed in strict mode: {exc}") from exc
        print(f"Document analysis failed, using local placeholder: {exc}")
        with run_metrics.timed("stage.analyze_document.seconds"):
            slide_plan_text = analyze_document_with_local_placeholder(document_text, cropped_visuals)

    slide_plan_path = save_slide_plan(slide_plan_text, ARTIFACTS_DIR)
    thesis_summary, _ = parse_slide_plan(slide_plan_text)

    print("4/6 Generating plots from manifest")
    with run_metrics.timed("stage.generate_plots.seconds"):
        if PLOT_MANIFEST_PATH.exists():
            generated_plot_paths = create_plots_from_manifest(PLOT_MANIFEST_PATH, GENERATED_PLOTS_DIR)
            print(f"Generated {len(generated_plot_paths)} plot(s) from {PLOT_MANIFEST_PATH.name}")
        else:
            generated_plot_paths = []
            print(f"No plot manifest found at {PLOT_MANIFEST_PATH}, skipping plot generation")

    data = PipelineData(
        pdf_path=PDF_PATH,
        raw_text_path=raw_text_path,
        page_images=page_images,
        cropped_visuals=cropped_visuals,
        generated_plot_paths=generated_plot_paths,
        thesis_summary=thesis_summary,
        slide_plan_path=slide_plan_path,
        output_pptx=OUTPUT_PPTX,
    )
    write_debug_manifest(data, ARTIFACTS_DIR)

    print("5/6 Building editable PPTX")
    with run_metrics.timed("stage.build_pptx.seconds"):
        build_editable_pptx(
            data=data,
            slide_plan_text=slide_plan_text,
            output_path=OUTPUT_PPTX,
            max_slides=MAX_SLIDES,
            enable_fal_image_generation=ENABLE_FAL_IMAGE_GENERATION,
        )

    print("6/6 Done")
    print(f"PPTX: {OUTPUT_PPTX}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    metrics_path = run_metrics.write_json(ARTIFACTS_DIR / "analysis" / "run_metrics.json")
    _print_run_metrics(metrics_path)


if __name__ == "__main__":
    main()
