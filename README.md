# Clean PPT Pipeline Demo

Prototype pipeline that turns a source PDF into an editable PowerPoint draft.

The goal is not to claim perfect slide quality. The goal is to demonstrate a realistic end-to-end system:
- local PDF text extraction
- visual extraction from rendered pages
- section planning and slide generation
- optional Gemini usage for text or image annotation
- FAISS-based retrieval to anchor slide content in the source document
- editable `.pptx` export
- run-level metrics for timing and component usage

## What The Pipeline Does

At a high level, the pipeline:
1. extracts raw text from a PDF with `PyMuPDF`
2. renders pages as images and crops candidate visuals with OpenCV heuristics
3. optionally annotates visuals with Gemini
4. builds a FAISS index over the document text
5. creates a section plan
6. generates slide content section by section, using RAG context
7. exports an editable PowerPoint deck with a small set of supported layouts
8. saves intermediate artifacts and run metrics

## Project Structure

```text
clean_ppt_pipeline_demo/
  data/                  Input PDFs
  output/                Generated outputs and final runs
  reference_decks/       Human / external reference decks used for comparison
  src/
    document_analysis.py RAG-backed planning and slide generation
    faiss_rag.py         FAISS index, chunking, embedding helpers
    gemini_client.py     Gemini text generation wrapper
    pdf_pipeline.py      Text extraction, page rendering, visual cropping
    plot_builder.py      Optional generated charts from structured manifests
    ppt_builder.py       Editable PPTX generation
    run_metrics.py       Timing and usage instrumentation
    types.py             Shared dataclasses
  main.py                Orchestrates the full pipeline
  requirements.txt
  placeholder.env
```

## Current Execution Model

Most configuration is currently hardcoded in [main.py](./main.py), including:
- input PDF path
- output paths
- max slide count
- whether Gemini visual annotation is enabled
- whether FAL image generation is enabled
- whether analysis fallback is allowed

This was kept intentionally simple for demo speed.

## How To Run

Install dependencies, configure `.env`, then run:

```powershell
python main.py
```

By default, `main.py` points to:

```python
PDF_PATH = DATA_DIR / "mna_test_pdf.pdf"
```

To run the other bundled test PDF, change it directly in [main.py](./main.py):

```python
# PDF_PATH = DATA_DIR / "mna_test_pdf.pdf"
PDF_PATH = DATA_DIR / "Projet_IA_ROY_VANHUYSSE.pdf"
```

## Environment Variables

Expected in `.env` when the corresponding features are enabled:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
FAL_KEY=...
```

## Outputs

Typical outputs:
- `output/.../artifacts/analysis/slide_plan.json`
- `output/.../artifacts/analysis/run_metrics.json`
- `output/.../artifacts/manifest.json`
- `output/.../artifacts/text/...`
- `output/.../artifacts/rendered_pages/...`
- `output/.../artifacts/candidate_visuals/...`
- `output/.../pptx/generated_presentation_skeleton.pptx`

The published example output currently kept in the repository is:

```text
output/mna/
```

## PowerPoint Generation

The PowerPoint builder currently supports a small set of layouts:
- `text_only`
- `text_image_right`
- `image_focus`
- `two_images_compare`
- `chart_focus`
- `section_break`

The deck content is generated as structured JSON plus lightweight markdown-like body content, then rendered into `.pptx`.

## RAG Design

The current RAG flow:
- builds a FAISS index over chunked document text
- creates a section plan with retrieval queries
- fetches section-specific context for each section
- generates slides from that local context instead of from one monolithic prompt

This improves coverage and reduces generic slide content compared with the earlier single-prompt version.

Important note:
- chunks currently contain mostly raw text
- page-level source metadata is not yet preserved through the full generation pipeline

Next step:
- preserve page-level metadata and attach source IDs to each generated slide bullet

## Benchmarking And Run Metrics

Each run records:
- stage timings
- API usage counters
- local LLM usage
- FAISS / embedding usage
- section planning and section generation counts
- coverage repair counters

This is saved in:

```text
artifacts/analysis/run_metrics.json
```

## Limitations / What I Would Improve Today

- Chunking is character-based and should become page-aware or section-aware.
- Visual extraction uses OpenCV heuristics and can fail on sparse or vector-based charts.
- Evaluation is limited to run metrics and manual review.
- Model configuration should be standardized for deployment.
- The PowerPoint template should be improved with a stronger reusable visual design and most layouts should be constrained/fixed.

## Additional Technical Notes

- Generated charts are supported, but only if a structured manifest exists. Thus, no graphs are produced in the current pipeline and a method should be implemented to allow the LLM to use that.
