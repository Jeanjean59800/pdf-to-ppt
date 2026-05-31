import json
import os
from pathlib import Path

from src.faiss_rag import FaissTextIndex
from src.gemini_client import generate_text
from src.pdf_pipeline import format_visual_inventory_for_prompt
from src.run_metrics import run_metrics
from src.types import VisualAsset


LOCAL_LLM_MODEL_PATH = Path(r"C:\Users\jeanl\Desktop\local_gpt\models\gpt-oss-20b-F16.gguf")
USE_LOCAL_LLM = True
_LOCAL_LLM = None
ALLOWED_LAYOUTS = {
    "text_only",
    "text_image_right",
    "image_focus",
    "two_images_compare",
    "chart_focus",
    "section_break",
}
ALLOWED_VISUALS = {
    "extracted figure",
    "generated illustration",
    "simple diagram",
    "text only",
}


def generate_pdf_transcript(pdf_path: str | Path, prompt: str = "Give me a transcript of this pdf file.") -> str:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    run_metrics.increment("api.gemini.upload_pdf.calls")
    with run_metrics.timed("api.gemini.upload_pdf.seconds"):
        uploaded_pdf = genai.upload_file(str(pdf_path))
    run_metrics.increment("api.gemini.generate_pdf_transcript.calls")
    with run_metrics.timed("api.gemini.generate_pdf_transcript.seconds"):
        response = model.generate_content([prompt, uploaded_pdf])
    return response.text


def describe_visual_with_gemini(image_path: str | Path) -> str:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    run_metrics.increment("api.gemini.upload_image.calls")
    with run_metrics.timed("api.gemini.upload_image.seconds"):
        uploaded_image = genai.upload_file(str(image_path))
    run_metrics.increment("api.gemini.describe_visual.calls")
    with run_metrics.timed("api.gemini.describe_visual.seconds"):
        response = model.generate_content(
            [
                "Describe this extracted PDF figure for PowerPoint reuse. "
                "State what it is, the likely topic, whether it looks like a chart, table, diagram, screenshot, or illustration, "
                "and give a short presentation-friendly caption in 1 to 3 sentences.",
                uploaded_image,
            ]
        )
    return response.text.strip()


def annotate_visuals_with_gemini(visuals: list[VisualAsset]) -> list[VisualAsset]:
    run_metrics.increment("pipeline.visuals.total", len(visuals))
    for asset in visuals:
        try:
            asset.description = describe_visual_with_gemini(asset.path)
            run_metrics.increment("pipeline.visuals.annotated")
        except Exception:
            run_metrics.increment("pipeline.visuals.annotation_failures")
    return visuals


def analyze_document(
    document_text: str,
    visuals: list[VisualAsset],
    max_text_chars: int,
) -> str:
    index = build_document_index(document_text[:max_text_chars])
    section_plan = build_section_plan(document_text, visuals, max_text_chars)
    slide_plan = build_slide_plan_from_sections(section_plan, index, visuals)
    return serialize_slide_plan(slide_plan)


def analyze_document_with_local_placeholder(
    document_text: str,
    visuals: list[VisualAsset],
    max_text_chars: int = 12000,
) -> str:
    try:
        return analyze_document(document_text, visuals, max_text_chars=max_text_chars)
    except Exception:
        run_metrics.increment("local_llm.fallback_placeholder.used")
        return serialize_slide_plan(build_placeholder_slide_plan(document_text, visuals))


def build_document_index(document_text: str) -> FaissTextIndex:
    with run_metrics.timed("stage.rag_build.seconds"):
        return FaissTextIndex.from_text(document_text, chunk_size=1000, overlap=120)


def build_section_plan(document_text: str, visuals: list[VisualAsset], max_text_chars: int) -> dict:
    planning_context = build_planning_context(document_text[:max_text_chars], max_chars=4500)
    prompt = f"""
You are planning a presentation from a long PDF.

Return exactly one JSON object and nothing else.

Schema:
{{
  "thesis": "short paragraph",
  "sections": [
    {{
      "id": "sec_1",
      "title": "section title",
      "goal": "what this section must explain",
      "transition_from_previous": "short bridge from previous section",
      "coverage_tags": ["short tag 1", "short tag 2"],
      "queries": ["retrieval query 1", "retrieval query 2"],
      "slide_count": 1
    }}
  ]
}}

Rules:
- Return 4 to 7 sections.
- Total planned slides across sections must be between 6 and 10.
- Make the sections collectively cover the major document blocks, not just the introduction.
- For each section, add 2 to 4 `coverage_tags` naming the concepts or blocks that must appear in the final deck.
- Each query must be specific enough to retrieve relevant document chunks.
- Keep section titles concise.
- Do not output markdown.

Available visual assets:
{format_visual_inventory_for_prompt(visuals)}

Document excerpts:
{planning_context}
""".strip()
    payload = _run_json_generation(prompt, "analysis.section_plan")
    return normalize_section_plan(payload)


def build_slide_plan_from_sections(section_plan: dict, index: FaissTextIndex, visuals: list[VisualAsset]) -> dict:
    slides = []
    previous_summary = ""
    section_contexts: dict[str, str] = {}
    for section in section_plan["sections"]:
        context = build_section_context(index, section)
        section_contexts[section["id"]] = context
        try:
            payload = build_section_slides(section, context, previous_summary, visuals)
            normalized_slides = normalize_section_slides(payload, section["id"])
        except Exception:
            run_metrics.increment("analysis.section_slides.deterministic_fallback")
            normalized_slides = build_deterministic_section_slides(section, context)
        slides.extend(normalized_slides)
        previous_summary = normalized_slides[-1]["title"] if normalized_slides else previous_summary
    if not slides:
        raise ValueError("No slides generated from section plan.")
    slide_plan = {"thesis": section_plan["thesis"], "slides": slides}
    slide_plan = enforce_slide_plan_coverage(slide_plan, section_plan, section_contexts)
    return slide_plan


def build_section_context(index: FaissTextIndex, section: dict) -> str:
    parts = []
    for query in section["queries"]:
        context = index.build_context(query=query, top_k=4, max_context_chars=2500)
        if context:
            parts.append(f"QUERY: {query}\n{context}")
    if not parts:
        raise ValueError(f"No RAG context found for section {section['id']}.")
    run_metrics.increment("pipeline.sections.contexts_built")
    return "\n\n".join(parts)


def build_section_slides(section: dict, context: str, previous_summary: str, visuals: list[VisualAsset]) -> dict:
    anchor_points = extract_context_sentences(context, limit=5)
    prompt = f"""
You are writing PowerPoint slides for one section of a presentation.

Return exactly one JSON object and nothing else.

Schema:
{{
  "slides": [
    {{
      "title": "slide title",
      "layout": "text_only | text_image_right | image_focus | two_images_compare | chart_focus | section_break",
      "visual": "extracted figure | generated illustration | simple diagram | text only",
      "content_md": "markdown content for the body"
    }}
  ]
}}

Rules:
- Produce exactly {section['slide_count']} slides for this section.
- Use only the allowed layout values.
- Use only the allowed visual values.
- `content_md` may use only short paragraphs, `##`, `-`, `**bold**`, `*italic*`.
- No code fences, no tables, no HTML.
- Each slide must be concise and presentation-ready.
- At least one slide in the section should directly address the section goal.
- Every slide must rely on the retrieved context, not on generic M&A knowledge.
- Reuse at least 2 concrete anchor points from the retrieved context across the section.
- Use `section_break` only if the section should open with a transition slide.

Section metadata:
- id: {section['id']}
- title: {section['title']}
- goal: {section['goal']}
- transition_from_previous: {section['transition_from_previous']}
- coverage_tags: {", ".join(section["coverage_tags"])}
- previous_section_summary: {previous_summary or 'N/A'}

Available visual assets:
{format_visual_inventory_for_prompt(visuals)}

Context anchor points that should be reflected explicitly:
{format_anchor_points(anchor_points)}

Retrieved context:
{context}
""".strip()
    return _run_json_generation(prompt, "analysis.section_slides")


def parse_slide_plan(plan_text: str) -> tuple[str, list[dict]]:
    payload = _extract_json_payload(plan_text)
    normalized = normalize_slide_plan(payload)
    return normalized["thesis"], normalized["slides"]


def save_slide_plan(slide_plan_text: str, artifacts_dir: Path) -> Path:
    path = artifacts_dir / "analysis" / "slide_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(slide_plan_text, encoding="utf-8")
    return path


def serialize_slide_plan(payload: dict) -> str:
    return json.dumps(normalize_slide_plan(payload), indent=2, ensure_ascii=True)


def normalize_section_plan(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Section plan must be a JSON object.")
    thesis = str(payload.get("thesis", "")).strip()
    sections_payload = payload.get("sections")
    if not thesis:
        raise ValueError("Section plan thesis is missing.")
    if not isinstance(sections_payload, list) or not sections_payload:
        raise ValueError("Section plan sections are missing.")

    normalized_sections = []
    total_slide_count = 0
    for index, item in enumerate(sections_payload, start=1):
        if not isinstance(item, dict):
            continue
        section_id = str(item.get("id", "")).strip() or f"sec_{index}"
        title = str(item.get("title", "")).strip()
        goal = str(item.get("goal", "")).strip()
        transition = str(item.get("transition_from_previous", "")).strip()
        queries_payload = item.get("queries", [])
        coverage_tags_payload = item.get("coverage_tags", [])
        slide_count = int(item.get("slide_count", 1))
        if not isinstance(queries_payload, list):
            raise ValueError(f"Section {section_id} queries must be a list.")
        if not isinstance(coverage_tags_payload, list):
            raise ValueError(f"Section {section_id} coverage_tags must be a list.")
        queries = [str(query).strip() for query in queries_payload if str(query).strip()]
        coverage_tags = [str(tag).strip().lower() for tag in coverage_tags_payload if str(tag).strip()]
        if not title or not goal or not queries or not coverage_tags:
            raise ValueError(f"Section {section_id} is incomplete.")
        slide_count = min(max(slide_count, 1), 2)
        total_slide_count += slide_count
        normalized_sections.append(
            {
                "id": section_id,
                "title": title,
                "goal": goal,
                "transition_from_previous": transition,
                "coverage_tags": coverage_tags[:4],
                "queries": queries[:3],
                "slide_count": slide_count,
            }
        )
    if not normalized_sections:
        raise ValueError("No valid sections in section plan.")
    if total_slide_count < 6 or total_slide_count > 10:
        raise ValueError(f"Planned slide count out of range: {total_slide_count}.")
    return {"thesis": thesis, "sections": normalized_sections}


def normalize_section_slides(payload: dict, section_id: str) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Section slide payload must be a JSON object.")
    slides_payload = payload.get("slides")
    if not isinstance(slides_payload, list) or not slides_payload:
        raise ValueError(f"Section {section_id} returned no slides.")

    slides = []
    for item in slides_payload:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        layout = str(item.get("layout", "text_only")).strip().lower()
        visual = str(item.get("visual", "text only")).strip().lower()
        content_md = str(item.get("content_md", "")).strip()
        if not title or not content_md:
            raise ValueError(f"Section {section_id} contains an incomplete slide.")
        if layout not in ALLOWED_LAYOUTS:
            raise ValueError(f"Invalid layout '{layout}' in section {section_id}.")
        if visual not in ALLOWED_VISUALS:
            raise ValueError(f"Invalid visual '{visual}' in section {section_id}.")
        slides.append(
            {
                "title": title,
                "layout": layout,
                "content_md": content_md,
                "visual": visual,
                "section_id": section_id,
            }
        )
    if not slides:
        raise ValueError(f"Section {section_id} has no valid slides.")
    return slides


def normalize_slide_plan(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Slide plan payload must be a JSON object.")
    thesis = str(payload.get("thesis", "")).strip()
    slides_payload = payload.get("slides")
    if not thesis:
        raise ValueError("Slide plan thesis is missing.")
    if not isinstance(slides_payload, list) or not slides_payload:
        raise ValueError("Slide plan slides are missing.")

    slides = []
    for item in slides_payload:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        layout = str(item.get("layout", "text_only")).strip().lower()
        visual = str(item.get("visual", "text only")).strip().lower()
        content_md = str(item.get("content_md", "")).strip()
        if not title or not content_md:
            raise ValueError("Each slide must have a title and content_md.")
        if layout not in ALLOWED_LAYOUTS:
            raise ValueError(f"Slide '{title}' has invalid layout '{layout}'.")
        if visual not in ALLOWED_VISUALS:
            raise ValueError(f"Slide '{title}' has invalid visual '{visual}'.")
        slides.append(
            {
                "title": title,
                "layout": layout,
                "content_md": content_md,
                "visual": visual,
                "section_id": str(item.get("section_id", "")).strip(),
            }
        )
    if not slides:
        raise ValueError("Slide plan contains no valid slides.")
    return {"thesis": thesis, "slides": slides}


def build_deterministic_section_slides(section: dict, context: str) -> list[dict]:
    sentences = extract_context_sentences(context)
    overview_lines = ["## Key points"]
    for sentence in sentences[:3]:
        overview_lines.append(f"- {sentence}")
    overview_lines.append(f"- Focus block: {', '.join(section['coverage_tags'][:2])}")

    slides = [
        {
            "title": section["title"],
            "layout": infer_layout_from_section(section),
            "visual": infer_visual_from_section(section),
            "content_md": "\n".join(overview_lines),
            "section_id": section["id"],
        }
    ]

    if section["slide_count"] > 1:
        detail_lines = ["## Why it matters", f"- {section['goal']}"]
        if section["transition_from_previous"]:
            detail_lines.append(f"- Transition: {section['transition_from_previous']}")
        for sentence in sentences[3:5]:
            detail_lines.append(f"- {sentence}")
        if len(section["coverage_tags"]) > 2:
            detail_lines.append(f"- Coverage focus: {', '.join(section['coverage_tags'][2:4])}")
        slides.append(
            {
                "title": f"{section['title']} Details",
                "layout": "text_image_right" if infer_visual_from_section(section) == "extracted figure" else "text_only",
                "visual": infer_visual_from_section(section),
                "content_md": "\n".join(detail_lines),
                "section_id": section["id"],
            }
        )
    return slides


def extract_context_sentences(context: str, limit: int = 6) -> list[str]:
    compact = " ".join(context.split())
    raw_sentences = compact.split(". ")
    sentences = []
    for sentence in raw_sentences:
        cleaned = sentence.strip().strip(".")
        if len(cleaned) < 40:
            continue
        if cleaned.startswith("QUERY:"):
            continue
        if cleaned not in sentences:
            sentences.append(cleaned)
        if len(sentences) >= limit:
            break
    if not sentences:
        sentences.append("Relevant document context was retrieved but needs manual refinement.")
    return sentences


def infer_layout_from_section(section: dict) -> str:
    title_goal = f"{section['title']} {section['goal']}".lower()
    if any(token in title_goal for token in ("result", "performance", "comparison", "metric", "accuracy")):
        return "chart_focus"
    if any(token in title_goal for token in ("architecture", "workflow", "diagram", "method")):
        return "text_image_right"
    return "text_only"


def infer_visual_from_section(section: dict) -> str:
    title_goal = f"{section['title']} {section['goal']}".lower()
    if any(token in title_goal for token in ("figure", "chart", "diagram", "architecture", "workflow", "result")):
        return "extracted figure"
    return "text only"


def validate_slide_plan_coverage(slide_plan: dict, section_plan: dict) -> None:
    deck_text = " ".join(
        f"{slide.get('title', '')} {slide.get('content_md', '')}".lower()
        for slide in slide_plan["slides"]
    )
    total_tags = 0
    missing_by_section: dict[str, list[str]] = {}
    for section in section_plan["sections"]:
        matched = 0
        for tag in section["coverage_tags"]:
            total_tags += 1
            if tag in deck_text:
                matched += 1
            else:
                missing_by_section.setdefault(section["id"], []).append(tag)
        if matched == 0:
            raise ValueError(f"Coverage check failed for section {section['id']}: none of its tags appear in the deck.")
    run_metrics.increment("analysis.coverage.tags_total", total_tags)
    run_metrics.increment("analysis.coverage.tags_missing", len({tag for tags in missing_by_section.values() for tag in tags}))


def enforce_slide_plan_coverage(slide_plan: dict, section_plan: dict, section_contexts: dict[str, str]) -> dict:
    try:
        validate_slide_plan_coverage(slide_plan, section_plan)
        return slide_plan
    except ValueError:
        repaired_slides = [dict(slide) for slide in slide_plan["slides"]]
        for section in section_plan["sections"]:
            slides_for_section = [slide for slide in repaired_slides if slide.get("section_id") == section["id"]]
            if not slides_for_section:
                continue
            section_text = " ".join(
                f"{slide.get('title', '')} {slide.get('content_md', '')}".lower() for slide in slides_for_section
            )
            missing_tags = [tag for tag in section["coverage_tags"] if tag not in section_text]
            if not missing_tags:
                continue
            run_metrics.increment("analysis.coverage.repairs")
            context_sentences = extract_context_sentences(section_contexts.get(section["id"], ""), limit=6)
            repair_lines = ["## Coverage additions"]
            for tag in missing_tags[:3]:
                supporting_sentence = next(
                    (sentence for sentence in context_sentences if any(token in sentence.lower() for token in tag.split())),
                    context_sentences[0] if context_sentences else f"{tag} requires explicit coverage.",
                )
                repair_lines.append(f"- {tag}: {supporting_sentence}")
            target_slide = slides_for_section[-1]
            target_slide["content_md"] = f"{target_slide['content_md']}\n" + "\n".join(repair_lines)
        repaired_plan = {"thesis": slide_plan["thesis"], "slides": repaired_slides}
        validate_slide_plan_coverage(repaired_plan, section_plan)
        return repaired_plan


def format_anchor_points(anchor_points: list[str]) -> str:
    if not anchor_points:
        return "- No anchor points found."
    return "\n".join(f"- {point}" for point in anchor_points)


def build_placeholder_slide_plan(document_text: str, visuals: list[VisualAsset]) -> dict:
    tokens = document_text.split()
    segments = []
    if tokens:
        window = min(60, len(tokens))
        for start in (0, max(0, len(tokens) // 2 - window // 2), max(0, len(tokens) - window)):
            segment = " ".join(tokens[start : start + window]).strip()
            if segment and segment not in segments:
                segments.append(segment)
    return {
        "thesis": "Fallback deck generated from deterministic PDF extraction only.",
        "slides": [
            {
                "title": "Document Overview",
                "layout": "text_only",
                "visual": "text only",
                "content_md": "\n".join(
                    [
                        "## What happened",
                        "- Source PDF was processed locally.",
                        f"- {len(visuals)} candidate visual assets were extracted.",
                        f"- Opening text: {segments[0] if segments else 'N/A'}",
                    ]
                ),
                "section_id": "fallback",
            },
            {
                "title": "Visual Inventory",
                "layout": "text_image_right",
                "visual": "extracted figure",
                "content_md": "\n".join(
                    [
                        "## Available assets",
                        f"- Candidate visuals kept: {len(visuals)}",
                        "- Visual matching still needs semantic validation.",
                    ]
                ),
                "section_id": "fallback",
            },
        ],
    }


def build_planning_context(document_text: str, max_chars: int) -> str:
    normalized = " ".join(document_text.split())
    if not normalized:
        return ""

    segment_size = max(600, max_chars // 3)
    head = normalized[:segment_size].strip()
    middle_start = max(0, len(normalized) // 2 - segment_size // 2)
    middle = normalized[middle_start : middle_start + segment_size].strip()
    tail = normalized[max(0, len(normalized) - segment_size) :].strip()

    segments = []
    for label, segment in (("BEGINNING", head), ("MIDDLE", middle), ("END", tail)):
        if segment and segment not in segments:
            segments.append(f"{label}:\n{segment}")
    return "\n\n".join(segments)


def _run_json_generation(prompt: str, metric_prefix: str) -> dict:
    run_metrics.increment(f"{metric_prefix}.calls")
    if USE_LOCAL_LLM:
        response_text = _generate_text_with_local_llm(prompt)
    else:
        response_text = generate_text(prompt)
    payload = _extract_json_payload(response_text)
    run_metrics.increment(f"{metric_prefix}.success")
    return payload


def _generate_text_with_local_llm(prompt: str) -> str:
    llm = _get_local_llm()
    run_metrics.increment("local_llm.inference.calls")
    with run_metrics.timed("local_llm.inference.seconds"):
        response = llm(
            prompt,
            max_tokens=int(os.getenv("LOCAL_LLM_MAX_TOKENS", "2200")),
            temperature=float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("LOCAL_LLM_TOP_P", "0.9")),
            stop=["<|endoftext|>", "```", "\n\nHuman:"],
        )
    return response["choices"][0]["text"].strip()


def _get_local_llm():
    global _LOCAL_LLM
    if _LOCAL_LLM is not None:
        run_metrics.increment("local_llm.reused")
        return _LOCAL_LLM

    if not LOCAL_LLM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Local model not found: {LOCAL_LLM_MODEL_PATH}")

    from llama_cpp import Llama

    run_metrics.increment("local_llm.loads")
    with run_metrics.timed("local_llm.load.seconds"):
        _LOCAL_LLM = Llama(
            model_path=str(LOCAL_LLM_MODEL_PATH),
            n_ctx=int(os.getenv("LOCAL_LLM_CTX", "8192")),
            n_gpu_layers=int(os.getenv("LOCAL_LLM_GPU_LAYERS", "-1")),
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
    return _LOCAL_LLM


def _extract_json_payload(raw_text: str) -> dict:
    stripped = raw_text.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("Model output is not valid JSON.")
