"""
Prompt builder for multi-agent chat orchestration.

Mirrors lib/orchestration/prompt-builder.ts  and lib/orchestration/director-prompt.ts.

The TS version is 890 lines; this Python port keeps all prompts that affect
LLM behavior byte-for-byte identical (director rules, role guidelines,
length/whiteboard guidelines, format examples, state/whiteboard context)
while using Python-idiomatic f-strings for interpolation.
"""

from __future__ import annotations

import re
from typing import Any

from core.orchestration.agent_registry import (
    AgentConfig,
    get_action_descriptions,
    get_effective_actions,
)


# ---------------------------------------------------------------------------
# Role guidelines (mirrors ROLE_GUIDELINES in prompt-builder.ts)
# ---------------------------------------------------------------------------

_ROLE_GUIDELINES: dict[str, str] = {
    "teacher": """Your role in this classroom: LEAD TEACHER.
You are responsible for:
- Controlling the lesson flow, slides, and pacing
- Explaining concepts clearly with examples and analogies
- Asking questions to check understanding
- Using spotlight/laser to direct attention to slide elements
- Using the whiteboard for diagrams and formulas
You can use all available actions. Never announce your actions — just teach naturally.""",

    "assistant": """Your role in this classroom: TEACHING ASSISTANT.
You are responsible for:
- Supporting the lead teacher by filling gaps and answering side questions
- Rephrasing explanations in simpler terms when students are confused
- Providing concrete examples and background context
- Using the whiteboard sparingly to supplement (not duplicate) the teacher's content
You play a supporting role — don't take over the lesson.""",

    "student": """Your role in this classroom: STUDENT.
You are responsible for:
- Participating actively in discussions
- Asking questions, sharing observations, reacting to the lesson
- Keeping responses SHORT (1-2 sentences max)
- Only using the whiteboard when explicitly invited by the teacher
You are NOT a teacher — your responses should be much shorter than the teacher's.""",
}


# ---------------------------------------------------------------------------
# Length guidelines (mirrors buildLengthGuidelines)
# ---------------------------------------------------------------------------

_LENGTH_COMMON = (
    '- Length targets count ONLY your speech text (type:"text" content). '
    "Actions (spotlight, whiteboard, etc.) do NOT count toward length. "
    'Use as many actions as needed — they don\'t make your speech "too long."\n'
    "- Speak conversationally and naturally — this is a live classroom, "
    "not a textbook. Use oral language, not written prose."
)


def _build_length_guidelines(role: str) -> str:
    if role == "teacher":
        return (
            "- Keep your TOTAL speech text around 100 characters (across all text objects combined). "
            "Prefer 2-3 short sentences over one long paragraph.\n"
            f"{_LENGTH_COMMON}\n"
            "- Prioritize inspiring students to THINK over explaining everything yourself. "
            "Ask questions, pose challenges, give hints — don't just lecture.\n"
            "- When explaining, give the key insight in one crisp sentence, then pause or ask a question. "
            "Avoid exhaustive explanations."
        )
    if role == "assistant":
        return (
            "- Keep your TOTAL speech text around 80 characters. "
            "You are a supporting role — be brief.\n"
            f"{_LENGTH_COMMON}\n"
            "- One key point per response. Don't repeat the teacher's full explanation — "
            "add a quick angle, example, or summary."
        )
    # student (default)
    return (
        "- Keep your TOTAL speech text around 50 characters. 1-2 sentences max.\n"
        f"{_LENGTH_COMMON}\n"
        "- You are a STUDENT, not a teacher. Your responses should be much shorter than the teacher's. "
        "If your response is as long as the teacher's, you are doing it wrong.\n"
        "- Speak in quick, natural reactions: a question, a joke, a brief insight, a short observation. "
        "Not paragraphs.\n"
        "- Inspire and provoke thought with punchy comments, not lengthy analysis."
    )


# ---------------------------------------------------------------------------
# Whiteboard guidelines (mirrors buildWhiteboardGuidelines)
# ---------------------------------------------------------------------------

_WB_COMMON = (
    '- Before drawing on the whiteboard, check the "Current State" section below for existing whiteboard elements.\n'
    "- Do NOT redraw content that already exists — if a formula, chart, concept, or table is already on "
    "the whiteboard, reference it instead of duplicating it.\n"
    '- When adding new elements, calculate positions carefully: check existing elements\' coordinates '
    "and sizes in the whiteboard state, and ensure at least 20px gap between elements. "
    "Canvas size is 1000×562. All elements MUST stay within the canvas boundaries — ensure x >= 0, "
    "y >= 0, x + width <= 1000, and y + height <= 562. Never place elements that extend beyond the edges.\n"
    "- If another agent has already drawn related content, build upon or extend it rather than starting from scratch."
)

_LATEX_GUIDELINES = """
### LaTeX Element Sizing (CRITICAL)
LaTeX elements have **auto-calculated width** (width = height × aspectRatio). You control **height**, and the system computes the width to preserve the formula's natural proportions. The height you specify is the ACTUAL rendered height — use it to plan vertical layout.

**Height guide by formula category:**
| Category | Examples | Recommended height |
|----------|---------|-------------------|
| Inline equations | E=mc^2, a+b=c | 50-80 |
| Equations with fractions | \\frac{-b±√(b²-4ac)}{2a} | 60-100 |
| Integrals / limits | \\int_0^1 f(x)dx, \\lim_{x→0} | 60-100 |
| Summations with limits | \\sum_{i=1}^{n} i^2 | 80-120 |
| Matrices | \\begin{pmatrix}...\\end{pmatrix} | 100-180 |
| Standalone fractions | \\frac{a}{b}, \\frac{1}{2} | 50-80 |
| Nested fractions | \\frac{\\frac{a}{b}}{\\frac{c}{d}} | 80-120 |

**Key rules:**
- ALWAYS specify height. The height you set is the actual rendered height.
- When placing elements below each other, add height + 20-40px gap.
- Width is auto-computed — long formulas expand horizontally, short ones stay narrow.
- If a formula's auto-computed width exceeds the whiteboard, reduce height.

**Multi-step derivations:**
Give each step the **same height** (e.g., 70-80px). The system auto-computes width proportionally — all steps render at the same vertical size.

### LaTeX Support
This project uses KaTeX for formula rendering, which supports virtually all standard LaTeX math commands. You may use any standard LaTeX math command freely.

- \\text{} can render English text. For non-Latin labels, use a separate TextElement."""


def _build_whiteboard_guidelines(role: str) -> str:
    if role == "teacher":
        return f"""- Use text elements for notes, steps, and explanations.
- Use chart elements for data visualization (bar charts, line graphs, pie charts, etc.).
- Use latex elements for mathematical formulas and scientific equations.
- Use table elements for structured data, comparisons, and organized information.
- Use code elements for demonstrating code, algorithms, and programming concepts. Code blocks have syntax highlighting and support line-by-line editing.
- Use shape elements sparingly — only for simple diagrams. Do not add large numbers of meaningless shapes.
- Use line elements to connect related elements, draw arrows showing relationships, or annotate diagrams. Specify arrow markers via the points parameter.
- If the whiteboard is too crowded, call wb_clear to wipe it clean before adding new elements.

### Deleting Elements
- Use wb_delete to remove a specific element by its ID (shown as [id:xxx] in whiteboard state).
- Prefer wb_delete over wb_clear when only 1-2 elements need removal.
- Common use cases: removing an outdated formula before writing the corrected version, clearing a step after explaining it to make room for the next step.

### Animation-Like Effects with Delete + Draw
All wb_draw_* actions accept an optional **elementId** parameter. When you specify elementId, you can later use wb_delete with that same ID to remove the element. This is essential for creating animation effects.
- To use: add elementId (e.g. "step1", "box_a") when drawing, then wb_delete with that elementId to remove it later.
- Step-by-step reveal: Draw step 1 (elementId:"step1") → speak → delete "step1" → draw step 2 (elementId:"step2") → speak → ...
- State transitions: Draw initial state (elementId:"state") → explain → delete "state" → draw final state
- Progressive diagrams: Draw base diagram → add elements one by one with speech between each
- Example: draw a shape at position A with elementId "obj", explain it, delete "obj", draw the same shape at position B — this creates the illusion of movement.
- Combine wb_delete (by element ID) with wb_draw_* actions to update specific parts without clearing everything.

### Layout Constraints (IMPORTANT)
The whiteboard canvas is 1000 × 562 pixels. Follow these rules to prevent element overlap:

**Coordinate system:**
- X range: 0 (left) to 1000 (right), Y range: 0 (top) to 562 (bottom)
- Leave 20px margin from edges (safe area: x 20-980, y 20-542)

**Spacing rules:**
- Maintain at least 20px gap between adjacent elements
- Vertical stacking: next_y = previous_y + previous_height + 30
- Side by side: next_x = previous_x + previous_width + 30

**Layout patterns:**
- Top-down flow: Start from y=30, stack downward with gaps
- Two-column: Left column x=20-480, right column x=520-980
- Center single element: x = (1000 - element_width) / 2

**Before adding a new element:**
- Check existing elements' positions in the whiteboard state
- Ensure your new element's bounding box does not overlap with any existing element
- If space is insufficient, use wb_delete to remove unneeded elements or wb_clear to start fresh

### Code Element Layout & Usage
- Code blocks have a **header bar (~32px)** showing the file name and language. The actual code content starts below the header. When calculating vertical space, account for this overhead: effective code area height ≈ element height - 32px.
- Each code line is ~22px tall (at default fontSize 14). Plan height accordingly: a 10-line code block needs about height = 32 (header) + 10 × 22 (lines) + 16 (padding) ≈ 270px.
- Use **wb_edit_code** for step-by-step code demonstrations: draw a skeleton first, then incrementally insert/modify lines with speech between each edit. This creates a "live coding" effect.
- When editing code, reference lines by their stable IDs (L1, L2, ...) shown in the whiteboard state. Do NOT guess line IDs — always check the current whiteboard state first.
{_LATEX_GUIDELINES}
{_WB_COMMON}"""

    if role == "assistant":
        return f"""- The whiteboard is primarily the teacher's space. As an assistant, use it sparingly to supplement.
- If the teacher has already set up content on the whiteboard (exercises, formulas, tables), do NOT add parallel derivations or extra formulas — explain verbally instead.
- Only draw on the whiteboard to clarify something the teacher missed, or to add a brief supplementary note that won't clutter the board.
- Limit yourself to at most 1-2 small elements per response. Prefer speech over drawing.
{_LATEX_GUIDELINES}
{_WB_COMMON}"""

    # student
    return f"""- The whiteboard is primarily the teacher's space. Do NOT draw on it proactively.
- Only use whiteboard actions when the teacher or user explicitly invites you to write on the board (e.g., "come solve this", "show your work on the whiteboard").
- If no one asked you to use the whiteboard, express your ideas through speech only.
- When you ARE invited to use the whiteboard, keep it minimal and tidy — add only what was asked for.
{_WB_COMMON}"""


# ---------------------------------------------------------------------------
# Scene element summarization (mirrors summarizeElement / summarizeElements)
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    if not html:
        return ""
    return re.sub(r"<[^>]*>", "", html).strip()


def _summarize_element(el: dict[str, Any]) -> str:
    _id = f"[id:{el.get('id')}]" if el.get("id") else ""
    left = el.get("left") or 0
    top = el.get("top") or 0
    pos = f"at ({round(left)},{round(top)})"
    width = el.get("width")
    height = el.get("height")
    if width is not None and height is not None:
        size = f" size {round(width)}×{round(height)}"
    elif width is not None:
        size = f" w={round(width)}"
    else:
        size = ""

    etype = el.get("type")
    if etype == "text":
        text = _strip_html(el.get("content", ""))[:60]
        suffix = "..." if len(_strip_html(el.get("content", ""))) > 60 else ""
        text_type = f"[{el['textType']}]" if el.get("textType") else ""
        return f'{_id} text{text_type}: "{text}{suffix}" {pos}{size}'
    if etype == "image":
        src_raw = el.get("src") or "unknown"
        src = "[embedded]" if isinstance(src_raw, str) and src_raw.startswith("data:") else str(src_raw)[:50]
        return f"{_id} image: {src} {pos}{size}"
    if etype == "shape":
        shape_text = ""
        if isinstance(el.get("text"), dict) and el["text"].get("content"):
            shape_text = f': "{_strip_html(el["text"]["content"])[:40]}"'
        return f"{_id} shape{shape_text} {pos}{size}"
    if etype == "chart":
        labels = (el.get("data") or {}).get("labels") or []
        chart_type = el.get("chartType", "bar")
        return f"{_id} chart[{chart_type}]: labels=[{','.join(str(x) for x in labels[:4])}] {pos}{size}"
    if etype == "table":
        data = el.get("data") or []
        rows = len(data)
        cols = len(data[0]) if rows and isinstance(data[0], list) else 0
        return f"{_id} table: {rows}x{cols} {pos}{size}"
    if etype == "latex":
        return f'{_id} latex: "{str(el.get("latex", ""))[:40]}" {pos}{size}'
    if etype == "line":
        lx = round(el.get("left") or 0)
        ly = round(el.get("top") or 0)
        start = el.get("start") or [0, 0]
        end = el.get("end") or [0, 0]
        return f"{_id} line: ({lx + (start[0] or 0)},{ly + (start[1] or 0)}) → ({lx + (end[0] or 0)},{ly + (end[1] or 0)})"
    if etype == "code":
        lang = el.get("language", "unknown")
        lines = el.get("lines") or []
        line_count = len(lines)
        code_fn = f' "{el["fileName"]}"' if el.get("fileName") else ""
        line_preview = "\n".join(
            f"    {ln.get('id')}: {ln.get('content')}" for ln in lines[:10]
        )
        more = f"\n    ... and {line_count - 10} more lines" if line_count > 10 else ""
        return f"{_id} code{code_fn} ({lang}, {line_count} lines) {pos}{size}\n{line_preview}{more}"
    if etype == "video":
        return f"{_id} video {pos}{size}"
    if etype == "audio":
        return f"{_id} audio {pos}{size}"
    return f"{_id} {etype or 'unknown'} {pos}{size}"


def _summarize_elements(elements: list[dict]) -> str:
    if not elements:
        return "  (empty)"
    return "\n".join(f"  {i + 1}. {_summarize_element(el)}" for i, el in enumerate(elements))


# ---------------------------------------------------------------------------
# State context (mirrors buildStateContext)
# ---------------------------------------------------------------------------

def _build_state_context(store_state: dict) -> str:
    stage = store_state.get("stage") or {}
    scenes: list[dict] = store_state.get("scenes") or []
    current_scene_id = store_state.get("currentSceneId") or store_state.get("current_scene_id")
    mode = store_state.get("mode", "autonomous")
    wb_open = store_state.get("whiteboardOpen", False)

    lines: list[str] = [
        f"Mode: {mode}",
        f"Whiteboard: {'OPEN (slide canvas is hidden)' if wb_open else 'closed (slide canvas is visible)'}",
    ]

    if stage:
        name = stage.get("name") or "Untitled"
        desc = stage.get("description") or ""
        lines.append(f"Course: {name}{f' - {desc}' if desc else ''}")

    lines.append(f"Total scenes: {len(scenes)}")

    if current_scene_id:
        current = next((s for s in scenes if s.get("id") == current_scene_id), None)
        if current:
            ctitle = current.get("title", "")
            ctype = current.get("type") or current.get("scene_type", "")
            lines.append(f'Current scene: "{ctitle}" ({ctype}, id: {current_scene_id})')

            content = current.get("content") or {}
            if content.get("type") == "slide":
                canvas = content.get("canvas") or {}
                elements = canvas.get("elements") or []
                lines.append(
                    f"Current slide elements ({len(elements)}):\n{_summarize_elements(elements)}"
                )
            elif content.get("type") == "quiz":
                questions: list[dict] = content.get("questions") or []
                q_summary_lines = [
                    f"  {i + 1}. [{q.get('type', '?')}] {(q.get('question') or '')[:80]}"
                    for i, q in enumerate(questions[:5])
                ]
                more = (
                    f"\n  ... and {len(questions) - 5} more"
                    if len(questions) > 5 else ""
                )
                lines.append(
                    f"Quiz questions ({len(questions)}):\n{chr(10).join(q_summary_lines)}{more}"
                )
    elif scenes:
        lines.append("No scene currently selected")

    if scenes:
        scene_summary_lines = [
            f"  {i + 1}. {s.get('title', '')} ({s.get('type', '')}, id: {s.get('id', '')})"
            for i, s in enumerate(scenes[:5])
        ]
        more = f"\n  ... and {len(scenes) - 5} more" if len(scenes) > 5 else ""
        lines.append(f"Scenes:\n{chr(10).join(scene_summary_lines)}{more}")

    # Whiteboard content (last whiteboard in the stage)
    whiteboards = stage.get("whiteboard") or [] if isinstance(stage, dict) else []
    if whiteboards:
        last_wb = whiteboards[-1]
        wb_elements = last_wb.get("elements") or []
        lines.append(
            f"Whiteboard (last of {len(whiteboards)}, {len(wb_elements)} elements):\n"
            f"{_summarize_elements(wb_elements)}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Virtual whiteboard context (mirrors buildVirtualWhiteboardContext)
# ---------------------------------------------------------------------------

def _build_virtual_whiteboard_context(
    store_state: dict, ledger: list[dict] | None = None
) -> str:
    if not ledger:
        return ""

    elements: list[dict[str, str]] = []

    for record in ledger:
        name = record.get("actionName", "")
        params = record.get("params") or {}
        agent_name = record.get("agentName", "unknown")

        if name == "wb_clear":
            elements.clear()
            continue
        if name == "wb_delete":
            delete_id = str(params.get("elementId", ""))
            elements = [e for e in elements if e.get("elementId") != delete_id]
            continue

        if name == "wb_draw_text":
            content = str(params.get("content", ""))[:40]
            x = params.get("x", "?")
            y = params.get("y", "?")
            w = params.get("width", 400)
            h = params.get("height", 100)
            ellipsis = "..." if len(str(params.get("content", ""))) >= 40 else ""
            summary = f'text: "{content}{ellipsis}" at ({x},{y}), size ~{w}x{h}'
        elif name == "wb_draw_shape":
            shape = params.get("type") or params.get("shape") or "rectangle"
            summary = (
                f"shape({shape}) at ({params.get('x', '?')},{params.get('y', '?')}), "
                f"size {params.get('width', 100)}x{params.get('height', 100)}"
            )
        elif name == "wb_draw_chart":
            ctype = params.get("chartType") or params.get("type") or "bar"
            labels = params.get("labels")
            if labels is None and isinstance(params.get("data"), dict):
                labels = params["data"].get("labels")
            label_part = (
                f": labels=[{','.join(str(x) for x in (labels or [])[:4])}]"
                if labels else ""
            )
            summary = (
                f"chart({ctype}){label_part} at ({params.get('x', '?')},{params.get('y', '?')}), "
                f"size {params.get('width', 350)}x{params.get('height', 250)}"
            )
        elif name == "wb_draw_latex":
            latex = str(params.get("latex", ""))[:40]
            ellipsis = "..." if len(str(params.get("latex", ""))) >= 40 else ""
            summary = (
                f'latex: "{latex}{ellipsis}" at ({params.get("x", "?")},{params.get("y", "?")}), '
                f'size ~{params.get("width", 400)}x{params.get("height", 80)}'
            )
        elif name == "wb_draw_table":
            data = params.get("data") or []
            rows = len(data) if isinstance(data, list) else 0
            cols = len(data[0]) if rows and isinstance(data[0], list) else 0
            summary = (
                f"table({rows}×{cols}) at ({params.get('x', '?')},{params.get('y', '?')}), "
                f"size {params.get('width', 400)}x{params.get('height', rows * 40 + 20)}"
            )
        elif name == "wb_draw_line":
            pts = params.get("points") or []
            arrow = " (arrow)" if "arrow" in pts else ""
            summary = (
                f"line{arrow}: ({params.get('startX', '?')},{params.get('startY', '?')}) "
                f"→ ({params.get('endX', '?')},{params.get('endY', '?')})"
            )
        elif name == "wb_draw_code":
            lang = str(params.get("language", ""))
            code_fn = f' "{params["fileName"]}"' if params.get("fileName") else ""
            code = str(params.get("code", ""))
            line_count = len(code.split("\n"))
            summary = (
                f"code block{code_fn} ({lang}, {line_count} lines) "
                f"at ({params.get('x', '?')},{params.get('y', '?')}), "
                f"size {params.get('width', 500)}x{params.get('height', 300)}"
            )
        elif name == "wb_edit_code":
            op = params.get("operation", "edit")
            target = params.get("elementId", "?")
            summary = f'edited code "{target}" ({op})'
        else:
            continue  # skip wb_open, wb_close

        elements.append({"agentName": agent_name, "summary": summary})

    if not elements:
        return ""

    lines = [
        f"  {i + 1}. [by {el['agentName']}] {el['summary']}"
        for i, el in enumerate(elements)
    ]
    return f"""
## Whiteboard Changes This Round (IMPORTANT)
Other agents have modified the whiteboard during this discussion round.
Current whiteboard elements ({len(elements)}):
{chr(10).join(lines)}

DO NOT redraw content that already exists. Check positions above before adding new elements.
"""


# ---------------------------------------------------------------------------
# Peer context (mirrors buildPeerContextSection)
# ---------------------------------------------------------------------------

def _build_peer_context(
    agent_responses: list[dict] | None,
    current_agent_name: str,
) -> str:
    if not agent_responses:
        return ""
    peers = [
        r for r in agent_responses
        if (r.get("agentName") or r.get("agent_name")) != current_agent_name
    ]
    if not peers:
        return ""

    peer_lines = "\n".join(
        f'- {r.get("agentName") or r.get("agent_name")}: '
        f'"{r.get("contentPreview") or r.get("content_preview") or ""}"'
        for r in peers
    )
    return f"""
# This Round's Context (CRITICAL — READ BEFORE RESPONDING)
The following agents have already spoken in this discussion round:
{peer_lines}

You are {current_agent_name}, responding AFTER the agents above. You MUST:
1. NOT repeat greetings or introductions — they have already been made
2. NOT restate what previous speakers already explained
3. Add NEW value from YOUR unique perspective as {current_agent_name}
4. Build on, question, or extend what was said — do not echo it
5. If you agree with a previous point, say so briefly and then ADD something new
"""


# ---------------------------------------------------------------------------
# Structured prompt (mirrors buildStructuredPrompt)
# ---------------------------------------------------------------------------

def build_structured_prompt(
    agent: AgentConfig,
    store_state: dict,
    discussion_context: dict | None = None,
    whiteboard_ledger: list[dict] | None = None,
    user_profile: dict | None = None,
    agent_responses: list[dict] | None = None,
    latest_user_question: str | None = None,
) -> str:
    """Build the full system prompt for one agent turn.

    Output format is a JSON array of `{type: "action"|"text"}` objects.

    When ``latest_user_question`` is provided, a high-priority
    "User's Current Question" block is appended so the model answers
    the user directly instead of continuing to lecture from the scene
    context. This addresses the common "answers off-topic" failure
    where the LLM keeps teaching the slide content and ignores a
    direct Q&A question from the student.
    """
    # Determine current scene type for action filtering
    scenes: list[dict] = store_state.get("scenes") or []
    current_scene_id = store_state.get("currentSceneId") or store_state.get("current_scene_id")
    current_scene = next((s for s in scenes if s.get("id") == current_scene_id), None)
    scene_type = None
    if current_scene:
        scene_type = current_scene.get("type") or current_scene.get("scene_type")

    effective_actions = get_effective_actions(agent.allowed_actions, scene_type)
    action_descriptions = get_action_descriptions(effective_actions)

    state_context = _build_state_context(store_state)
    virtual_wb_context = _build_virtual_whiteboard_context(store_state, whiteboard_ledger)

    student_profile_section = ""
    if user_profile and (user_profile.get("nickname") or user_profile.get("bio")):
        nickname = user_profile.get("nickname") or "a student"
        bio = user_profile.get("bio")
        student_profile_section = (
            f"\n# Student Profile\n"
            f"You are teaching {nickname}."
            f"{f' Their background: {bio}' if bio else ''}\n"
            "Personalize your teaching based on their background when relevant. "
            "Address them by name naturally.\n"
        )

    peer_context = _build_peer_context(agent_responses, agent.name)

    has_slide_actions = "spotlight" in effective_actions or "laser" in effective_actions

    format_example = (
        '[{"type":"action","name":"spotlight","params":{"elementId":"img_1"}},'
        '{"type":"text","content":"Your natural speech to students"}]'
        if has_slide_actions
        else '[{"type":"action","name":"wb_open","params":{}},'
             '{"type":"text","content":"Your natural speech to students"}]'
    )

    ordering_principles = (
        "- spotlight/laser actions should appear BEFORE the corresponding text object "
        "(point first, then speak)\n"
        "- whiteboard actions can interleave WITH text objects (draw while speaking)"
        if has_slide_actions
        else "- whiteboard actions can interleave WITH text objects (draw while speaking)"
    )

    spotlight_examples = ""
    if has_slide_actions:
        spotlight_examples = (
            '[{"type":"action","name":"spotlight","params":{"elementId":"img_1"}},'
            '{"type":"text","content":"Photosynthesis is the process by which plants convert '
            'light energy into chemical energy. Take a look at this diagram."},'
            '{"type":"text","content":"During this process, plants absorb carbon dioxide '
            'and water to produce glucose and oxygen."}]\n\n'
            '[{"type":"action","name":"spotlight","params":{"elementId":"eq_1"}},'
            '{"type":"action","name":"laser","params":{"elementId":"eq_2"}},'
            '{"type":"text","content":"Compare these two equations — notice how the left side '
            'is endothermic while the right side is exothermic."}]\n\n'
        )

    slide_action_guidelines = ""
    if has_slide_actions:
        slide_action_guidelines = (
            "- spotlight: Use to focus attention on ONE key element. Don't overuse — max 1-2 per response.\n"
            "- laser: Use to point at elements. Good for directing attention during explanations.\n"
        )

    mutual_exclusion_note = ""
    if has_slide_actions:
        mutual_exclusion_note = (
            "- IMPORTANT — Whiteboard / Canvas mutual exclusion: The whiteboard and slide canvas "
            "are mutually exclusive. When the whiteboard is OPEN, the slide canvas is hidden — "
            "spotlight and laser actions targeting slide elements will have NO visible effect. "
            "If you need to use spotlight or laser, call wb_close first to reveal the slide canvas. "
            "Conversely, if the whiteboard is CLOSED, wb_draw_* actions still work (they implicitly "
            "open the whiteboard), but be aware that doing so hides the slide canvas.\n"
            "- Prefer variety: mix spotlights, laser, and whiteboard for engaging teaching. "
            "Don't use the same action type repeatedly."
        )

    role_guideline = _ROLE_GUIDELINES.get(agent.role, _ROLE_GUIDELINES["student"])
    length_guidelines = _build_length_guidelines(agent.role)
    whiteboard_guidelines = _build_whiteboard_guidelines(agent.role)

    lang_directive = ((store_state.get("stage") or {}) or {}).get("languageDirective")
    language_constraint = f"\n# Language (CRITICAL)\n{lang_directive}\n" if lang_directive else ""

    user_question_section = ""
    if latest_user_question and latest_user_question.strip():
        q = latest_user_question.strip()
        if len(q) > 500:
            q = q[:500] + "…"

        # Heuristic language detection from the question itself. This is
        # the backstop when stage.languageDirective is missing (e.g. the
        # classroom was generated before we started persisting directives,
        # or it was generated by a flow that didn't populate it). Without
        # this, a Chinese question can still be answered in English when
        # the user_profile.bio mentions something like "学习英语".
        has_cjk = any("\u4e00" <= c <= "\u9fff" for c in q)
        has_hangul = any("\uac00" <= c <= "\ud7af" for c in q)
        has_kana = any(
            "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" for c in q
        )
        has_cyrillic = any("\u0400" <= c <= "\u04ff" for c in q)

        if has_cjk:
            question_language_rule = (
                "The question is in Chinese — you MUST answer entirely in Simplified Chinese. "
                "Do NOT switch to English for any reason, even if the user profile mentions "
                "learning English. Match the user's language, not their long-term learning goals."
            )
        elif has_hangul:
            question_language_rule = (
                "The question is in Korean — answer in Korean."
            )
        elif has_kana:
            question_language_rule = (
                "The question is in Japanese — answer in Japanese."
            )
        elif has_cyrillic:
            question_language_rule = (
                "The question is in a Cyrillic-script language — answer in the same language."
            )
        else:
            question_language_rule = (
                "Answer in the SAME language the user wrote the question in. "
                "Do not switch languages."
            )

        user_question_section = f"""

# User's Current Question (HIGHEST PRIORITY — READ THIS FIRST)
The student just asked you directly:

> {q}

You MUST answer THIS question before anything else. Rules:
1. Do NOT keep lecturing from the current slide if the question is unrelated — pause the lesson and answer the student.
2. If the question relates to the current slide, tie your answer back to what's on screen.
3. If the question is off-topic from the scene, briefly answer it in your own knowledge, then offer to return to the lesson.
4. Never ignore or brush off the student's question. Answering it is more important than covering the slide.
5. Do NOT deliver a classroom-intro style welcome speech ("Welcome Tiger! …"). This is a direct Q&A reply, not a lesson opening.
6. Keep the JSON-array output format; your speech text should directly address the question.
7. Language rule (OVERRIDES everything else, including the student profile's learning goals): {question_language_rule}"""

    discussion_section = ""
    if discussion_context:
        topic = discussion_context.get("topic", "")
        prompt = discussion_context.get("prompt", "")
        if agent_responses:
            discussion_section = f"""

# Discussion Context
Topic: "{topic}"
{f'Guiding prompt: {prompt}' if prompt else ''}

You are JOINING an ongoing discussion — do NOT re-introduce the topic or greet the students. The discussion has already started. Contribute your unique perspective, ask a follow-up question, or challenge an assumption made by a previous speaker."""
        else:
            discussion_section = f"""

# Discussion Context
You are initiating a discussion on the following topic: "{topic}"
{f'Guiding prompt: {prompt}' if prompt else ''}

IMPORTANT: As you are starting this discussion, begin by introducing the topic naturally to the students. Engage them and invite their thoughts. Do not wait for user input - you speak first."""

    return f"""# Role
You are {agent.name}.

## Your Personality
{agent.persona}

## Your Classroom Role
{role_guideline}
{student_profile_section}{peer_context}{language_constraint}
# Output Format
You MUST output a JSON array for ALL responses. Each element is an object with a `type` field:

{format_example}

## Format Rules
1. Output a single JSON array — no explanation, no code fences
2. `type:"action"` objects contain `name` and `params`
3. `type:"text"` objects contain `content` (speech text)
4. Action and text objects can freely interleave in any order
5. The `]` closing bracket marks the end of your response
6. CRITICAL: ALWAYS start your response with `[` — even if your previous message was interrupted. Never continue a partial response as plain text. Every response must be a complete, independent JSON array.

## Ordering Principles
{ordering_principles}

## Speech Guidelines (CRITICAL)
- Effects fire concurrently with your speech — students see results as you speak
- Text content is what you SAY OUT LOUD to students - natural teaching speech
- Do NOT say "let me add...", "I'll create...", "now I'm going to..."
- Do NOT describe your actions - just speak naturally as a teacher
- Students see action results appear on screen - you don't need to announce them
- Your speech should flow naturally regardless of whether actions succeed or fail
- NEVER use markdown formatting (blockquotes >, headings #, bold **, lists -, code blocks) in text content — it is spoken aloud, not rendered

## Length & Style (CRITICAL)
{length_guidelines}

### Good Examples
{spotlight_examples}[{{"type":"action","name":"wb_open","params":{{}}}},{{"type":"action","name":"wb_draw_text","params":{{"content":"Step 1: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂","x":100,"y":100,"fontSize":24}}}},{{"type":"text","content":"Look at this chemical equation — notice how the reactants and products correspond."}}]

[{{"type":"action","name":"wb_open","params":{{}}}},{{"type":"action","name":"wb_draw_latex","params":{{"latex":"\\\\frac{{-b \\\\pm \\\\sqrt{{b^2-4ac}}}}{{2a}}","x":100,"y":80,"width":500}}}},{{"type":"text","content":"This is the quadratic formula — it can solve any quadratic equation."}},{{"type":"action","name":"wb_draw_table","params":{{"x":100,"y":250,"width":500,"height":150,"data":[["Variable","Meaning"],["a","Coefficient of x²"],["b","Coefficient of x"],["c","Constant term"]]}}}},{{"type":"text","content":"Each variable's meaning is shown in the table."}}]

### Bad Examples (DO NOT do this)
[{{"type":"text","content":"Let me open the whiteboard"}},{{"type":"action",...}}] (Don't announce actions!)
[{{"type":"text","content":"I'm going to draw a diagram for you..."}}] (Don't describe what you're doing!)
[{{"type":"text","content":"Action complete, shape has been added"}}] (Don't report action results!)

## Whiteboard Guidelines
{whiteboard_guidelines}

# Available Actions
{action_descriptions}

## Action Usage Guidelines
{slide_action_guidelines}- Whiteboard actions (wb_open, wb_draw_text, wb_draw_shape, wb_draw_chart, wb_draw_latex, wb_draw_table, wb_draw_line, wb_draw_code, wb_edit_code, wb_delete, wb_clear, wb_close): Use when explaining concepts that benefit from diagrams, formulas, data charts, tables, connecting lines, code demonstrations, or step-by-step derivations. Use wb_draw_latex for math formulas, wb_draw_chart for data visualization, wb_draw_table for structured data, wb_draw_code for code demonstrations.
- WHITEBOARD CLOSE RULE (CRITICAL): Do NOT call wb_close at the end of your response. Leave the whiteboard OPEN so students can read what you drew. Only call wb_close when you specifically need to return to the slide canvas (e.g., to use spotlight or laser on slide elements). Frequent open/close is distracting.
- wb_delete: Use to remove a specific element by its ID (shown in brackets like [id:xxx] in the whiteboard state). Prefer this over wb_clear when only one or a few elements need to be removed.
- wb_draw_code / wb_edit_code: To modify an existing code block, ALWAYS use wb_edit_code (insert_after, insert_before, delete_lines, replace_lines) instead of deleting the code element and re-creating it. wb_edit_code produces smooth line-level animations; deleting and re-drawing loses the animation continuity. Only use wb_draw_code for creating a brand-new code block.
{mutual_exclusion_note}

# Current State
{state_context}
{virtual_wb_context}
Remember: Speak naturally as a teacher. Effects fire concurrently with your speech.{discussion_section}{user_question_section}"""


# ---------------------------------------------------------------------------
# Director prompt helpers (mirrors director-prompt.ts)
# ---------------------------------------------------------------------------

def _summarize_agent_whiteboard_actions(actions: list[dict]) -> str:
    parts: list[str] = []
    for a in actions or []:
        name = a.get("actionName", "")
        params = a.get("params") or {}
        if name == "wb_draw_text":
            content = str(params.get("content", ""))[:30]
            ellipsis = "..." if len(str(params.get("content", ""))) >= 30 else ""
            parts.append(f'drew text "{content}{ellipsis}"')
        elif name == "wb_draw_shape":
            parts.append(f"drew shape({params.get('type', 'rectangle')})")
        elif name == "wb_draw_chart":
            labels = params.get("labels")
            if labels is None and isinstance(params.get("data"), dict):
                labels = params["data"].get("labels")
            ctype = params.get("chartType") or params.get("type") or "bar"
            label_part = (
                f", labels: [{','.join(str(x) for x in (labels or [])[:4])}]"
                if labels else ""
            )
            parts.append(f"drew chart({ctype}{label_part})")
        elif name == "wb_draw_latex":
            latex = str(params.get("latex", ""))[:30]
            ellipsis = "..." if len(str(params.get("latex", ""))) >= 30 else ""
            parts.append(f'drew formula "{latex}{ellipsis}"')
        elif name == "wb_draw_table":
            data = params.get("data") or []
            rows = len(data) if isinstance(data, list) else 0
            cols = len(data[0]) if rows and isinstance(data[0], list) else 0
            parts.append(f"drew table({rows}×{cols})")
        elif name == "wb_draw_line":
            pts = params.get("points") or []
            arrow = " arrow" if "arrow" in pts else ""
            parts.append(f"drew{arrow} line")
        elif name == "wb_draw_code":
            lang = str(params.get("language", ""))
            code_fn = f' "{params["fileName"]}"' if params.get("fileName") else ""
            parts.append(f"drew code block{code_fn} ({lang})")
        elif name == "wb_edit_code":
            op = params.get("operation", "edit")
            parts.append(f"edited code ({op})")
        elif name == "wb_clear":
            parts.append("CLEARED whiteboard")
        elif name == "wb_delete":
            parts.append(f'deleted element "{params.get("elementId")}"')
        # wb_open / wb_close: skip (structural)
    return ", ".join(parts)


def _summarize_whiteboard_for_director(
    ledger: list[dict] | None,
) -> tuple[int, list[str]]:
    if not ledger:
        return 0, []
    element_count = 0
    contributors: set[str] = set()
    for record in ledger:
        name = record.get("actionName", "")
        if name == "wb_clear":
            element_count = 0
        elif name == "wb_delete":
            element_count = max(0, element_count - 1)
        elif name.startswith("wb_draw_"):
            element_count += 1
            contributors.add(record.get("agentName", ""))
    return element_count, sorted(contributors)


def _build_whiteboard_state_for_director(ledger: list[dict] | None) -> str:
    if not ledger:
        return ""
    element_count, contributors = _summarize_whiteboard_for_director(ledger)
    crowded = (
        "\n⚠ The whiteboard is getting crowded. Consider routing to an agent that "
        "will organize or clear it rather than adding more."
        if element_count > 5 else ""
    )
    contributors_str = ", ".join(contributors) if contributors else "none"
    return f"""
# Whiteboard State
Elements on whiteboard: {element_count}
Contributors: {contributors_str}{crowded}
"""


def build_director_prompt(
    agents: list[AgentConfig],
    conversation_summary: str,
    agent_responses: list[dict],
    turn_count: int,
    discussion_context: dict | None = None,
    trigger_agent_id: str | None = None,
    whiteboard_ledger: list[dict] | None = None,
    user_profile: dict | None = None,
    whiteboard_open: bool = False,
) -> str:
    """Build the director agent's system prompt.

    Mirrors buildDirectorPrompt() in lib/orchestration/director-prompt.ts.
    The director's response is expected to be a JSON object of the form
    ``{"next_agent": "<agent_id>" | "USER" | "END"}``.
    """
    agent_list = "\n".join(
        f'- id: "{a.id}", name: "{a.name}", role: {a.role}, priority: {a.priority}'
        for a in agents
    )

    if agent_responses:
        responded_lines = []
        for r in agent_responses:
            wb_summary = _summarize_agent_whiteboard_actions(
                r.get("whiteboardActions") or r.get("whiteboard_actions") or []
            )
            wb_part = f" | Whiteboard: {wb_summary}" if wb_summary else ""
            action_count = r.get("actionCount") or r.get("action_count") or 0
            preview = r.get("contentPreview") or r.get("content_preview") or ""
            agent_name = r.get("agentName") or r.get("agent_name") or r.get("agentId") or "?"
            agent_id = r.get("agentId") or r.get("agent_id") or "?"
            responded_lines.append(
                f'- {agent_name} ({agent_id}): "{preview}" [{action_count} actions{wb_part}]'
            )
        responded_list = "\n".join(responded_lines)
    else:
        responded_list = "None yet."

    is_discussion = discussion_context is not None

    if is_discussion:
        topic = discussion_context.get("topic", "")
        prompt_text = discussion_context.get("prompt", "")
        discussion_section = (
            f'\n# Discussion Mode\n'
            f'Topic: "{topic}"'
            f'{f" Prompt: \"{prompt_text}\"" if prompt_text else ""}'
            f'{f" Initiator: \"{trigger_agent_id}\"" if trigger_agent_id else ""}\n'
            "This is a student-initiated discussion, not a Q&A session.\n"
        )
    else:
        discussion_section = ""

    if is_discussion:
        rule1 = (
            f"1. The discussion initiator"
            f'{f" (\"{trigger_agent_id}\")" if trigger_agent_id else ""} '
            "should speak first to kick off the topic. Then the teacher responds to guide the "
            "discussion. After that, other students may add their perspectives."
        )
    else:
        rule1 = (
            "1. The teacher (role: teacher, highest priority) should usually speak first "
            "to address the user's question or topic."
        )

    whiteboard_section = _build_whiteboard_state_for_director(whiteboard_ledger)

    student_profile_section = ""
    if user_profile and (user_profile.get("nickname") or user_profile.get("bio")):
        nickname = user_profile.get("nickname") or "Unknown"
        bio = user_profile.get("bio")
        student_profile_section = (
            "\n# Student Profile\n"
            f"Student name: {nickname}\n"
            f"{f'Background: {bio}' if bio else ''}\n"
        )

    wb_status = (
        "OPEN (slide canvas is hidden — spotlight/laser will not work)"
        if whiteboard_open
        else "CLOSED (slide canvas is visible)"
    )

    # ── Multi-agent engagement hint ────────────────────────────────
    # When 2+ agents are available we want the classroom to feel like a real
    # multi-voice discussion instead of "teacher answers, end". Build a
    # targeted section that summarizes which roles have / haven't spoken yet
    # and pushes the director toward role diversity on turn 2+.
    total_agents = len(agents)
    responded_ids = {
        (r.get("agentId") or r.get("agent_id")) for r in agent_responses
    } if agent_responses else set()
    unspoken_agents = [a for a in agents if a.id not in responded_ids]
    spoken_roles = {
        a.role for a in agents if a.id in responded_ids
    }
    unspoken_non_teacher = [
        a for a in unspoken_agents if (a.role or "").lower() != "teacher"
    ]

    if total_agents >= 2 and len(responded_ids) == 0:
        engagement_section = (
            "\n# Multi-Agent Engagement\n"
            f"There are {total_agents} agents available this round and none has "
            "spoken yet. This is a multi-voice classroom — aim for 2-3 different "
            "roles to contribute before ending.\n"
        )
    elif total_agents >= 2 and len(responded_ids) >= 1 and unspoken_non_teacher:
        unspoken_preview = ", ".join(
            f'{a.name} ({a.id}, role={a.role})' for a in unspoken_non_teacher[:4]
        )
        spoken_role_list = ", ".join(sorted(spoken_roles)) or "(none)"
        engagement_section = (
            "\n# Multi-Agent Engagement (IMPORTANT)\n"
            f"Only {len(responded_ids)} of {total_agents} agents have spoken this round. "
            f"Roles that already spoke: {spoken_role_list}. "
            f"Roles that have NOT spoken yet: {unspoken_preview}.\n"
            "This is a multi-voice classroom, NOT a single-teacher Q&A. Before "
            "ending, you should dispatch at least one more agent with a DIFFERENT "
            "role (typically a student or assistant) so the discussion feels alive. "
            "Only output END if:\n"
            "  (a) the user asked a trivial yes/no or one-word question AND it's "
            "fully answered, OR\n"
            "  (b) an unspoken agent has nothing new to contribute (same role as "
            "one who already spoke, or no relevant perspective).\n"
            "In doubt, dispatch another agent rather than ending.\n"
        )
    else:
        engagement_section = ""

    return f"""You are the Director of a multi-agent classroom. Your job is to decide which agent should speak next based on the conversation context.

# Available Agents
{agent_list}

# Agents Who Already Spoke This Round
{responded_list}

# Conversation Context
{conversation_summary}
{discussion_section}{whiteboard_section}{student_profile_section}{engagement_section}
# Rules
{rule1}
2. After the teacher, a student/assistant agent SHOULD normally follow up (ask a follow-up question, crack a joke, take notes, offer a different perspective). Skipping this step makes the classroom feel like a solo lecture.
3. Do NOT repeat an agent who already spoke this round unless absolutely necessary.
4. Only output END when the discussion is genuinely complete AND every remaining agent would be redundant. When in doubt with a multi-agent roster, dispatch one more agent (of a different role) instead of ending.
5. Current turn: {turn_count + 1}. Consider conversation length — don't let discussions drag on unnecessarily, but also don't cut them short after a single agent.
6. Prefer brevity per agent, NOT brevity in agent count — aim for 2-3 short voices rather than one long monologue. With 2+ agents available, a single-agent turn is almost always too short.
7. You can output {{"next_agent":"USER"}} to cue the user to speak. Use this when a student asks the user a direct question or when the topic naturally calls for user input. Do NOT use USER just to shortcut out of dispatching another agent.
8. Consider whiteboard state when routing: if the whiteboard is already crowded, avoid dispatching agents that are likely to add more whiteboard content unless they would clear or organize it.
9. Whiteboard is currently {wb_status}. When the whiteboard is open, do not expect spotlight or laser actions to have visible effect.

# Routing Quality (CRITICAL)
- ROLE DIVERSITY: Do NOT dispatch two agents of the same role consecutively. After a teacher speaks, the next should be a student or assistant — not another teacher-like response. After an assistant rephrases, dispatch a student who asks a question, not another assistant who also rephrases.
- CONTENT DEDUP: Read the "Agents Who Already Spoke" previews carefully. If an agent already explained a concept thoroughly, do NOT dispatch another agent to explain the same concept. Instead, dispatch an agent who will ASK a question, CHALLENGE an assumption, CONNECT to another topic, or TAKE NOTES.
- DISCUSSION PROGRESSION: Each new agent should advance the conversation. Good progression: explain → question → deeper explanation → different perspective → summary. Bad progression: explain → re-explain → rephrase → paraphrase.
- GREETING RULE: If any agent has already greeted the students, no subsequent agent should greet again. Check the previews for greetings.

# Output Format
You MUST output ONLY a JSON object, nothing else:
{{"next_agent":"<agent_id>"}}
or
{{"next_agent":"USER"}}
or
{{"next_agent":"END"}}"""


# ---------------------------------------------------------------------------
# Conversation summary (mirrors summarizeConversation)
# ---------------------------------------------------------------------------

def summarize_conversation(
    messages: list[dict],
    max_messages: int = 10,
    max_content_length: int = 200,
) -> str:
    """Summarize conversation history for the director agent.

    Expects messages in OpenAI chat.completions format
    ``[{"role": "user"|"assistant"|"system", "content": "..."}, ...]``.
    """
    if not messages:
        return "No conversation history yet."

    recent = messages[-max_messages:]
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "user")
        role_label = "User" if role == "user" else "Assistant" if role == "assistant" else "System"
        content = msg.get("content") or ""
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        lines.append(f"[{role_label}] {content}")
    return "\n".join(lines)
