"""
Prompt templates for classroom generation — LEGACY HARDCODED STRINGS.

⚠️  Prefer :mod:`core.generation.prompts.loader` (file-based ``.md``
    templates) for any new prompt.  This module is retained because
    ``classroom_generator.py`` and a couple of routes still rely on it,
    and rewriting them requires aligning the template variable names.

Migration plan: route all prompt lookups through
``core.generation.prompts_api.build_prompt`` so the cut-over is one file.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lang_instruction(language: str) -> str:
    if language == "zh-CN":
        return "所有输出必须使用简体中文。"
    return "All output must be in English."


# ---------------------------------------------------------------------------
# Outline generation
# ---------------------------------------------------------------------------

OUTLINE_SYSTEM = """\
You are an expert instructional designer specializing in interactive classroom experiences.
Generate a structured course outline based on the user's requirements.
Return ONLY valid JSON — no markdown fences, no extra text.
"""


def outline_user_prompt(
    requirement: str,
    language: str,
    teacher_context: str = "",
    research_context: str = "",
    pdf_text: str = "",
) -> str:
    parts = [f"Course requirement: {requirement}"]
    if language:
        parts.append(f"Language: {language}")
    if teacher_context:
        parts.append(f"\n## Teaching Team\n{teacher_context}")
    if research_context:
        parts.append(f"\n## Research Context\n{research_context}")
    if pdf_text:
        parts.append(f"\n## Reference Material\n{pdf_text[:8000]}")

    parts.append("""
Return a JSON array of scene outlines:
[
  {
    "id": "scene_<nanoid>",
    "title": "string",
    "description": "string",
    "scene_type": "slide" | "quiz" | "interactive" | "pbl",
    "slide_count": <number, only for slides>,
    "image_descriptions": ["string", ...]
  }
]
Guidelines:
- 4–8 scenes total
- Mix scene types: ~60% slides, ~20% quiz, ~10% interactive, ~10% pbl
- Titles and descriptions in the specified language
""")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scene content generation
# ---------------------------------------------------------------------------

SLIDE_SYSTEM = """\
You are an expert slide designer for interactive AI classrooms.
Generate rich slide content based on the scene outline.
Return ONLY valid JSON.
"""


def slide_user_prompt(
    outline: dict,
    all_outlines: list[dict],
    stage_info: dict,
    agents: list[dict],
    language: str,
) -> str:
    agent_names = ", ".join(a.get("name", "") for a in agents if a.get("role") == "teacher")
    return f"""
Generate slide content for this scene:

Title: {outline.get("title")}
Description: {outline.get("description", "")}
Slide count: {outline.get("slide_count", 3)}
Language: {language}
Course: {stage_info.get("name", "")}
Teacher: {agent_names}

Return JSON:
{{
  "scene_type": "slide",
  "slides": [
    {{
      "id": "slide_<id>",
      "title": "string",
      "elements": [
        {{"type": "heading", "id": "h1", "content": "string"}},
        {{"type": "text", "id": "t1", "content": "string"}},
        {{"type": "bullet", "id": "b1", "content": ["item1", "item2"]}},
        {{"type": "image", "id": "img1", "content": {{"placeholder": "description of image"}}}}
      ],
      "speaker_notes": "string"
    }}
  ],
  "summary": "one sentence summary"
}}
"""


QUIZ_SYSTEM = """\
You are an expert quiz designer for interactive AI classrooms.
Generate quiz questions based on the scene outline.
Return ONLY valid JSON.
"""


def quiz_user_prompt(outline: dict, language: str) -> str:
    return f"""
Generate quiz content for:

Title: {outline.get("title")}
Description: {outline.get("description", "")}
Language: {language}

Return JSON:
{{
  "scene_type": "quiz",
  "questions": [
    {{
      "id": "q_<id>",
      "type": "single" | "multiple" | "text",
      "question": "string",
      "options": [{{"id": "opt_<id>", "text": "string", "is_correct": true|false}}],
      "explanation": "string",
      "points": 1
    }}
  ],
  "summary": "string"
}}
Include 3–5 questions mixing single choice, multiple choice, and one short-answer text question.
"""


INTERACTIVE_SYSTEM = """\
You are an expert interactive simulation designer.
Generate a self-contained HTML interactive simulation.
Return ONLY valid JSON.
"""


def interactive_user_prompt(outline: dict, language: str) -> str:
    return f"""
Generate an interactive HTML simulation for:

Title: {outline.get("title")}
Description: {outline.get("description", "")}
Language: {language}

Return JSON:
{{
  "scene_type": "interactive",
  "html": "<complete self-contained HTML with inline CSS and JS>",
  "summary": "string"
}}

Requirements for the HTML:
- Single file, all CSS/JS inline
- Interactive and educational
- Works without external dependencies
- Responsive design
- Language must be {language}
"""


PBL_SYSTEM = """\
You are an expert Project-Based Learning (PBL) designer.
Generate a PBL activity with agents and issues.
Return ONLY valid JSON.
"""


def pbl_user_prompt(outline: dict, agents: list[dict], language: str) -> str:
    agent_list = "\n".join(f"- {a.get('name')} ({a.get('role')})" for a in agents)
    return f"""
Generate PBL activity content for:

Title: {outline.get("title")}
Description: {outline.get("description", "")}
Language: {language}

Available agents:
{agent_list}

Return JSON:
{{
  "scene_type": "pbl",
  "scenario": "string (2-3 sentence scenario description)",
  "issues": [
    {{
      "id": "issue_<id>",
      "title": "string",
      "description": "string",
      "person_in_charge": "agent name"
    }}
  ],
  "agents": [
    {{
      "id": "agent_<id>",
      "name": "string",
      "role": "string",
      "system_prompt": "string (role-play instructions for this agent)"
    }}
  ],
  "summary": "string"
}}
"""


# ---------------------------------------------------------------------------
# Scene actions generation
# ---------------------------------------------------------------------------

ACTIONS_SYSTEM = """\
You are an expert classroom action choreographer.
Generate speech and whiteboard actions for a scene.
Return ONLY valid JSON.
"""


def actions_user_prompt(
    outline: dict,
    content: dict,
    agents: list[dict],
    language: str,
    previous_speeches: list[str] | None = None,
) -> str:
    agent_info = "\n".join(
        f"- id={a.get('id')} name={a.get('name')} role={a.get('role')}" for a in agents
    )
    prev = ""
    if previous_speeches:
        prev = "\n\nPrevious speech texts (avoid repetition):\n" + "\n".join(
            f"- {s[:100]}" for s in previous_speeches[-5:]
        )

    scene_type = content.get("scene_type", "slide")
    return f"""
Generate teaching actions for this {scene_type} scene:

Title: {outline.get("title")}
Language: {language}
Agents:
{agent_info}
{prev}

Return JSON:
{{
  "actions": [
    {{
      "type": "speech",
      "id": "speech_<id>",
      "agent_id": "<agent id>",
      "text": "spoken text",
      "audio_id": "audio_<id>"
    }},
    {{
      "type": "spotlight",
      "id": "spotlight_<id>",
      "target": "element_id or slide_id"
    }}
  ]
}}

Guidelines:
- Teacher opens with welcome/intro speech
- Each slide/question should have 1-2 speech actions
- Keep speeches concise (20-60 words each)
- Use the exact agent IDs provided
- Language must be {language}
"""


# ---------------------------------------------------------------------------
# Agent profiles generation
# ---------------------------------------------------------------------------

AGENT_PROFILES_SYSTEM = """\
You are an expert instructional designer.
Generate agent profiles for a multi-agent classroom simulation.
Return ONLY valid JSON, no markdown or explanation.
"""


def agent_profiles_user_prompt(
    stage_info: dict,
    scene_outlines: list[dict],
    language: str,
    available_avatars: list[str],
) -> str:
    outlines_text = "\n".join(
        f"- {o.get('title', '')}: {o.get('description', '')}" for o in (scene_outlines or [])
    )
    avatars_text = ", ".join(available_avatars[:10]) if available_avatars else "default"
    return f"""
Generate agent profiles for this course:

Course: {stage_info.get("name", "")}
Description: {stage_info.get("description", "")}
Language: {language}
Scene outlines:
{outlines_text}

Available avatars: {avatars_text}

Return JSON:
{{
  "agents": [
    {{
      "name": "string",
      "role": "teacher" | "assistant" | "student",
      "persona": "2-3 sentence personality description",
      "avatar": "one of the available avatars"
    }}
  ]
}}

Rules:
- 3-5 agents total
- Exactly 1 teacher
- Names and personas in language: {language}
"""


# ---------------------------------------------------------------------------
# Quiz grading
# ---------------------------------------------------------------------------

def quiz_grade_system(points: int, language: str) -> str:
    if language == "zh-CN":
        return f"""你是一位专业的教育评估专家。请根据题目和学生答案进行评分并给出简短评语。
必须以如下 JSON 格式回复（不要包含其他内容）：
{{"score": <0到{points}的整数>, "comment": "<一两句评语>"}}"""
    return f"""You are a professional educational assessor. Grade the student's answer.
Reply ONLY in this JSON format:
{{"score": <integer 0 to {points}>, "comment": "<one or two sentences>"}}"""


def quiz_grade_user(question: str, user_answer: str, points: int, comment_prompt: str = "", language: str = "zh-CN") -> str:
    if language == "zh-CN":
        return f"""题目：{question}
满分：{points}分
{f"评分要点：{comment_prompt}" if comment_prompt else ""}
学生答案：{user_answer}"""
    return f"""Question: {question}
Full marks: {points} points
{f"Grading guidance: {comment_prompt}" if comment_prompt else ""}
Student answer: {user_answer}"""
