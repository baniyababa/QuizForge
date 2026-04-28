"""
QuizForge - Terminal-based AI study tool
Usage: python main.py --notes ml_basics.txt
Commands: /quiz, /quiz open, /stats, /plan, /help, /quit
"""

import argparse
import json
import os
import re
import sys
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check — give a friendly message if something is missing
# ---------------------------------------------------------------------------
MISSING = []
try:
    import google.generativeai as genai
except ImportError:
    MISSING.append("google-generativeai")
try:
    import chromadb
except ImportError:
    MISSING.append("chromadb")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING.append("sentence-transformers")

if MISSING:
    print("❌  Missing packages. Run this command first:\n")
    print(f"   pip install {' '.join(MISSING)}\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERF_FILE = "performance.json"
CHUNK_SIZE = 3          # sentences per chunk  (see decisions.md)
CHUNK_OVERLAP = 1       # overlap in sentences for context continuity
COLLECTION_NAME = "notes"
DIFFICULTY_UP = 0.70    # 70 % → level up
DIFFICULTY_DOWN = 0.40  # 40 % → level down
DIFFICULTIES = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# 1.  Document loading, chunking, and vector store
# ---------------------------------------------------------------------------

def load_notes(filepath: str) -> str:
    """Read the notes file and return its full text."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_topics(text: str) -> list[str]:
    """Pull topic names from numbered headers followed by underlines like '1. Title\\n---'"""
    return re.findall(r"^\d+\.\s+(.+)\n-{2,}", text, re.MULTILINE)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks of `chunk_size` sentences.
    Each chunk carries metadata with the topic it belongs to.
    """
    chunks = []
    current_topic = "General"

    # Split by numbered topic sections (e.g. "1. What is Machine Learning?\n---")
    sections = re.split(r"(\d+\.\s+.+\n-+)", text)

    for section in sections:
        topic_match = re.match(r"\d+\.\s+(.+)\n-+", section.strip())
        if topic_match:
            current_topic = topic_match.group(1).strip()
            continue

        # Skip empty / header-only sections
        clean = section.strip()
        if not clean or clean.startswith("Machine Learning Basics") or clean.startswith("==="):
            continue

        # Split section into sentences
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Sliding window
        i = 0
        while i < len(sentences):
            window = sentences[i : i + chunk_size]
            chunk_text_str = " ".join(window)
            chunks.append({"text": chunk_text_str, "topic": current_topic})
            i += chunk_size - overlap  # step forward by (size - overlap)

    return chunks


def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """Embed chunks and store them in an in-memory ChromaDB collection."""
    print("⏳  Building vector store (first run takes ~30 s to download model)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()  # in-memory, no server needed
    # Delete existing collection if it exists so re-runs are clean
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    topics = [c["topic"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"topic": t} for t in topics],
        ids=ids,
    )
    print(f"✅  Indexed {len(chunks)} chunks across {len(set(topics))} topics.\n")
    return collection, embedder


def retrieve_chunks(collection, embedder, query: str, n: int = 5, topic: str = None) -> list[dict]:
    """Retrieve the top-n most relevant chunks for a query, optionally filtered by topic."""
    query_emb = embedder.encode([query]).tolist()

    where_filter = {"topic": topic} if topic else None

    results = collection.query(
        query_embeddings=query_emb,
        n_results=n,
        where=where_filter,
    )
    out = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        out.append({"text": doc, "topic": meta["topic"]})
    return out


# ---------------------------------------------------------------------------
# 2.  Performance tracking  (Milestone 3)
# ---------------------------------------------------------------------------

def load_performance() -> dict:
    if os.path.exists(PERF_FILE):
        with open(PERF_FILE, "r") as f:
            return json.load(f)
    return {}


def save_performance(perf: dict):
    with open(PERF_FILE, "w") as f:
        json.dump(perf, f, indent=2)


def update_performance(perf: dict, topic: str, score: float, max_score: float):
    """Update performance dict for a topic after a quiz attempt."""
    if topic not in perf:
        perf[topic] = {
            "attempts": 0,
            "total_score": 0.0,
            "max_possible": 0.0,
            "difficulty": "easy",
        }
    entry = perf[topic]
    entry["attempts"] += 1
    entry["total_score"] += score
    entry["max_possible"] += max_score

    # Adjust difficulty
    pct = entry["total_score"] / entry["max_possible"] if entry["max_possible"] else 0
    current_idx = DIFFICULTIES.index(entry["difficulty"])
    if pct >= DIFFICULTY_UP and current_idx < len(DIFFICULTIES) - 1:
        entry["difficulty"] = DIFFICULTIES[current_idx + 1]
    elif pct <= DIFFICULTY_DOWN and current_idx > 0:
        entry["difficulty"] = DIFFICULTIES[current_idx - 1]

    save_performance(perf)


def show_stats(perf: dict):
    """Print the /stats table."""
    if not perf:
        print("\n📊  No performance data yet. Take a quiz first!\n")
        return
    print("\n" + "=" * 65)
    print(f"{'Topic':<28} {'Attempts':>8} {'Avg %':>8} {'Level':>10}")
    print("-" * 65)
    for topic, d in sorted(perf.items()):
        avg = (d["total_score"] / d["max_possible"] * 100) if d["max_possible"] else 0
        print(f"{topic:<28} {d['attempts']:>8} {avg:>7.1f}% {d['difficulty']:>10}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# 3.  LLM helpers  (Gemini)
# ---------------------------------------------------------------------------

def get_client():
    """Configure and return a Gemini model. Requires GEMINI_API_KEY env var."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌  Set your API key first:")
        print('    $env:GEMINI_API_KEY="your-key-here"')
        print("  Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def call_llm(model, system_prompt: str, user_prompt: str) -> str:
    """Call Gemini and return the text response."""
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}"
    response = model.generate_content(full_prompt)
    return response.text


# ---------------------------------------------------------------------------
# 4.  MCQ Quiz  (Milestone 1)
# ---------------------------------------------------------------------------

MCQ_SYSTEM = """You are a quiz generator for a study tool called QuizForge.
You MUST generate questions ONLY from the provided source material.
Do NOT use any outside knowledge.

{difficulty_instruction}

Return EXACTLY this JSON format and nothing else — no markdown fences, no commentary:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "topic": "...",
    "explanation": "..."
  }}
]
Generate exactly {n} questions. Each question must have exactly 4 options (A-D).
The "topic" field must match one of the topics in the source material exactly.
"""

DIFFICULTY_PROMPTS = {
    "easy": "Generate EASY questions: test basic definitions and recall of key terms.",
    "medium": "Generate MEDIUM questions: test understanding of concepts and how they relate to each other.",
    "hard": "Generate HARD questions: test application, comparison across topics, and nuanced understanding. Include plausible distractors.",
}


def run_mcq_quiz(model, collection, embedder, perf: dict, topics: list[str]):
    """Milestone 1 + 3:  MCQ quiz with adaptive difficulty."""
    difficulty = "medium"  # default
    if perf:
        worst = min(perf.items(), key=lambda x: x[1]["total_score"] / max(x[1]["max_possible"], 1)) if perf else None
        if worst:
            difficulty = worst[1]["difficulty"]

    chunks = retrieve_chunks(collection, embedder, "key concepts and definitions", n=10)
    context = "\n\n".join([f"[Topic: {c['topic']}]\n{c['text']}" for c in chunks])

    difficulty_instruction = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["medium"])
    system = MCQ_SYSTEM.format(difficulty_instruction=difficulty_instruction, n=5)
    user = f"Source material:\n\n{context}"

    print(f"\n🎯  Generating {difficulty.upper()} MCQ quiz...\n")
    raw = call_llm(model, system, user)

    # Parse JSON — strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️  LLM returned malformed JSON. Trying again...")
        raw2 = call_llm(model, system, user)
        raw2 = raw2.strip()
        if raw2.startswith("```"):
            raw2 = re.sub(r"^```\w*\n?", "", raw2)
            raw2 = re.sub(r"\n?```$", "", raw2)
        questions = json.loads(raw2)

    # --- Hallucination guard (Milestone 4 decision) ---
    verified_questions = []
    for q in questions:
        q_topic = q.get("topic", "")
        if q_topic in topics:
            verified_questions.append(q)
        else:
            for t in topics:
                if t.lower() in q_topic.lower() or q_topic.lower() in t.lower():
                    q["topic"] = t
                    verified_questions.append(q)
                    break
    questions = verified_questions[:5]

    if not questions:
        print("⚠️  Could not generate valid questions. Try again.\n")
        return

    # Present quiz
    score = 0
    topic_scores = {}

    for i, q in enumerate(questions, 1):
        print(f"━━━ Question {i}/{len(questions)} ━━━")
        print(f"  {q['question']}\n")
        for letter in ["A", "B", "C", "D"]:
            print(f"    {letter}. {q['options'][letter]}")

        while True:
            ans = input("\n  Your answer (A/B/C/D): ").strip().upper()
            if ans in ("A", "B", "C", "D"):
                break
            print("  Please enter A, B, C, or D.")

        correct = q["answer"].upper()
        topic = q.get("topic", "General")
        if ans == correct:
            print(f"  ✅  Correct!")
            score += 1
            earned = 1
        else:
            print(f"  ❌  Wrong. The answer is {correct}.")
            earned = 0
        print(f"  💡  {q.get('explanation', '')}\n")

        if topic not in topic_scores:
            topic_scores[topic] = [0, 0]
        topic_scores[topic][0] += earned
        topic_scores[topic][1] += 1

    print(f"\n🏆  Final Score: {score}/{len(questions)} ({score/len(questions)*100:.0f}%)\n")

    for topic, (earned, possible) in topic_scores.items():
        update_performance(perf, topic, earned, possible)


# ---------------------------------------------------------------------------
# 5.  Open-ended Quiz  (Milestone 2)
# ---------------------------------------------------------------------------

OPEN_SYSTEM = """You are a quiz generator for a study tool.
Generate questions ONLY from the provided source material. No outside knowledge.

{difficulty_instruction}

Return EXACTLY this JSON and nothing else — no markdown fences, no commentary:
[
  {{
    "question": "...",
    "topic": "...",
    "reference_passage": "the exact passage from the source that contains the answer"
  }}
]
Generate exactly {n} open-ended questions that require short written answers.
"""

GRADING_SYSTEM = """You are a fair grader for a study tool.
Compare the student's answer against the reference passage.
The student may use different wording — that is fine as long as the meaning is correct.
Give partial credit for partially correct answers.

Return EXACTLY this JSON and nothing else — no markdown fences, no commentary:
{{
  "score": <0-10>,
  "feedback": "<one concise line of feedback>",
  "source_passage": "<the reference passage you graded against>"
}}
"""


def run_open_quiz(model, collection, embedder, perf: dict, topics: list[str]):
    """Milestone 2 + 3: Open-ended quiz with LLM grading."""
    difficulty = "medium"
    if perf:
        worst = min(perf.items(), key=lambda x: x[1]["total_score"] / max(x[1]["max_possible"], 1)) if perf else None
        if worst:
            difficulty = worst[1]["difficulty"]

    chunks = retrieve_chunks(collection, embedder, "explain concepts in detail", n=10)
    context = "\n\n".join([f"[Topic: {c['topic']}]\n{c['text']}" for c in chunks])

    difficulty_instruction = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["medium"])
    system = OPEN_SYSTEM.format(difficulty_instruction=difficulty_instruction, n=3)
    user = f"Source material:\n\n{context}"

    print(f"\n📝  Generating {difficulty.upper()} open-ended questions...\n")
    raw = call_llm(model, system, user)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️  Could not parse questions. Try again.\n")
        return

    total_score = 0
    topic_scores = {}

    for i, q in enumerate(questions, 1):
        print(f"━━━ Question {i}/{len(questions)} ━━━")
        print(f"  {q['question']}\n")
        answer = input("  Your answer: ").strip()
        if not answer:
            print("  ⏭️  Skipped.\n")
            topic = q.get("topic", "General")
            if topic not in topic_scores:
                topic_scores[topic] = [0, 0]
            topic_scores[topic][0] += 0
            topic_scores[topic][1] += 10
            continue

        grade_user = (
            f"Question: {q['question']}\n"
            f"Reference passage: {q.get('reference_passage', 'N/A')}\n"
            f"Student's answer: {answer}"
        )
        grade_raw = call_llm(model, GRADING_SYSTEM, grade_user)
        grade_raw = grade_raw.strip()
        if grade_raw.startswith("```"):
            grade_raw = re.sub(r"^```\w*\n?", "", grade_raw)
            grade_raw = re.sub(r"\n?```$", "", grade_raw)

        try:
            grade = json.loads(grade_raw)
        except json.JSONDecodeError:
            print("  ⚠️  Could not parse grade. Giving 5/10 by default.\n")
            grade = {"score": 5, "feedback": "Could not parse grading response.", "source_passage": "N/A"}

        s = grade["score"]
        total_score += s
        print(f"  📊  Score: {s}/10")
        print(f"  💬  {grade['feedback']}")
        print(f"  📖  Source: \"{grade['source_passage'][:120]}...\"\n")

        topic = q.get("topic", "General")
        if topic not in topic_scores:
            topic_scores[topic] = [0, 0]
        topic_scores[topic][0] += s
        topic_scores[topic][1] += 10

    print(f"\n🏆  Total Score: {total_score}/{len(questions)*10} ({total_score/(len(questions)*10)*100:.0f}%)\n")

    for topic, (earned, possible) in topic_scores.items():
        update_performance(perf, topic, earned, possible)


# ---------------------------------------------------------------------------
# 6.  Study Plan with Tool Calling  (Milestone 4)
# ---------------------------------------------------------------------------

def get_plan_tools():
    """Build Gemini function declarations for the study plan tools."""
    return [
        genai.types.Tool(
            function_declarations=[
                genai.types.FunctionDeclaration(
                    name="get_performance_summary",
                    description="Returns an overview of the student's performance across all topics including attempts, average score percentage, and current difficulty level for each topic.",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={},
                    ),
                ),
                genai.types.FunctionDeclaration(
                    name="get_weak_topics",
                    description="Returns topics where the student's average score is below the given threshold (0-1 scale). Only returns topics where the student has answered at least 3 questions to ensure statistical significance.",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "threshold": genai.types.Schema(
                                type=genai.types.Type.NUMBER,
                                description="Score threshold between 0 and 1. Topics with average below this are considered weak.",
                            ),
                        },
                        required=["threshold"],
                    ),
                ),
                genai.types.FunctionDeclaration(
                    name="get_notes_outline",
                    description="Returns the list of all topic names available in the study notes, so the study plan can reference real topics.",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={},
                    ),
                ),
            ]
        )
    ]


def execute_tool(tool_name: str, tool_input: dict, perf: dict, topics: list[str]) -> str:
    """Execute a tool call and return the result as a JSON string."""
    if tool_name == "get_performance_summary":
        if not perf:
            return json.dumps({"message": "No performance data yet."})
        summary = {}
        for topic, d in perf.items():
            avg = (d["total_score"] / d["max_possible"] * 100) if d["max_possible"] else 0
            summary[topic] = {
                "attempts": d["attempts"],
                "average_score_pct": round(avg, 1),
                "current_difficulty": d["difficulty"],
                "total_questions_answered": round(d["max_possible"]),
            }
        return json.dumps(summary, indent=2)

    elif tool_name == "get_weak_topics":
        threshold = tool_input.get("threshold", 0.5)
        MIN_QUESTIONS = 3  # Milestone 4 decision: minimum data threshold
        weak = []
        for topic, d in perf.items():
            total_qs = d["max_possible"]
            if total_qs < MIN_QUESTIONS:
                continue  # not enough data to judge
            avg = d["total_score"] / d["max_possible"] if d["max_possible"] else 0
            if avg < threshold:
                weak.append({
                    "topic": topic,
                    "average_score": round(avg, 3),
                    "attempts": d["attempts"],
                    "current_difficulty": d["difficulty"],
                    "total_questions_answered": round(total_qs),
                })
        if not weak:
            return json.dumps({"message": "No weak topics found (or not enough data — need at least 3 questions per topic)."})
        return json.dumps(weak, indent=2)

    elif tool_name == "get_notes_outline":
        return json.dumps({"topics": topics})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


PLAN_SYSTEM = """You are a study plan generator for QuizForge.
You have access to tools that let you pull student performance data.
Use the tools to understand the student's strengths and weaknesses, then create a personalized study plan.

Your study plan MUST include:
1. Topics to focus on, ordered by importance (weakest first)
2. For each topic: recommended number of practice questions and starting difficulty level
3. Brief reasoning for each recommendation

Use the tools first to gather data before generating the plan. Call get_performance_summary first, then get_weak_topics with a threshold of 0.5, then get_notes_outline.

If there is insufficient data (very few attempts), say so honestly and recommend a general study approach instead of over-fitting to sparse data.
"""


def run_study_plan(perf: dict, topics: list[str]):
    """Milestone 4: Tool-calling study plan generation using Gemini function calling."""
    if not perf:
        print("\n📋  No performance data yet. Take some quizzes first!\n")
        return

    print("\n📋  Generating your personalized study plan...\n")

    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    plan_tools = get_plan_tools()
    plan_model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=PLAN_SYSTEM,
        tools=plan_tools,
    )

    chat = plan_model.start_chat()
    response = chat.send_message("Create a personalized study plan for me based on my quiz performance.")

    # Tool-call loop
    max_iterations = 10
    for _ in range(max_iterations):
        function_calls = []
        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                function_calls.append(part.function_call)

        if not function_calls:
            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)
            print()
            return

        # Execute each function call and send results back
        function_responses = []
        for fc in function_calls:
            tool_input = dict(fc.args) if fc.args else {}
            result = execute_tool(fc.name, tool_input, perf, topics)
            function_responses.append(
                genai.types.Part.from_function_response(
                    name=fc.name,
                    response={"result": json.loads(result)},
                )
            )

        response = chat.send_message(function_responses)

    print("⚠️  Study plan generation took too many iterations.\n")


# ---------------------------------------------------------------------------
# 7.  Main loop
# ---------------------------------------------------------------------------

def print_help():
    print("""
╔══════════════════════════════════════════════╗
║            QuizForge Commands                ║
╠══════════════════════════════════════════════╣
║  /quiz       → 5 MCQ questions              ║
║  /quiz open  → 3 open-ended questions        ║
║  /stats      → View your performance stats   ║
║  /plan       → Get a personalized study plan  ║
║  /help       → Show this help message         ║
║  /quit       → Exit QuizForge                 ║
╚══════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(description="QuizForge — AI-powered study tool")
    parser.add_argument("--notes", required=True, help="Path to your notes .txt file")
    args = parser.parse_args()

    # Load notes
    if not os.path.exists(args.notes):
        print(f"❌  File not found: {args.notes}")
        sys.exit(1)

    text = load_notes(args.notes)
    topics = extract_topics(text)
    chunks = chunk_text(text)

    print(f"\n📚  Loaded {len(topics)} topics from {args.notes}")
    print(f"    Topics: {', '.join(topics)}\n")

    # Build vector store
    collection, embedder = build_vector_store(chunks)

    # LLM client
    model = get_client()

    # Load existing performance
    perf = load_performance()

    print_help()

    # Command loop
    while True:
        try:
            cmd = input("quizforge> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if not cmd:
            continue
        elif cmd == "/quit" or cmd == "/exit":
            print("👋  Goodbye!")
            break
        elif cmd == "/help":
            print_help()
        elif cmd == "/quiz open":
            run_open_quiz(model, collection, embedder, perf, topics)
        elif cmd == "/quiz":
            run_mcq_quiz(model, collection, embedder, perf, topics)
        elif cmd == "/stats":
            show_stats(perf)
        elif cmd == "/plan":
            run_study_plan(perf, topics)
        else:
            print("  Unknown command. Type /help for options.\n")


if __name__ == "__main__":
    main()