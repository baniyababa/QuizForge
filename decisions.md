# QuizForge — decisions.md

This document explains the key design decisions I made while building QuizForge, organized by milestone.

---

## Milestone 1: Chunk Size

### What I picked and why

I chose a chunk size of **3 sentences with 1 sentence of overlap**.

The notes file is structured with topic headers followed by dense, information-packed paragraphs. Each sentence usually introduces a distinct fact or definition, but individual sentences often reference the one before them. Three sentences hits a sweet spot: each chunk contains enough context to form a meaningful, self-contained idea (usually a concept + its explanation + an example or implication), but stays short enough that the retriever can distinguish between different concepts within the same topic.

The 1-sentence overlap ensures that if a concept spans a sentence boundary, it won't be lost.

### What breaks if chunks are too small (1 sentence each)?

If I chunk at one sentence, I lose context. For example, consider these two sentences from the notes:

> "ReLU outputs the input directly if positive, otherwise outputs zero."
> "The sigmoid function squashes values between 0 and 1 and is often used in binary classification output layers."

As standalone single-sentence chunks, a question about "which activation function is used for binary classification output layers" might retrieve the ReLU sentence instead of (or in addition to) the sigmoid one, because both mention activation-function-related terms. The retriever has no surrounding context to disambiguate. The LLM then gets a fragment without enough information to write a good question with meaningful distractors.

Also, embedding a single sentence gives the vector model less semantic signal to work with — short texts produce noisier embeddings.

### What breaks if chunks are too large (4–6 sentences)?

If chunks are too large, two problems emerge:

1. **Retrieval precision drops.** A 6-sentence chunk might cover two or three sub-concepts. When I query "tell me about overfitting," I might get back a chunk that's 50% about overfitting and 50% about regularization. That's not terrible, but it means the LLM has to sift through more irrelevant text, and the questions it generates might conflate separate ideas.

2. **Fewer chunks total = less variety.** With my notes file, 6-sentence chunks would produce roughly 12–15 chunks instead of ~25–30. That means the quiz draws from a smaller pool and is more likely to repeat concepts across quiz sessions.

The short answer: too small loses context and creates noise; too large reduces precision and variety. Three sentences is the practical middle ground for this document size.

---

## Milestone 2: Grading Logic

### How and why I gave scores

The grading works by sending the student's answer, the original question, and the reference passage from the notes to Claude with a grading prompt. The LLM compares the meaning of the student's answer against the source material and returns a score from 0 to 10 with one line of feedback.

The key instruction in my grading prompt is: **"The student may use different wording — that is fine as long as the meaning is correct."** This is critical because a student who writes "neural nets use backprop to calculate how much each weight contributed to the error" is saying the same thing as "backpropagation computes gradients of the loss function with respect to each weight using the chain rule" — just in simpler language.

The scoring scale works roughly like this:
- **9–10**: Captures the core concept correctly and completely
- **7–8**: Mostly correct, might miss a nuance or a detail
- **5–6**: Partially correct — gets the general idea but misses key parts
- **3–4**: Shows some understanding but has significant gaps or errors
- **0–2**: Incorrect or completely off-topic

I also include the source passage in the grading response so the student can see exactly what they were being graded against — this makes the grading transparent and helps them learn from mistakes.

---

## Milestone 3: Difficulty Encoding

### How I encoded the three difficulty levels

I used three separate system-prompt paragraphs, one per level. Here are the relevant parts:

```
EASY: "Generate EASY questions: test basic definitions and recall of key terms."

MEDIUM: "Generate MEDIUM questions: test understanding of concepts and how they relate to each other."

HARD: "Generate HARD questions: test application, comparison across topics, and nuanced understanding.
        Include plausible distractors."
```

The difficulty instruction is injected directly into the system prompt before the LLM generates questions. The adaptive logic checks the student's cumulative score percentage per topic: above 70% moves up a level, below 40% moves down.

### What if the LLM ignores instructions and gives a hard question when asked for an easy one?

This is a real risk. Prompt instructions are suggestions to the LLM, not hard constraints. If I ask for "easy" and the LLM generates a question comparing backpropagation across different network architectures, that's clearly not easy regardless of what the prompt said.

**How I handled it (current approach):** Honestly, right now I rely on the prompt being clear and specific about what each difficulty level means. Defining "easy = definitions and recall" vs "hard = application and comparison" gives the LLM concrete criteria rather than vague labels.

**One concrete fix I would add:** A post-generation validation step. After the LLM returns the questions, I'd send each question back to the LLM with a second prompt like:

```
"Rate this question's difficulty as easy/medium/hard based on these criteria:
 - easy: tests recall of a single definition
 - medium: tests understanding or relationship between two concepts
 - hard: tests application, comparison, or requires synthesizing multiple concepts
 Question: [question text]"
```

If the LLM's rating doesn't match the requested difficulty, I'd regenerate that specific question. This is an actual code check (a second LLM call), not just a better prompt. It adds latency but guarantees difficulty alignment.

---

## Milestone 4: Hallucination and Data Sufficiency

### Question 1: How does my system catch hallucination? If it doesn't, suggest a concrete fix.

My system has **one basic check**: after the LLM generates quiz questions, I verify that each question's `topic` field matches one of the real topics extracted from the notes file. If the LLM invents a topic like "Reinforcement Learning" that doesn't exist in `ml_basics.txt`, that question gets filtered out.

But this doesn't catch the more subtle case: a question whose topic is correct but whose content is fabricated. For example, the LLM might generate a question about "Neural Networks" (valid topic) but claim that "ReLU outputs values between -1 and 1" (wrong — that's tanh, not ReLU). My current system would let that through.

**Concrete fix I would add:**

After the LLM generates each question, I would:
1. Take the question text and the claimed correct answer
2. Retrieve the top-3 most relevant chunks from the vector store using the question as a query
3. Send those chunks + the question + claimed answer to the LLM with a verification prompt: "Is the following question and answer fully supported by the source passages? Reply YES or NO with a brief explanation."
4. If NO, discard the question and generate a replacement.

This is a **retrieval-based fact-check** — I'm using RAG a second time to verify the LLM's own output. It's not a prompt tweak; it's a second retrieval + LLM call in the pipeline. It adds cost and latency, but it catches factual hallucinations that topic-matching alone cannot.

### Question 2: A student has only answered 2 questions total. `get_weak_topics(0.5)` returns 4 "weak" topics. Is that plan good?

**No, that plan is bad.** Here's the problem:

If a student answered just 2 questions and got one wrong, they might show 0% on one topic and 100% on another. The tool might then label 4 topics as "weak" — but it's basing that judgment on 1 question per topic at most. That's not a pattern; that's noise. You can't conclude someone is weak at "Neural Networks" because they missed a single question about activation functions.

A study plan built on this data would send the student chasing 4 different topics when they barely started studying. It over-fits to random variance.

**How I fixed it:**

In my `get_weak_topics()` implementation, I added a `MIN_QUESTIONS = 3` threshold. The tool skips any topic where the student has answered fewer than 3 questions. So if you've only answered 2 questions total, `get_weak_topics()` returns a message saying "not enough data" instead of a misleading list.

I also instructed the LLM in the study plan system prompt: "If there is insufficient data (very few attempts), say so honestly and recommend a general study approach instead of over-fitting to sparse data." This way, even if the tool somehow returns thin data, the LLM is primed to flag the uncertainty instead of pretending it has a reliable picture.

The rule of thumb: **don't optimize a plan on a sample size of 2.** Tell the student to take more quizzes first.
