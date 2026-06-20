# 🤖 DecoBot — Rule-Based AI Chatbot
### Project 1 | Decode Labs AI Internship | Batch 2026

> *"Before you build systems that learn on their own, you must master the art of teaching a machine through explicit if-else instructions."*
> — Decode Labs Industrial Training Kit

---

## 📌 Project Overview

**DecoBot** is a Rule-Based AI Chatbot built as **Project 1** of the Decode Labs AI Internship (Batch 2026).

This project demonstrates the **Logic Engine** — the foundation of all intelligent systems.  
Instead of deep learning, it uses **Control Flow + Dictionary-based intent matching** to simulate human conversation with deterministic, traceable responses.

| Field        | Details                          |
|--------------|----------------------------------|
| **Intern**   | Kanwal Fatima                    |
| **Track**    | Artificial Intelligence (AI)     |
| **Company**  | Decode Labs (`decodelabs.tech`)  |
| **Mode**     | Remote / Virtual                 |
| **Language** | Python 3.x                       |

---

## 🏗️ Architecture — The IPO Model

```
INPUT  →  PROCESS  →  OUTPUT
  │           │           │
Sanitize   Intent     Response
(lower +   Matching   Generation
 strip)    (O(1) Dict) (Formatted)
```

### Phase 1 — Input & Sanitization
```python
clean_input = raw_input.lower().strip()
```
Normalises `HeLLo`, `HELLO`, `  hello  ` → all map to the **same response**.

### Phase 2 — Intent Matching (Hash Map)
Uses a **Python Dictionary** for O(1) lookup — NOT a fragile `if-elif` ladder.
```python
reply = RESPONSES.get(user_input, FALLBACK_MESSAGE)
```

### Phase 3 — The Heartbeat (Infinite Loop)
```python
while True:
    user_input = get_input()
    if user_input in EXIT_COMMANDS:
        break        # ← Kill command
    process(user_input)
```

---

## ✅ Project 1 — Logic Skeleton Checklist

| Requirement       | Status | Implementation |
|-------------------|--------|----------------|
| **INPUT LOOP**    | ✅     | `while True` continuous cycle |
| **SANITIZATION**  | ✅     | `.lower().strip()` |
| **KNOWLEDGE BASE**| ✅     | Dictionary with 30+ intents (>5 required) |
| **FALLBACK**      | ✅     | Default response for unknown inputs |
| **EXIT STRATEGY** | ✅     | `exit`, `quit`, `bye`, `goodbye`, `q` |

---

## 💬 Supported Intents (30+)

| Category        | Example Inputs |
|-----------------|----------------|
| 🗣️ Greetings    | `hello`, `hi`, `hey`, `good morning`, `good evening` |
| 🤖 Identity     | `who are you`, `what are you`, `your name` |
| 🧠 AI & Tech    | `what is ai`, `machine learning`, `deep learning`, `python`, `chatbot` |
| 🏢 Decode Labs  | `what is decode labs`, `internship`, `about decode labs` |
| 😂 Fun/Jokes    | `joke`, `tell me a joke`, `are you human`, `are you smart` |
| ❤️ Feelings     | `how are you`, `i am sad`, `i am bored`, `i am fine` |
| 🙏 Gratitude    | `thank you`, `thanks`, `great`, `nice` |
| ℹ️ Help         | `help`, `topics`, `what can you do` |
| 🚪 Exit         | `exit`, `quit`, `bye`, `goodbye`, `q` |

---

## 🚀 How to Run

### Requirements
- Python 3.6 or above
- No external libraries required (pure Python!)

### Run the Chatbot
```bash
git clone https://github.com/KanwalAi/DecoBot-Rule-Based-Chatbot.git
cd DecoBot-Rule-Based-Chatbot
python chatbot.py
```

---

## 📂 Project Structure

```
project1_chatbot/
│
├── chatbot.py          # Main chatbot — all logic lives here
├── README.md           # This file
└── screenshots/        # Demo screenshots
```

---

## 🔬 Key Concepts Demonstrated

| Concept                   | Where Used |
|---------------------------|------------|
| **Dictionary (Hash Map)** | `RESPONSES` — O(1) intent lookup |
| **Input Sanitization**    | `sanitize()` function |
| **Infinite Loop**         | `while True` in `run_chatbot()` |
| **Exit Strategy**         | `EXIT_COMMANDS` set + `break` |
| **Fallback Logic**        | `FALLBACK` constant |
| **Multi-word Matching**   | `KEYWORD_MAP` longest-match scan |
| **Modular Functions**     | `sanitize()`, `match_intent()`, `get_response()` |

---

## 📊 Why Dictionary over If-Elif Ladder?

| Approach        | Complexity | Maintainability | Scalability |
|-----------------|------------|-----------------|-------------|
| `if-elif`       | O(n)       | ❌ High Debt    | ❌ Breaks at scale |
| **Dict Lookup** | **O(1)**   | **✅ Clean**    | **✅ Add any key** |

---

## 📸 Screenshots

> *See `/screenshots` folder in this repo.*

---

## 🎓 Learning Outcomes

- ✅ Built a continuous, interactive loop-based program
- ✅ Applied dictionary-based O(1) intent resolution
- ✅ Implemented input sanitization pipeline
- ✅ Understood the Hybrid AI Architecture (Rule + LLM)

---

## 👩‍💻 Author

**Kanwal Fatima**  
AI Student | Software Developer  
📧 kanwal.ai.pk@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/kanwal-fatima-72a352357/)  
🐙 [GitHub](https://github.com/KanwalAi)

---

## 🏢 About Decode Labs

**Decode Labs** — Your Digital Lab  
🌐 [www.decodelabs.tech](https://www.decodelabs.tech)  
✉️ hr@decodelabs.tech  
📍 Greater Lucknow, India  
✦ Govt. Registered Enterprise

---

*Project 1 of 3 — Rule-Based AI Chatbot | Decode Labs AI Internship 2026*
