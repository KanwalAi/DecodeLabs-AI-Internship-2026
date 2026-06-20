# ============================================================
# Project 1: Rule-Based AI Chatbot
# Intern   : Kanwal Fatima
# Company  : Decode Labs  |  Batch: 2026
# Track    : Artificial Intelligence (AI)
# ============================================================

# ── KNOWLEDGE BASE ──────────────────────────────────────────
# Dictionary-based O(1) lookup instead of fragile if-elif ladder
RESPONSES = {
    # Greetings
    "hello"       : "Hello! 👋 I'm DecoBot, your AI assistant. How can I help you today?",
    "hi"          : "Hi there! 😊 Welcome to Decode Labs AI. What can I do for you?",
    "hey"         : "Hey! Great to see you. Ask me anything!",
    "good morning": "Good morning! ☀️ Ready to learn some AI today?",
    "good evening": "Good evening! 🌙 Hope your day went well. How can I assist?",
    "good afternoon":"Good afternoon! 🌤️ What's on your mind?",

    # About
    "who are you" : "I'm DecoBot 🤖 — a rule-based AI chatbot built by Kanwal Fatima as part of Decode Labs Project 1!",
    "what are you": "I'm a Rule-Based AI Chatbot. I respond using a dictionary knowledge base and if-else logic — no deep learning yet!",
    "your name"   : "My name is DecoBot! Built with Python and pure control-flow logic. 🐍",

    # AI / Tech questions
    "what is ai"          : "AI (Artificial Intelligence) is the simulation of human intelligence by machines. It includes learning, reasoning, and problem-solving.",
    "what is machine learning" : "Machine Learning is a subset of AI where machines learn from data to improve performance without being explicitly programmed.",
    "what is python"      : "Python is a high-level, general-purpose programming language widely used in AI, data science, and web development. 🐍",
    "what is deep learning": "Deep Learning uses neural networks with many layers to learn complex patterns from large amounts of data.",
    "what is a chatbot"   : "A chatbot is a software application that simulates human conversation — just like me! I use rules; smarter ones use NLP and ML.",

    # Decode Labs
    "what is decode labs" : "Decode Labs is a Government-Registered Digital Enterprise providing hands-on AI internship experience. 🚀",
    "about decode labs"   : "Decode Labs (decodelabs.tech) offers industrial AI training kits to build real-world portfolios for future employers.",
    "internship"          : "This is a 1-month remote AI internship at Decode Labs! I'm completing 3 projects — chatbot, classification, and recommendation.",

    # Help
    "help": (
        "\n📋 Here's everything you can type:\n"
        "\n🗣️  GREETINGS\n"
        "   hello | hi | hey | good morning | good evening | good afternoon\n"
        "\n🤖  ABOUT ME\n"
        "   who are you | what are you | your name\n"
        "\n🧠  AI & TECH\n"
        "   what is ai | what is python | what is machine learning\n"
        "   what is deep learning | what is a chatbot\n"
        "\n🏢  DECODE LABS\n"
        "   what is decode labs | about decode labs | internship\n"
        "\n😂  FUN\n"
        "   joke | tell me a joke | another joke\n"
        "   are you smart | are you human\n"
        "\n❤️   FEELINGS\n"
        "   how are you | i am fine | i am sad | i am bored\n"
        "\n🙏  GRATITUDE\n"
        "   thank you | thanks | great | nice\n"
        "\nℹ️   OTHER\n"
        "   help | topics | what can you do\n"
        "\n🚪  EXIT\n"
        "   exit | quit | bye | goodbye | q\n"
        "\n💡 Tip: Type 'topics' to see a quick category overview!"
    ),
    "topics": (
        "\n📌 Topic Categories:\n"
        "  🗣️  Greetings     → try: hello, hi, good morning\n"
        "  🤖  About Me      → try: who are you, your name\n"
        "  🧠  AI & Tech     → try: what is ai, what is python\n"
        "  🏢  Decode Labs   → try: what is decode labs, internship\n"
        "  😂  Fun & Jokes   → try: joke, tell me a joke\n"
        "  ❤️   Feelings      → try: how are you, i am sad\n"
        "  🙏  Gratitude     → try: thank you, thanks\n"
        "  🚪  Exit          → try: bye, exit, quit\n"
        "\n💡 Type 'help' to see ALL exact commands!"
    ),
    "what can you do": "I can answer questions about AI & tech, chat about Decode Labs, tell jokes, and respond to your feelings! Type 'help' to see all exact commands 😊",

    # Fun
    "tell me a joke"  : "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😂",
    "joke"            : "Why did the AI break up with the algorithm? It said 'You're too predictable!' 😂",
    "another joke"    : "How many programmers does it take to change a light bulb? None — that's a hardware problem! 💡",
    "are you smart"   : "I'm rule-based smart! I can't learn yet, but I respond really fast. Ask me a tech question 😄",
    "are you human"   : "Nope! I'm 100% Python code — no feelings, no coffee breaks, just pure logic. ☕❌",

    # Feelings / State
    "how are you"     : "I'm running perfectly — no bugs today! 🎉 How about you?",
    "i am fine"       : "Great to hear! 😊 Anything I can help you with?",
    "i am sad"        : "I'm sorry to hear that 😔 Remember — every great AI engineer faces bugs before breakthroughs. You've got this!",
    "i am bored"      : "Let's fix that! Ask me something interesting about AI or request a joke. 🎭",

    # Gratitude
    "thank you"  : "You're welcome! 😊 Happy to help. Keep building!",
    "thanks"     : "Anytime! 🙌 Good luck with your projects.",
    "great"      : "Awesome! Keep up the momentum. 🚀",
    "nice"       : "Glad you think so! 😄",

    # Exit aliases (handled in main loop, listed here for clarity)
    "bye"     : "Goodbye! 👋 Keep coding and stay curious!",
    "goodbye" : "See you later! 🌟 Best of luck with your Decode Labs journey!",
    "quit"    : "Exiting... Remember: an LLM without rules is a hallucination engine. You built the skeleton! 🏗️",
}

# Keywords → intent mapping  (multi-word input support)
KEYWORD_MAP = {
    "machine learning" : "what is machine learning",
    "deep learning"    : "what is deep learning",
    "good morning"     : "good morning",
    "good evening"     : "good evening",
    "good afternoon"   : "good afternoon",
    "who are you"      : "who are you",
    "what are you"     : "what are you",
    "your name"        : "your name",
    "what is ai"       : "what is ai",
    "what is python"   : "what is python",
    "what is a chatbot": "what is a chatbot",
    "what is decode labs"  : "what is decode labs",
    "about decode labs"    : "about decode labs",
    "how are you"      : "how are you",
    "i am fine"        : "i am fine",
    "i am sad"         : "i am sad",
    "i am bored"       : "i am bored",
    "tell me a joke"   : "tell me a joke",
    "another joke"     : "another joke",
    "are you smart"    : "are you smart",
    "are you human"    : "are you human",
    "what can you do"  : "what can you do",
    "thank you"        : "thank you",
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "q"}

FALLBACK = "🤔 Hmm, I don't understand that yet. Try asking about AI, Python, or type 'help' to see what I know!"

BANNER = """
╔══════════════════════════════════════════════════╗
║         DecoBot — Rule-Based AI Chatbot          ║
║   Decode Labs  |  Project 1  |  Batch 2026       ║
║   Built by: Kanwal Fatima                        ║
╠══════════════════════════════════════════════════╣
║  Type 'help' for topics  |  'exit' to quit       ║
╚══════════════════════════════════════════════════╝
"""

# ── PHASE 1: INPUT SANITIZATION ─────────────────────────────
def sanitize(raw: str) -> str:
    """Lowercase + strip whitespace — the normalization layer."""
    return raw.lower().strip()

# ── PHASE 2: INTENT MATCHING ────────────────────────────────
def match_intent(clean_input: str) -> str:
    """
    O(1) dictionary lookup with multi-word keyword fallback.
    Priority:
      1. Exact match in RESPONSES
      2. Multi-word keyword match in KEYWORD_MAP
      3. Single-word keyword scan
      4. Fallback
    """
    # 1. Exact match
    if clean_input in RESPONSES:
        return RESPONSES[clean_input]

    # 2. Multi-word keyword match (longest match wins)
    for phrase, key in sorted(KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in clean_input:
            return RESPONSES.get(key, FALLBACK)

    # 3. Single-word scan
    for word in clean_input.split():
        if word in RESPONSES:
            return RESPONSES[word]

    # 4. Fallback
    return FALLBACK

# ── PHASE 3: RESPONSE GENERATION ────────────────────────────
def get_response(user_input: str) -> str:
    clean = sanitize(user_input)
    return match_intent(clean)

# ── THE HEARTBEAT: INFINITE LOOP ────────────────────────────
def run_chatbot():
    print(BANNER)
    print("DecoBot: Hello! I'm ready to chat. What's on your mind? 😊\n")

    while True:                          # ── while True: infinite loop
        try:
            raw_input_text = input("You: ").strip()

            if not raw_input_text:       # ignore empty enter
                continue

            clean = sanitize(raw_input_text)

            # ── EXIT STRATEGY: clean break command ──────────
            if clean in EXIT_COMMANDS:
                farewell = RESPONSES.get(clean, "Goodbye! 👋 Keep learning!")
                print(f"\nDecoBot: {farewell}\n")
                break                    # ── kill command

            # ── PROCESS & RESPOND ───────────────────────────
            reply = get_response(raw_input_text)
            print(f"\nDecoBot: {reply}\n")

        except KeyboardInterrupt:
            print("\n\nDecoBot: Ctrl+C detected. Shutting down gracefully. Bye! 👋\n")
            break

# ── ENTRY POINT ─────────────────────────────────────────────
if __name__ == "__main__":
    run_chatbot()
