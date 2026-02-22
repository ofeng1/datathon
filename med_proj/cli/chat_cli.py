"""Terminal REPL chatbot for ED risk assessment.

Usage:
    python -m med_proj.cli.chat_cli
"""

from __future__ import annotations

import sys

from med_proj.chatbot.engine import ChatEngine


def main() -> None:
    engine = ChatEngine()

    print("=" * 60)
    print("  ED Risk Assessment Chatbot  (type 'quit' to exit)")
    print("=" * 60)
    print()
    print(engine.respond("hello"))
    print()

    while True:
        try:
            msg = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        reply = engine.respond(msg)
        print()
        print(reply)
        print()


if __name__ == "__main__":
    main()
