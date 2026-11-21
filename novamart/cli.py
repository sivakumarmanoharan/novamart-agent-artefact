"""Command-line chat with the NovaMart support agent.

    python -m novamart.cli                      # interactive
    python -m novamart.cli -q "Where is NM-10002?"   # one question
    python -m novamart.cli --trace              # also print tool calls
"""

from __future__ import annotations

import argparse
import json
import sys

from novamart.agent import SupportAgent, Turn
from novamart.quota import ModelOverloadedError, QuotaExhaustedError


def print_turn(turn: Turn, trace: bool) -> None:
    if trace:
        for call in turn.tool_calls:
            summary = json.dumps(call.result)[:300]
            print(f"  [tool] {call.name}({json.dumps(call.args)}) -> {summary}")
    print(f"\nNovaMart Assistant: {turn.answer}\n")


def ask_and_print(agent: SupportAgent, question: str, trace: bool) -> bool:
    """Ask one question; on an exhausted quota or overloaded model, say so and return False."""
    try:
        print_turn(agent.ask(question), trace)
        return True
    except QuotaExhaustedError as exc:
        print(f"\n!!! QUOTA EXHAUSTED: {exc}\n", file=sys.stderr)
        return False
    except ModelOverloadedError as exc:
        print(f"\n!!! MODEL OVERLOADED: {exc}\n", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="NovaMart internal support assistant")
    parser.add_argument("-q", "--question", help="ask a single question and exit")
    parser.add_argument("--trace", action="store_true", help="show tool calls and results")
    args = parser.parse_args()

    try:
        agent = SupportAgent.create()  # may embed the policy documents
    except QuotaExhaustedError as exc:
        sys.exit(f"!!! QUOTA EXHAUSTED: {exc}")

    if args.question:
        if not ask_and_print(agent, args.question, args.trace):
            sys.exit(2)
        return

    print("NovaMart support assistant. Commands: /reset (new conversation), /trace, /quit")
    trace = args.trace
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/reset":
            agent.reset()
            print("(conversation cleared)")
            continue
        if question == "/trace":
            trace = not trace
            print(f"(trace {'on' if trace else 'off'})")
            continue
        ask_and_print(agent, question, trace)


if __name__ == "__main__":
    main()
