import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import run_agent


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}\n")
        answer = run_agent(query)
        print(f"\n{answer}\n")
        return

    print("Sistema de recomendação Nintendo Switch: escreve 'sair' para terminar.\n")
    while True:
        query = input("Tu: ").strip()
        if query.lower() in ("sair", "exit", "quit"):
            break
        if not query:
            continue
        print()
        answer = run_agent(query)
        print(f"\nAssistente: {answer}\n")


if __name__ == "__main__":
    main()
