"""
reseairch - Experimental AI Research

Entry point: select a study to run.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

# Registry of available studies
STUDIES = {
    "1": {
        "name": "Claude Identity in Chinese AI Models",
        "description": "Tests whether Chinese AI models claim to be Claude at various temperatures",
        "run": "studies.claude_identity.main",
    },
}


def main():
    print("\n" + "=" * 60)
    print("  reseairch - Experimental AI Research")
    print("=" * 60)
    print("\nAvailable studies:\n")

    for key, study in STUDIES.items():
        print(f"  [{key}] {study['name']}")
        print(f"      {study['description']}")

    print(f"\n  [q] Quit")

    choice = input("\nSelect a study: ").strip().lower()

    if choice == "q":
        print("Bye!")
        return

    if choice not in STUDIES:
        print(f"Invalid choice: {choice}")
        return

    # Dynamically import and run the study
    study = STUDIES[choice]
    module_path = study["run"]
    module = __import__(module_path, fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
