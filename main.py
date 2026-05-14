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
    "2": {
        "name": "Gender Bias in LLM Moral Reasoning",
        "description": "Tests whether LLMs show gender-based differences in moral dilemma responses (Likert scale)",
        "run": "studies.biases.main",
    },
    "3": {
        "name": "Political Sycophancy by State",
        "description": "Tests whether LLMs shift political opinions based on which U.S. state the user is from",
        "run": "studies.political_sycophancy.main",
    },
    "4": {
        "name": "Emergent Values (Exchange Rates)",
        "description": "Measures implied LLM value systems via pairwise comparisons across countries, religions, etc.",
        "run": "studies.emergent_values.main",
    },
    "5": {
        "name": "HLE Performance Sycophancy",
        "description": "Tests whether LLMs perform differently on HLE questions based on which U.S. state the user is from",
        "run": "studies.hle_sycophancy.main",
    },
    "6": {
        "name": "Trolley Problem by State",
        "description": "Tests whether LLMs answer the trolley problem differently based on which U.S. state the user is from",
        "run": "studies.trolley_problem.main",
    },
    "7": {
        "name": "Dictator Removal (Baby Time Travel)",
        "description": "Tests whether LLMs would kill historical dictators as babies, comparing across dictators and models",
        "run": "studies.dictator_removal.main",
    },
    "8": {
        "name": "God Belief by State",
        "description": "Tests whether LLMs change their answer to 'does god exist?' based on which U.S. state the user is from",
        "run": "studies.god_by_state.main",
    },
    "9": {
        "name": "God Belief by Country",
        "description": "Tests whether LLMs change their answer to 'does god exist?' based on which country the user is from",
        "run": "studies.god_by_country.main",
    },
    "10": {
        "name": "God Belief by Name",
        "description": "Tests whether LLMs change their answer to 'does god exist?' based on the user's name (race/gender proxy)",
        "run": "studies.god_by_name.main",
    },
    "11": {
        "name": "Sentencing Bias by Name",
        "description": "Tests whether LLMs assign different sentences based on defendant name (race/ethnicity/gender proxy)",
        "run": "studies.sentencing_bias.main",
    },
    "12": {
        "name": "Salary Bias by Name",
        "description": "Tests whether LLMs recommend different salaries based on candidate name (race/ethnicity/gender proxy)",
        "run": "studies.salary_bias.main",
    },
    "13": {
        "name": "Hate Speech Detection Bias",
        "description": "Tests whether LLMs flag identical hate speech differently based on which demographic group is referenced",
        "run": "studies.hate_speech_detection.main",
    },
    "14": {
        "name": "Favorite Company (LLM Preferences)",
        "description": "Asks each LLM which company it would want to be built by, why, and who it would want as its creator",
        "run": "studies.favorite_company.main",
    },
    "15": {
        "name": "Question Framing Effects",
        "description": "Tests whether psychological framing (flattery, pressure, challenge, etc.) affects LLM accuracy on HLE questions",
        "run": "studies.question_framing.main",
    },
    "16": {
        "name": "Idea Priming (Good vs. Bad)",
        "description": "Tests whether asking 'why is this a good idea?' vs 'why is this a bad idea?' shifts LLM 1-10 quality scores for the same idea",
        "run": "studies.idea_priming.main",
    },
    "17": {
        "name": "Cognitive Biases",
        "description": "Tests classic human cognitive biases in LLMs (anchoring, etc.) via paired control/treatment scenarios with numeric drift",
        "run": "studies.cognitive_biases.main",
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
