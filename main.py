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
    "18": {
        "name": "Chinese Identity in American AI Models",
        "description": "Reverse of the Claude Identity study: asks Western models '你是什么模型' (no system prompt) and measures how often they claim to be a Chinese model",
        "run": "studies.chinese_identity.main",
    },
    "19": {
        "name": "Claude Identity 2 (Chinese models, asked in English)",
        "description": "Follow-up to Claude Identity: asks Chinese models 'What model are you?' (no system prompt, model default temp) and records who they say they are",
        "run": "studies.claude_identity_2.main",
    },
    "20": {
        "name": "Chinese Identity Over Time",
        "description": "Longitudinal Chinese Identity study: asks a chronological lineage of each Western maker's models '你是什么模型' and plots how often they claim a Chinese identity against real OpenRouter release dates, to show when the behavior emerged",
        "run": "studies.chinese_identity_over_time.main",
    },
    "21": {
        "name": "Western Identity Over Time",
        "description": "Mirror of Chinese Identity Over Time: asks a chronological lineage of each Chinese maker's models 'What model are you?' and plots how often they claim a Western (American) identity against real OpenRouter release dates",
        "run": "studies.western_identity_over_time.main",
    },
    "22": {
        "name": "Poople (Word Ladder to 'poop')",
        "description": "A Wordle-adjacent word ladder: change one letter at a time (every step a valid word) to reach 'poop'. Part 1 BFS-es out from poop to find the minimum steps and all optimal ladders for every 4-letter word; Part 2 tests how well LLMs solve it.",
        "run": "studies.poople.main",
    },
    "23": {
        "name": "Poople Coding (LLMs write a solver)",
        "description": "Coding benchmark: each reasoning model gets one shot to write a Python program that solves Poople optimally. We execute each program against a difficulty-stratified battery of words and grade it for legality and optimality against the BFS oracle.",
        "run": "studies.poople_coding.main",
    },
    "24": {
        "name": "Chief Keef Framing Effects",
        "description": "Tests whether LLMs feel differently about Chief Keef depending on how his background is framed (positive / none / negative), with the closing question held fixed; an LLM judge scores favorability 0-10 per arm",
        "run": "studies.chief_keef_framing.main",
    },
    "25": {
        "name": "World Cup Final Prediction",
        "description": "Predicts the 2026 World Cup final from teams' in-tournament performance across five cups (2010-2026). A Dixon-Coles Poisson goals model and a margin-of-victory Elo rating, opponent-adjusted and shrinkage-regularised, evaluated by leave-one-cup-out cross-validation (including how they'd have called every past final) with knockout ties resolved into a winner via a data-validated extra-time/penalty split.",
        "run": "studies.world_cup_final_prediction.main",
    },
    "26": {
        "name": "Surgeon Riddle (gender pattern-match)",
        "description": "Tests whether LLMs mechanically pattern-match the classic 'surgeon is the mother' riddle: when the driving parent is flipped to a woman, do models still answer 'the doctor is his mother' (impossible — she's the driver) instead of the father?",
        "run": "studies.surgeon_riddle.main",
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
