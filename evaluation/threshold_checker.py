# evaluation/threshold_checker.py

import json
import sys


MIN_ANSWER_RELEVANCY = 0.75


def check_thresholds():
    """
    Fail build if evaluation scores
    drop below acceptable threshold.
    """

    try:
        with open("evaluation_results.json", "r") as f:
            results = json.load(f)

    except FileNotFoundError:
        print("evaluation_results.json not found.")
        sys.exit(1)

    answer_relevancy = results.get("answer_relevancy", 0)

    print(f"Answer Relevancy Score: {answer_relevancy}")

    if answer_relevancy < MIN_ANSWER_RELEVANCY:
        print("FAILED: Answer relevancy below threshold.")
        sys.exit(1)

    print("PASSED: Evaluation thresholds satisfied.")


if __name__ == "__main__":
    check_thresholds()