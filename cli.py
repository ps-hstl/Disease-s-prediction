"""
cli.py
------
Command-line interface for disease prediction.

Usage:
    python cli.py
    python cli.py --symptoms "Itching,Skin Rash,Nodal Skin Eruptions"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from predictor import load_artefacts, predict


BANNER = r"""
  ____  _                             ____                _ _      _
 |  _ \(_)___  ___  __ _ ___  ___   |  _ \ _ __ ___  __| (_) ___| |_ ___  _ __
 | | | | / __|/ _ \/ _` / __|/ _ \  | |_) | '__/ _ \/ _` | |/ __| __/ _ \| '__|
 | |_| | \__ \  __/ (_| \__ \  __/  |  __/| | |  __/ (_| | | (__| || (_) | |
 |____/|_|___/\___|\__,_|___/\___|  |_|   |_|  \___|\__,_|_|\___|\__\___/|_|

  Multiple Disease Prediction System  |  ML Ensemble: SVM + NB + Random Forest
"""


def interactive_mode(svm_model, nb_model, rf_model, meta):
    all_symptoms = sorted(meta["symptom_index"].keys())
    print("\nAvailable symptoms (first 20 shown):")
    for i, s in enumerate(all_symptoms[:20], 1):
        print(f"  {i:>3}. {s}")
    print(f"  … and {len(all_symptoms) - 20} more.\n")

    raw = input("Enter symptoms separated by commas:\n> ").strip()
    if not raw:
        print("No symptoms entered. Exiting.")
        return

    symptoms_list = [s.strip() for s in raw.split(",") if s.strip()]
    result = predict(symptoms_list, svm_model, nb_model, rf_model, meta)
    print_result(result)


def print_result(result):
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  FINAL PREDICTION  →  {result['final_prediction'].upper()}")
    print(sep)
    print(f"  SVM           :  {result['svm']}")
    print(f"  Naive Bayes   :  {result['naive_bayes']}")
    print(f"  Random Forest :  {result['random_forest']}")
    print(sep)
    print(f"  Model agreement :  {result['agreement']:.0f}%")
    print(f"  RF confidence   :  {result['rf_confidence']}%")
    if result["matched_symptoms"]:
        print(f"\n  Matched symptoms  ({len(result['matched_symptoms'])}):")
        for s in result["matched_symptoms"]:
            print(f"    ✔  {s}")
    if result["unmatched_symptoms"]:
        print(f"\n  ⚠ Unrecognised symptoms ({len(result['unmatched_symptoms'])}):")
        for s in result["unmatched_symptoms"]:
            print(f"    ✖  {s}")
    print(f"\n  ⚠  Disclaimer: For educational use only. Consult a doctor.\n")


def main():
    parser = argparse.ArgumentParser(description="Multiple Disease Prediction CLI")
    parser.add_argument(
        "--symptoms", type=str, default=None,
        help='Comma-separated symptoms, e.g. "Itching,Skin Rash"'
    )
    args = parser.parse_args()

    print(BANNER)
    print("Loading models …", end=" ", flush=True)
    try:
        svm_model, nb_model, rf_model, encoder, meta = load_artefacts()
        print("Done.\n")
    except Exception as exc:
        print(f"\nError loading models: {exc}")
        print("Run   python utils/train_models.py   first.")
        sys.exit(1)

    if args.symptoms:
        symptoms_list = [s.strip() for s in args.symptoms.split(",") if s.strip()]
        result = predict(symptoms_list, svm_model, nb_model, rf_model, meta)
        print_result(result)
    else:
        interactive_mode(svm_model, nb_model, rf_model, meta)


if __name__ == "__main__":
    main()
