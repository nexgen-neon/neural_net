from pathlib import Path

from grade_drop.neural_net import StudentsNet


def main():

    project_root = Path(__file__).resolve().parent.parent
    model_file = project_root / "students1_model.json"

    network = StudentsNet()

    try:
        network.load(model_file)

    except FileNotFoundError:
        print("Trained model not found.")
        print("Please run training.py first.")
        return

    print("Trained model loaded successfully.")
    print()

    hours_studied = float(
        input("Enter hours studied: ")
    )

    hours_slept = float(
        input("Enter hours slept: ")
    )

    probabilities = network.predict(
        hours_studied,
        hours_slept
    )

    print()

    print("Grade Probability Distribution")
    print("--------------------------------")

    print(f"A: {probabilities['A']:.4f}")
    print(f"B: {probabilities['B']:.4f}")
    print(f"C: {probabilities['C']:.4f}")

    print()

    print("Probability Sum:")
    print(f"{sum(probabilities.values()):.4f}")


if __name__ == "__main__":
    main()