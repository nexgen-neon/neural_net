from students.neural_network import StudentsNet


def main():

    model_file = "students_model.json"

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

    (
        result_output,
        result,
        grade_output,
        grade
    ) = network.classification(
        hours_studied,
        hours_slept
    )

    print()
    print("Prediction")
    print("--------------------")

    print(
        f"Result probability: {result_output:.4f}"
    )

    print(
        f"Result: {result}"
    )

    print(
        f"Grade probability: {grade_output:.4f}"
    )

    print(
        f"Grade: {grade}"
    )


if __name__ == "__main__":
    main()