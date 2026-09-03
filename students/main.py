from students.neural_network import StudentsNet
from students.trainer import Trainer


def main():

    filename = "student_dataset.csv"

    network = StudentsNet(
        learning_rate=0.01
    )

    trainer = Trainer(network)

    print("Starting training...")
    print()

    trainer.train(
        filename,
        epochs=10,
        batch_size=1000
    )

    print()
    print("Training completed.")

    print()
    print("Enter student details")

    hours_studied = float(
        input("Hours studied: ")
    )

    hours_slept = float(
        input("Hours slept: ")
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

    print(
        f"Pass/Fail output: "
        f"{result_output:.4f}"
    )

    print(
        f"Result: {result}"
    )

    print()

    print(
        f"Grade output: "
        f"{grade_output:.4f}"
    )

    print(
        f"Grade: {grade}"
    )


if __name__ == "__main__":
    main()