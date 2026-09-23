from pathlib import Path

from new_students.neural_net import StudentsNet
from new_students.trainer import Trainer


def main():
    project_root = Path(__file__).resolve().parent.parent

    dataset_file = project_root / "student_dataset.csv"
    model_file = project_root / "students3_model.json"

    print("Starting neural network training...")
    print()

    network = StudentsNet(
        learning_rate=0.01,
        seed=42
    )

    trainer = Trainer(network)

    trainer.train(
        filename=dataset_file,
        epochs=10,
        batch_size=1000
    )

    network.save(model_file)

    print()
    print("Training completed.")
    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()