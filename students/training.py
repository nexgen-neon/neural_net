from students.neural_network import StudentsNet
from students.trainer import Trainer


def main():

    dataset_file = "student_dataset.csv"
    model_file = "students_model.json"

    print("Starting neural network training...")
    print()

    network = StudentsNet(
        learning_rate=0.01
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