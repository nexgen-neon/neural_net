import csv


NUM_RECORDS = 1_000_000
OUTPUT_FILE = "student_dataset.csv"


def generate_dataset(
    filename=OUTPUT_FILE,
    num_records=NUM_RECORDS
):
    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "hours_studied",
            "hours_slept",
            "result",
            "grade"
        ])

        for i in range(num_records):

            hours_studied = (i * 7 % 121) / 10
            hours_slept = 4 + ((i * 11) % 61) / 10

            score = (
                hours_studied * 0.7
                + hours_slept * 0.3
            )

            # Grade
            if score >= 7.0:
                grade = "A"
            elif score >= 5.5:
                grade = "B"
            else:
                grade = "C"

            # Pass / Fail
            if score >= 4.0 and hours_studied >= 4:
                result = "passed"
            else:
                result = "failed"

            writer.writerow([
                hours_studied,
                hours_slept,
                result,
                grade
            ])


if __name__ == "__main__":

    print("Generating dataset...")

    generate_dataset()

    print(
        f"Dataset created: {OUTPUT_FILE}"
    )

    print(
        f"Records: {NUM_RECORDS}"
    )