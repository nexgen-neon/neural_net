import csv


def grade_to_target(grade):

    grade = grade.strip().upper()

    if grade == "A":
        return 1.0

    if grade == "B":
        return 0.5

    if grade == "C":
        return 0.0

    raise ValueError(
        f"Unknown grade: {grade}"
    )


def result_to_target(result):

    result = result.strip().lower()

    if result == "passed":
        return 1.0

    if result == "failed":
        return 0.0

    raise ValueError(
        f"Unknown result: {result}"
    )


def read_batches(filename, batch_size=1000):

    with open(
        filename,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        batch = []

        for row in reader:

            hours_studied = float(
                row["hours_studied"]
            )

            hours_slept = float(
                row["hours_slept"]
            )

            result = row["result"]

            grade = row["grade"]

            result_target = result_to_target(
                result
            )

            grade_target = grade_to_target(
                grade
            )

            batch.append(
                (
                    [hours_studied, hours_slept],
                    result_target,
                    grade_target
                )
            )

            if len(batch) == batch_size:

                yield batch

                batch = []

        if batch:
            yield batch