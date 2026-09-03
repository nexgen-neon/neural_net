import csv


DEFAULT_BATCH_SIZE = 1000


def grade_to_targets(grade):

    if grade == "A":
        return 1.0, 0.0, 0.0

    if grade == "B":
        return 0.0, 1.0, 0.0

    if grade == "C":
        return 0.0, 0.0, 1.0

    raise ValueError(
        f"Unknown grade: {grade}"
    )


def read_batches(
    filename,
    batch_size=DEFAULT_BATCH_SIZE
):

    with open(
        filename,
        "r",
        newline=""
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

            grade = row["grade"]

            (
                grade_a_target,
                grade_b_target,
                grade_c_target
            ) = grade_to_targets(grade)

            record = (
                [hours_studied, hours_slept],
                grade_a_target,
                grade_b_target,
                grade_c_target
            )

            batch.append(record)

            if len(batch) == batch_size:

                yield batch

                batch = []

        if batch:
            yield batch