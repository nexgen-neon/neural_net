DEFAULT_XOR_TABLE = [
    {
        "x1": 0,
        "x2": 0,
        "expected": 0,
    },
    {
        "x1": 0,
        "x2": 1,
        "expected": 1,
    },
    {
        "x1": 1,
        "x2": 0,
        "expected": 1,
    },
    {
        "x1": 1,
        "x2": 1,
        "expected": 0,
    },
]


def validate_truth_table(rows):
    """
    Validate the user's truth table.
    """

    if len(rows) != 4:
        return (
            False,
            "Truth table must contain exactly 4 rows.",
        )

    combinations = set()

    for row in rows:

        x1 = row["x1"]
        x2 = row["x2"]
        expected = row["expected"]

        if x1 not in (0, 1):
            return (
                False,
                "X1 must contain only 0 or 1.",
            )

        if x2 not in (0, 1):
            return (
                False,
                "X2 must contain only 0 or 1.",
            )

        if expected not in (0, 1):
            return (
                False,
                "Expected output must contain only 0 or 1.",
            )

        combinations.add((x1, x2))

    if len(combinations) != 4:
        return (
            False,
            "All four input combinations are required.",
        )

    return (
        True,
        "Truth table is valid.",
    )


def is_xor_table(rows):
    """
    Check whether the expected outputs
    represent the XOR operation.
    """

    expected_outputs = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0,
    }

    for row in rows:

        key = (
            row["x1"],
            row["x2"],
        )

        if row["expected"] != expected_outputs[key]:
            return False

    return True