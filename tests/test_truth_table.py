from xor_neural_network.truth_table import (
    DEFAULT_XOR_TABLE,
    is_xor_table,
    validate_truth_table,
)


def test_default_truth_table_is_valid():
    valid, message = validate_truth_table(
        DEFAULT_XOR_TABLE
    )

    assert valid is True


def test_default_table_is_xor():
    assert is_xor_table(
        DEFAULT_XOR_TABLE
    ) is True


def test_invalid_truth_table():
    invalid_table = [
        {
            "x1": 0,
            "x2": 0,
            "expected": 0,
        },
        {
            "x1": 0,
            "x2": 1,
            "expected": 0,
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

    assert is_xor_table(
        invalid_table
    ) is False