from config import VALUE_COLUMNS

def validate_value_column(value_column: str) -> None:
    if value_column not in VALUE_COLUMNS:
        raise ValueError(
            f"The selected value column is unavailable, " \
            f"try one of the following: {VALUE_COLUMNS}"
        )