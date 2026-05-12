# src/quality/validation.py
import re

import pandas as pd
import great_expectations as gx
from great_expectations import expectations as exp
from great_expectations.core.expectation_suite import ExpectationSuite


def build_patient_expectation_suite(tmp_dir: str = None) -> ExpectationSuite:
    """
    Tao expectation suite cho anonymized patient data.
    Uses a temporary FileDataContext to handle the save operation.
    """
    import tempfile

    work_dir = tmp_dir or tempfile.mkdtemp()
    ctx = gx.get_context(mode="file", project_root_dir=work_dir)
    suite = ExpectationSuite(name="patient_data_suite")
    added = ctx.suites.add(suite)

    expectations = [
        (exp.ExpectColumnValuesToNotBeNull, {"column": "patient_id"}),
        (exp.ExpectColumnValueLengthsToEqual, {"column": "cccd", "value": 12}),
        (exp.ExpectColumnValuesToBeBetween,
         {"column": "ket_qua_xet_nghiem", "min_value": 0, "max_value": 50}),
        (exp.ExpectColumnValuesToBeInSet,
         {"column": "benh", "value_set": ["Tiểu đường", "Huyết áp cao", "Tim mạch", "Khỏe mạnh"]}),
        (exp.ExpectColumnValuesToMatchRegex,
         {"column": "email", "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}),
        (exp.ExpectColumnValuesToBeUnique, {"column": "patient_id"}),
    ]

    for cls, kwargs in expectations:
        added.add_expectation(cls(**kwargs))

    print(f"Expectation suite created with {len(added.expectations)} expectations:")
    for e in added.expectations:
        print(f"  - {e.expectation_type}")
    return added


def validate_anonymized_data(filepath: str, original_row_count: int = None) -> dict:
    """
    Validate anonymized data.
    Tra ve dict: {"success": bool, "failed_checks": list, "stats": dict}
    """
    df = pd.read_csv(filepath)
    results = {
        "success": True,
        "failed_checks": [],
        "stats": {
            "total_rows": len(df),
            "columns": list(df.columns)
        }
    }

    # Check 1: Khong con CCCD goc dang so thuan tuy (11-12 chu so)
    # Sau anonymization, cccd phai la fake hoac masked
    cccd_pattern = re.compile(r"^\d{11,12}$")
    original_cccd_count = df["cccd"].astype(str).str.match(cccd_pattern).sum()
    if original_cccd_count > 0:
        results["success"] = False
        results["failed_checks"].append(
            f"Found {original_cccd_count} original-format CCCD values in anonymized data"
        )

    # Check 2: Khong co null values trong cac cot quan trong
    important_cols = ["patient_id", "benh", "ket_qua_xet_nghiem"]
    for col in important_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                results["success"] = False
                results["failed_checks"].append(
                    f"Column '{col}' has {null_count} null values"
                )

    # Check 3: So rows phai bang original
    if original_row_count is not None:
        if len(df) != original_row_count:
            results["success"] = False
            results["failed_checks"].append(
                f"Row count mismatch: got {len(df)}, expected {original_row_count}"
            )

    return results
