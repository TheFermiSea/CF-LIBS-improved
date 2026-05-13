import re
import sys

def validate_metrics(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    expected_metrics = {
        "Hybrid-Union": 0.715,
        "Spectral NNLS": 0.442,
        "Alias": 0.141,
        "Correlation": 0.177,
        "Comb": 0.028
    }

    errors = []
    for method, expected_f1 in expected_metrics.items():
        # Look for the method and its F1 score in the table
        pattern = rf"\| {re.escape(method)} \| ({expected_f1}) \|"
        if not re.search(pattern, content):
            errors.append(f"Metric for {method} not found or incorrect. Expected F1: {expected_f1}")

    if errors:
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("All metrics validated successfully.")
        sys.exit(0)

if __name__ == "__main__":
    validate_metrics("papers/hybrid_union_libs/manuscript.md")
