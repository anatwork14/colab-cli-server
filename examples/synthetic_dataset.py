"""Generates synthetic tabular regression dataset for Colab E2E test."""
import csv
from pathlib import Path


def generate_synthetic_dataset(output_path: Path, num_samples: int = 100) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feat1", "feat2", "feat3", "feat4", "target"])
        for i in range(num_samples):
            f1 = float(i) * 0.1
            f2 = float(i % 10) * 0.5
            f3 = -float(i) * 0.05
            f4 = float((i * 3) % 7)
            target = 2.0 * f1 - 1.5 * f2 + 0.5 * f3 + 0.1 * f4 + 0.25
            writer.writerow([round(f1, 4), round(f2, 4), round(f3, 4), round(f4, 4), round(target, 4)])

    return output_path


if __name__ == "__main__":
    out = Path(__file__).parent / "synthetic_data.csv"
    generate_synthetic_dataset(out)
    print(f"Generated synthetic dataset at: {out}")
