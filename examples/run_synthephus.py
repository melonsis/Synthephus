import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

# To allow importing library code in this example script, we dynamically add the project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from synp_mechanisms.synthephus_mwem_pgm import mwem_plain, synthephus_mwem_pgm  # noqa: E402


def prepare_stream_folder(source_csv: Path, domain_json: Path, target_dir: Path, timestamps: int) -> None:
    """Split a single CSV into multi-timestamp sample data for quick experimentation."""

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with source_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Source CSV is empty, cannot construct sample input")

    header, data_rows = rows[0], rows[1:]
    chunk_size = max(math.ceil(len(data_rows) / timestamps), 1)

    for idx in range(timestamps):
        start = idx * chunk_size
        end = start + chunk_size
        chunk = data_rows[start:end]
        if not chunk:
            chunk = data_rows[-chunk_size:]
        out_path = target_dir / f"real_{idx + 1}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(header)
            writer.writerows(chunk)

    shutil.copy(domain_json, target_dir / "domain.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthephus example runner")
    parser.add_argument("--timestamps", type=int, default=5, help="Number of sample timestamps")
    parser.add_argument("--window", type=int, default=3, help="Sliding window size w")
    parser.add_argument("--epsilon", type=float, default=3.0, help="Total privacy budget ε")
    parser.add_argument("--rounds", type=int, default=30, help="MWEM rounds T per timestamp")
    parser.add_argument(
        "--source_csv",
        type=Path,
        default=ROOT_DIR / "data" / "colorado.csv",
        help="Original CSV for constructing sample timestamp data",
    )
    parser.add_argument(
        "--domain_json",
        type=Path,
        default=ROOT_DIR / "data" / "colorado.json",
        help="Domain.json corresponding to source CSV",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=ROOT_DIR / "examples" / "demo_stream",
        help="Cache directory for generated real_x.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT_DIR / "examples" / "demo_results",
        help="Output directory for synthesis results",
    )
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="Also run mwem_plain for comparison",
    )
    args = parser.parse_args()

    prepare_stream_folder(args.source_csv, args.domain_json, args.workdir, args.timestamps)

    result_path, log_path = synthephus_mwem_pgm(
        input_folder=str(args.workdir),
        epsilon=args.epsilon,
        w=args.window,
        timestamp_exp=args.timestamps,
        T=args.rounds,
        domain_path=str(args.workdir / "domain.json"),
        output_dir=str(args.output_dir),
        verbose=True,
    )

    print(f"Synthephus result: {result_path}")
    if log_path:
        print(f"Detailed log: {log_path}")

    if args.run_baseline:
        baseline_path = mwem_plain(
            input_folder=str(args.workdir),
            epsilon=args.epsilon,
            w=args.window,
            timestamp_exp=args.timestamps,
            T=args.rounds,
            domain_path=str(args.workdir / "domain.json"),
            output_dir=str(args.output_dir),
        )
        print(f"mwem_plain baseline result: {baseline_path}")


if __name__ == "__main__":
    main()
