# Synthephus: High-Utility w-Event Private Synthetic Data Publishing via Adaptive Budget Allocation

## Introduction

This code base contains the implementation of **Synthephus**, a novel approach for generating high-utility differentially private synthetic data over streaming w-event databases with adaptive budget allocation. Synthephus extends state-of-the-art PGM-based (Probabilistic Graphical Model) data synthesis mechanisms to handle dynamic data streams while efficiently managing privacy budgets across timestamps.

The repository includes two implementations based on different PGM synthesis mechanisms:
- **Synthephus-AIM**: Based on the Adaptive and Iterative Mechanism (AIM)
- **Synthephus-MWEM**: Based on the Multiplicative Weights Exponential Mechanism with PGM (MWEM+PGM)

For more details on Private-PGM, please visit [Private-PGM](https://github.com/ryan112358/private-pgm).

These implementations have additional dependencies: [Ektelo](https://github.com/ektelo/ektelo) and [autograd](https://github.com/HIPS/autograd).

## Key Features

- **Adaptive Budget Allocation**: Dynamically adjusts privacy budget allocation across timestamps based on model complexity growth patterns
- **Sliding Window Privacy**: Implements w-event differential privacy over streaming data with efficient budget reclamation
- **Quality Assurance Mechanism**: Includes automatic model quality comparison and rollback to prevent degradation
- **Dual Implementation**: Provides both AIM-based and MWEM-based variants for different use cases
- **Baseline Comparison**: Includes fixed-budget baseline implementations for performance evaluation

## File Structure

* **synp_mechanisms/** - Contains Synthephus implementations for streaming synthetic data generation
  * `synthephus_aim.py` - Synthephus implementation based on AIM mechanism
  * `synthephus_mwem_pgm.py` - Synthephus implementation based on MWEM+PGM mechanism
* **mechanisms/** - Contains baseline and auxiliary mechanisms
  * `adaptive_grid.py`, `aim.py`, `mst.py`, `mwem+pgm.py` - Base mechanism implementations
  * `cdp2adp.py` - Concentrated differential privacy to approximate DP conversion utilities
  * `gaussian+appgm.py`, `hdmm+appgm.py` - Additional mechanism variants
* **src/** - Core dependencies and libraries
  * **hdmm/** - High-Dimensional Matrix Mechanism utilities
  * **mbi/** - Marginal-based inference framework for PGM operations
* **examples/** - Example scripts and demonstration code
  * `run_synthephus.py` - Main example runner for Synthephus
  * `demo_stream/` - Sample timestamped data directory
  * `demo_results/` - Output directory for synthesis results
* **data/** - Contains datasets and domain specifications

## Installation
1. Before getting started, it is recommended to run the `setup_uv.py` script to configure the `uv` environment:
```bash
python setup_uv.py
```
This script:
- Due to compatibility issues with the autodp library, this project includes a packaged version of autodp and uses this script to modify the uv configuration to import this package.
- Additionally, the script redirects the `uv/cache` directory.
If you choose to skip this step, please ensure that the `autodp` package with a version `<=0.2.0` is installed in your Python environment.


2. This project uses `uv` to manage dependencies and execution environment. On first use, please run:

```bash
uv sync
```

If you do not have `uv` installed, you can install it via pip:

```bash
pip install uv
```


3. Export the `src` directory to your Python path. For example, in PowerShell on Windows:

```powershell
$Env:PYTHONPATH += ";E:\Project\Synthephus_release\src"
```

Or on Linux/Mac:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Synthephus_release/src"
```

## Usage

### Quick Start Example

The easiest way to get started is to use the provided example runner:

```bash
uv run examples/run_synthephus.py --timestamps 5 --window 3 --epsilon 3.0 --rounds 30
```

This will:
1. Split a sample dataset into 5 timestamps
2. Run Synthephus-MWEM with a sliding window of size 3
3. Use a total privacy budget of ε=3.0
4. Execute 30 MWEM rounds per timestamp

### Parameter Reference

- `--timestamps`: Number of timestamps to generate and process
- `--window`: Sliding window size (w parameter for w-event DP)
- `--epsilon`: Total privacy budget (ε)
- `--rounds`: Number of mechanism iterations per timestamp (T parameter)
- `--source_csv`: Path to source CSV file for creating sample data
- `--domain_json`: Path to domain specification JSON file
- `--workdir`: Working directory for temporary timestamp data
- `--output_dir`: Output directory for results
- `--run_baseline`: Flag to also run fixed-budget baseline for comparison

### Running  Synthephus-AIM

To use the AIM-based implementation directly:

```python
from synp_mechanisms.synthephus_aim import synthephus_aim

result_path, log_path = synthephus_aim(
    input_folder="path/to/timestamp/data",
    epsilon=3.0,
    w=3,
    timestamp_exp=5,
    domain_path="path/to/domain.json",
    max_model_size_mb=80.0,
    verbose=True,
    output_dir="results"
)
```

### Running Synthephus-MWEM

To use the MWEM-based implementation directly:

```python
from synp_mechanisms.synthephus_mwem_pgm import synthephus_mwem_pgm

result_path, log_path = synthephus_mwem_pgm(
    input_folder="path/to/timestamp/data",
    epsilon=3.0,
    w=3,
    timestamp_exp=5,
    T=30,
    domain_path="path/to/domain.json",
    max_model_size_mb=25.0,
    verbose=True,
    output_dir="results"
)
```

### Command Line Interface

Both mechanisms can also be run directly from the command line:

```bash
# Synthephus-AIM
uv run synp_mechanisms/synthephus_aim.py \
    --input_folder data/stream \
    --epsilon 3.0 \
    --w 3 \
    --timestamp_exp 5 \
    --mode synthephus \
    --verbose

# Synthephus-MWEM
uv run synp_mechanisms/synthephus_mwem_pgm.py \
    --input_folder data/stream \
    --epsilon 3.0 \
    --w 3 \
    --timestamp_exp 5 \
    --T 30 \
    --mode synthephus \
    --verbose
```

Use `--mode baseline` to run the fixed-budget baseline instead.

## Data Format

### Input Data Files

Each timestamp should be stored as a separate CSV file in the input folder:
- `real_1.csv` - Data for timestamp 1
- `real_2.csv` - Data for timestamp 2
- ...
- `real_T.csv` - Data for timestamp T

### Domain Specification

A `domain.json` file should be provided to specify attribute domains:

```json
{
  "attribute1": 10,
  "attribute2": 5,
  "attribute3": 20
}
```

Where the numbers indicate the domain size (number of possible values) for each attribute.

## Output

The algorithms generate CSV files containing:
- `timestamp`: Current timestamp index
- `allocated_budget`: Privacy budget allocated to this timestamp
- `actual_consumed_budget`: Actual privacy budget consumed
- `eps_remain`: Remaining privacy budget after this timestamp
- `workload_error`: Marginal query error metric
- `cliques_count`: Number of cliques in the synthesized graphical model

Optional detailed logs (when `verbose=True`) provide per-iteration information including selected cliques, budget consumption, and model quality metrics.


## Citation

If you use this code in your research, please cite:

```
[Citation information Citation information is not added since this work is under anonymous review]
```

## License

[License information Citation information is not added since this work is under anonymous review]

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the authors.

