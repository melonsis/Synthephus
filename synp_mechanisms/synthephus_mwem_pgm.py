import argparse
import csv
import datetime
import os
import pickle
import sys
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import itertools
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mbi import Dataset, GraphicalModel, FactoredInference
from mechanisms.cdp2adp import cdp_rho
from scipy import sparse
from scipy.special import softmax

"""Synthephus streaming synthesis implementation based on mwem+pgm."""


# =============================
# Utility Functions
# =============================


EPS_FLOOR = 1e-12
DEFAULT_DELTA = 1e-9
ALPHA = 0.9


def _worst_approximated(
    workload_answers: Dict[Tuple[str, ...], np.ndarray],
    est: GraphicalModel,
    workload: Iterable[Tuple[str, ...]],
    eps: float,
    penalty: bool = True,
    bounded: bool = False,
) -> Tuple[str, ...]:
    """Exponential mechanism to select the clique with the worst approximation error, used in MWEM iteration."""

    candidate_list = list(workload)
    errors = np.array([])
    for cl in candidate_list:
        bias = est.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = est.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xest).sum() - bias)
    sensitivity = 2.0 if bounded else 1.0
    prob = softmax(0.5 * eps / sensitivity * (errors - errors.max()))
    key = np.random.choice(len(errors), p=prob)
    return candidate_list[key]


def _compute_workload_error(
    data: Dataset, est: GraphicalModel, workload: Iterable[Tuple[str, ...]]
) -> float:
    """Compute workload error using the original mwem+pgm evaluation method."""

    errors: List[float] = []
    for proj in workload:
        X = data.project(proj).datavector().astype(float)
        Y = est.project(proj).datavector().astype(float)
        if X.sum() <= 0 or Y.sum() <= 0:
            errors.append(0.0)
            continue
        e = 0.5 * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(float(e))
    return float(np.mean(errors)) if errors else 0.0


def _clique_set_size(domain, cliques: Iterable[Tuple[str, ...]]) -> float:
    """Return the size of the graphical model corresponding to the clique set, avoiding redundant code."""

    cliques_list = list(cliques)
    if not cliques_list:
        return 0.0
    gm = GraphicalModel(domain, cliques_list)
    return float(gm.size)


# =============================
# MWEM Runner
# ==============================


@dataclass
class MWEMState:
    """Encapsulate MWEM iteration state for stage switching and replay."""

    measurements: List[Any]
    cliques: List[Tuple[str, ...]]
    est: GraphicalModel
    iterations_completed: int


class MWEMPGMRunner:
    """Data synthesis controller for a single timestamp, providing split-run capability."""

    def __init__(
        self,
        data: Dataset,
        workload: List[Tuple[str, ...]],
        total_rounds: int,
        pgm_iters: int = 1000,
        noise: str = "gaussian",
        delta: float = DEFAULT_DELTA,
        bounded: bool = False,
        maxsize_mb: float = 25.0,
        verbose: bool = False,
    ) -> None:
        self.data = data
        self.workload = workload
        self.workload_answers = {
            cl: data.project(cl).datavector() for cl in workload
        }
        self.total_rounds = max(total_rounds, 1)
        self.engine = FactoredInference(
            data.domain, log=False, iters=pgm_iters, warm_start=True
        )
        self.noise = noise
        self.delta = delta if delta > 0 else DEFAULT_DELTA
        self.bounded = bounded
        self.maxsize_mb = maxsize_mb
        self.alpha = ALPHA
        self.domain = data.domain
        self.total = data.records if bounded else None
        self.marginal_sensitivity = 2.0 if bounded else 1.0
        self.verbose = verbose
        self._logs: List[Dict[str, Any]] = []

    def _model_size_mb(self, cliques: List[Tuple[str, ...]]) -> float:
        gm = GraphicalModel(self.domain, cliques) if cliques else None
        size_bytes = gm.size * 8 if gm else 0.0
        return float(size_bytes / (2 ** 20))

    def _privatization_params(self, eps_round: float) -> Tuple[float, float]:
        eps_round = max(eps_round, EPS_FLOOR)
        if self.noise.lower() == "laplace":
            sigma = self.marginal_sensitivity / (self.alpha * eps_round)
            exp_eps = max((1.0 - self.alpha) * eps_round, EPS_FLOOR)
            return exp_eps, sigma
        if self.noise.lower() == "gaussian":
            delta_step = max(self.delta / self.total_rounds, 1e-12)
            rho_step = cdp_rho(eps_round, delta_step)
            if rho_step <= 0:
                rho_step = EPS_FLOOR
            sigma = np.sqrt(0.5 / (self.alpha * rho_step))
            exp_eps = np.sqrt(8.0 * (1.0 - self.alpha) * rho_step)
            return exp_eps, sigma
        raise ValueError(f"Unsupported noise type: {self.noise}")

    def _sample_noise(self, size: int, sigma: float) -> np.ndarray:
        if self.noise.lower() == "laplace":
            return np.random.laplace(loc=0.0, scale=sigma, size=size)
        return np.random.normal(loc=0.0, scale=sigma, size=size)

    def _log(self, iteration: int, clique: Optional[Tuple[str, ...]], budget: float, est: GraphicalModel) -> None:
        if not self.verbose:
            return
        model_mb = float(est.size * 8 / (2 ** 20)) if hasattr(est, "size") else 0.0
        self._logs.append(
            {
                "iteration": iteration,
                "clique": list(clique) if clique else None,
                "budget": float(budget),
                "model_size_mb": model_mb,
                "workload_error": _compute_workload_error(self.data, est, self.workload),
            }
        )

    def consume_logs(self) -> List[Dict[str, Any]]:
        logs = self._logs
        self._logs = []
        return logs

    def run(
        self,
        state: Optional[MWEMState],
        per_round_budgets: List[float],
        temp_save_path: Optional[str] = None,
        temp_load_path: Optional[str] = None,
    ) -> Tuple[MWEMState, List[Tuple[str, ...]], float]:
        """Execute multiple MWEM rounds according to the given budget sequence, return new state, newly added cliques, and actual consumed budget."""

        if state is None:
            base_measurements: List[Any] = []
            base_cliques: List[Tuple[str, ...]] = []
            iterations_completed = 0
            est = self.engine.estimate(base_measurements, self.total)
        else:
            base_measurements = list(state.measurements)
            base_cliques = list(state.cliques)
            iterations_completed = state.iterations_completed
            est = state.est

        if temp_load_path and os.path.exists(temp_load_path):
            with open(temp_load_path, "rb") as f:
                est = pickle.load(f)

        new_cliques: List[Tuple[str, ...]] = []
        consumed_budget = 0.0

        for local_idx, eps_round in enumerate(per_round_budgets, start=1):
            iteration_number = iterations_completed + local_idx
            allowed_size_mb = (
                self.maxsize_mb * iteration_number / self.total_rounds
            )
            if eps_round <= EPS_FLOOR:
                self._log(iteration_number, None, eps_round, est)
                continue

            candidate_pool = [
                cl
                for cl in self.workload
                if self._model_size_mb(base_cliques + new_cliques + [cl])
                <= allowed_size_mb
            ]
            if not candidate_pool:
                self._log(iteration_number, None, eps_round, est)
                continue

            exp_eps, sigma = self._privatization_params(eps_round)
            chosen_clique = _worst_approximated(
                self.workload_answers,
                est,
                candidate_pool,
                exp_eps,
                penalty=True,
                bounded=self.bounded,
            )
            n = self.domain.size(chosen_clique)
            x = self.data.project(chosen_clique).datavector()
            noisy_measurement = x + self._sample_noise(n, sigma)
            Q = sparse.eye(n)
            base_measurements.append((Q, noisy_measurement, 1.0, chosen_clique))
            new_cliques.append(chosen_clique)
            est = self.engine.estimate(base_measurements, self.total)
            consumed_budget += eps_round
            self._log(iteration_number, chosen_clique, eps_round, est)

        updated_state = MWEMState(
            measurements=base_measurements,
            cliques=base_cliques + new_cliques,
            est=est,
            iterations_completed=iterations_completed + len(per_round_budgets),
        )

        if temp_save_path:
            with open(temp_save_path, "wb") as f:
                pickle.dump(est, f)

        return updated_state, new_cliques, consumed_budget


# =============================
# Main Algorithm and Baseline
# =============================


def mwem_plain(
    input_folder: str,
    epsilon: float,
    w: int,
    timestamp_exp: int,
    T: int,
    domain_path: Optional[str] = None,
    workload: Optional[List[Tuple[str, ...]]] = None,
    pgm_iters: int = 1000,
    noise: str = "gaussian",
    verbose: bool = False,
    output_dir: str = "results",
    delta: float = DEFAULT_DELTA,
    max_model_size_mb: float = 25.0,
) -> str:
    """Baseline approach: fixed budget eps/w for each timestamp, no dynamic adjustment."""

    domain_path = domain_path or os.path.join(input_folder, "domain.json")
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Missing domain.json: {domain_path}")

    first_file = os.path.join(input_folder, "real_1.csv")
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"Missing input file: {first_file}")

    first_ds = Dataset.load(first_file, domain_path)
    workload = workload or list(itertools.combinations(first_ds.domain, 2))

    os.makedirs(output_dir, exist_ok=True)
    utc_tag = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result_path = os.path.join(output_dir, f"synp_mwem_plain_result_{utc_tag}.csv")

    per_timestamp_alloc = epsilon / float(max(w, 1))

    active_window: Deque[Tuple[int, float]] = deque()
    active_sum = 0.0

    results: List[Dict[str, Any]] = []
    model_history: Dict[int, GraphicalModel] = {}
    cl_history: Dict[int, List[Tuple[str, ...]]] = {}
    actual_consumed: Dict[int, float] = {}

    for t in tqdm(range(1, timestamp_exp + 1), desc="Baseline timestamp progress"):
        while active_window and (t - active_window[0][0]) >= w:
            _, removed = active_window.popleft()
            active_sum -= removed
        eps_remain = max(epsilon - active_sum, 0.0)

        data_path = os.path.join(input_folder, f"real_{t}.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing input file: {data_path}")
        data = Dataset.load(data_path, domain_path)

        runner = MWEMPGMRunner(
            data,
            workload,
            total_rounds=T,
            pgm_iters=pgm_iters,
            noise=noise,
            delta=delta,
            maxsize_mb=max_model_size_mb,
            verbose=verbose,
        )

        round_budget = per_timestamp_alloc / float(max(T, 1))
        state, _, consumed = runner.run(
            None, [round_budget] * max(T, 1)
        )

        actual_consumed[t] = consumed
        cl_history[t] = list(state.cliques)
        model_history[t] = state.est

        active_window.append((t, consumed))
        active_sum += consumed
        eps_after = max(epsilon - active_sum, 0.0)

        error_val = _compute_workload_error(data, state.est, workload)
        results.append(
            {
                "timestamp": t,
                "allocated_budget": per_timestamp_alloc,
                "actual_consumed_budget": consumed,
                "eps_remain": eps_after,
                "workload_error": error_val,
                "cliques_count": len(state.cliques),
            }
        )

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "allocated_budget",
                "actual_consumed_budget",
                "eps_remain",
                "workload_error",
                "cliques_count",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return result_path


def synthephus_mwem_pgm(
    input_folder: str,
    epsilon: float,
    w: int,
    timestamp_exp: int,
    T: int,
    domain_path: Optional[str] = None,
    workload: Optional[List[Tuple[str, ...]]] = None,
    pgm_iters: int = 1000,
    noise: str = "gaussian",
    verbose: bool = False,
    output_dir: str = "results",
    delta: float = DEFAULT_DELTA,
    max_model_size_mb: float = 25.0,
) -> Tuple[str, Optional[str]]:
    """Synthephus main workflow: dynamic budget, window management, and quality fallback."""

    domain_path = domain_path or os.path.join(input_folder, "domain.json")
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Missing domain.json: {domain_path}")

    first_file = os.path.join(input_folder, "real_1.csv")
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"Missing input file: {first_file}")

    first_ds = Dataset.load(first_file, domain_path)
    workload = workload or list(itertools.combinations(first_ds.domain, 2))

    os.makedirs(output_dir, exist_ok=True)
    utc_tag = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result_path = os.path.join(output_dir, f"synp_mwem_result_{utc_tag}.csv")
    log_path = (
        os.path.join(output_dir, f"synp_log_{utc_tag}.txt") if verbose else None
    )

    # V2 strategy implementation
    active_window: Deque[Tuple[int, float]] = deque()
    active_sum = 0.0

    model_history: Dict[int, GraphicalModel] = {}
    cl_history: Dict[int, List[Tuple[str, ...]]] = {}
    actual_consumed: Dict[int, float] = {}
    allocated: Dict[int, float] = {}
    half_sizes: Dict[int, float] = {}
    full_sizes: Dict[int, float] = {}

    results: List[Dict[str, Any]] = []

    T_front = T // 2
    T_back = T - T_front

    for t in tqdm(range(1, timestamp_exp + 1), desc="Synthephus timestamp progress"):
        # Window management
        while active_window and (t - active_window[0][0]) >= w:
            _, removed = active_window.popleft()
            active_sum -= removed
        eps_remain_pre = max(epsilon - active_sum, 0.0)

        data_path = os.path.join(input_folder, f"real_{t}.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing input file: {data_path}")
        data = Dataset.load(data_path, domain_path)

        runner = MWEMPGMRunner(
            data,
            workload,
            total_rounds=T,
            pgm_iters=pgm_iters,
            noise=noise,
            delta=delta,
            maxsize_mb=max_model_size_mb,
            verbose=verbose,
        )

        prev_model = model_history.get(t - 1)
        prev_cliques = cl_history.get(t - 1, [])

        # Stage A/B/C: First half
        if t == 1:
            total_alloc = epsilon / float(max(w, 1))
            per_iter_front = total_alloc / float(max(T, 1))
            front_budgets = [per_iter_front] * max(T, 1)
        elif T_front > 0:
            if 2 <= t <= w:
                denom = max(w - t, 1)
                base_front = eps_remain_pre / float(denom)
            else:
                base_front = eps_remain_pre / float(max(w, 1))
            per_iter_front = base_front / float(max(T, 1))
            front_budgets = [per_iter_front] * T_front
        else:
            front_budgets = []

        state_front: Optional[MWEMState] = None
        consumed_front = 0.0
        cl_half = list(prev_cliques)

        if front_budgets:
            state_front, _, consumed_front = runner.run(None, front_budgets)
            cl_half = list(state_front.cliques)

        half_size_current = _clique_set_size(first_ds.domain, cl_half)
        available_after_front = max(eps_remain_pre - consumed_front, 0.0)

        # Second half budget
        if t == 1 or T_back == 0:
            back_budgets = []
        elif 2 <= t <= w:
            denom = max(w - t, 1)
            size_prev = max(_clique_set_size(first_ds.domain, prev_cliques), 1.0)
            size_temp = max(half_size_current, 1.0)
            base_back = (2.0 * size_temp / size_prev) * eps_remain_pre / float(denom)
            per_iter_back = base_back / float(max(T, 1))
            back_budgets = [per_iter_back] * T_back
        else:
            window_start = max(1, t - w)
            ratios: List[float] = []
            size_values: List[float] = []
            for hist_t in range(window_start, t):
                half_hist = half_sizes.get(hist_t)
                full_hist = full_sizes.get(hist_t)
                if half_hist is None or full_hist is None:
                    continue
                if half_hist <= EPS_FLOOR:
                    ratios.append(1.0)
                else:
                    ratios.append(full_hist / half_hist)
                size_values.append(full_hist)
            hat_r = sum(ratios) / len(ratios) if ratios else 1.0
            bar_s = sum(size_values) / len(size_values) if size_values else max(
                _clique_set_size(first_ds.domain, prev_cliques), 1.0
            )
            hat_s_i = hat_r * half_size_current
            gamma = hat_s_i / bar_s if bar_s > EPS_FLOOR else 1.0
            base_back = min(gamma * eps_remain_pre, eps_remain_pre)
            per_iter_back = base_back / float(max(T, 1))
            back_budgets = [per_iter_back] * T_back

        if back_budgets:
            planned_back = sum(back_budgets)
            allowed_back = min(planned_back, available_after_front)
            ratio = 0.0 if planned_back <= EPS_FLOOR else allowed_back / planned_back
            back_budgets = [b * ratio for b in back_budgets]

        consumed_back = 0.0
        state_full = state_front
        if back_budgets:
            state_full, _, consumed_back = runner.run(state_front, back_budgets)

        allocated_total = sum(front_budgets) + sum(back_budgets)
        actual_total = consumed_front + consumed_back

        if state_full is None:
            final_cliques = prev_cliques
            final_model = prev_model
            half_size_used = half_sizes.get(t - 1, half_size_current)
            full_size_used = full_sizes.get(
                t - 1, _clique_set_size(first_ds.domain, prev_cliques)
            )
        else:
            if prev_model is None:
                err_prev = float("inf")
            else:
                err_prev = _compute_workload_error(data, prev_model, workload)
            err_curr = _compute_workload_error(data, state_full.est, workload)
            if prev_model is not None and err_prev < err_curr:
                actual_total = 0.0
                final_cliques = prev_cliques
                final_model = prev_model
                half_size_used = half_sizes.get(t - 1, half_size_current)
                full_size_used = full_sizes.get(
                    t - 1, _clique_set_size(first_ds.domain, prev_cliques)
                )
            else:
                final_cliques = list(state_full.cliques)
                final_model = state_full.est
                half_size_used = half_size_current
                full_size_used = _clique_set_size(first_ds.domain, final_cliques)

        cl_history[t] = final_cliques
        model_history[t] = final_model
        actual_consumed[t] = actual_total
        allocated[t] = allocated_total
        half_sizes[t] = half_size_used
        full_sizes[t] = full_size_used

        active_window.append((t, actual_total))
        active_sum += actual_total
        eps_after = max(epsilon - active_sum, 0.0)

        error_val = (
            _compute_workload_error(data, final_model, workload)
            if final_model is not None
            else 0.0
        )

        results.append(
            {
                "timestamp": t,
                "allocated_budget": allocated_total,
                "actual_consumed_budget": actual_total,
                "eps_remain": eps_after,
                "workload_error": error_val,
                "cliques_count": len(final_cliques),
            }
        )

        if verbose and log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                for entry in runner.consume_logs():
                    f.write(
                        f"t={t}, iter={entry['iteration']}, budget={entry['budget']:.6f}, "
                        f"clique={entry['clique']}, error={entry['workload_error']:.6f}\n"
                    )
        else:
            runner.consume_logs()

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "allocated_budget",
                "actual_consumed_budget",
                "eps_remain",
                "workload_error",
                "cliques_count",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return result_path, log_path


# =============================
# Command Line Entry
# =============================


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthephus streaming data synthesis")
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--timestamp_exp", type=int, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--domain", help="path to domain.json, defaults to input_folder if not provided")
    parser.add_argument("--pgm_iters", type=int, default=1000)
    parser.add_argument(
        "--noise", choices=["gaussian", "laplace"], default="gaussian"
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_model_size_mb", type=float, default=25.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["synthephus", "baseline"],
        default="synthephus",
        help="Choose to run Synthephus or baseline mwem_plain",
    )
    return parser


def _main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    if args.mode == "baseline":
        path = mwem_plain(
            input_folder=args.input_folder,
            epsilon=args.epsilon,
            w=args.w,
            timestamp_exp=args.timestamp_exp,
            T=args.T,
            domain_path=args.domain,
            pgm_iters=args.pgm_iters,
            noise=args.noise,
            verbose=args.verbose,
            output_dir=args.output_dir,
            delta=args.delta,
            max_model_size_mb=args.max_model_size_mb,
        )
        print(f"Baseline result saved to: {path}")
    else:
        result_path, log_path = synthephus_mwem_pgm(
            input_folder=args.input_folder,
            epsilon=args.epsilon,
            w=args.w,
            timestamp_exp=args.timestamp_exp,
            T=args.T,
            domain_path=args.domain,
            pgm_iters=args.pgm_iters,
            noise=args.noise,
            verbose=args.verbose,
            output_dir=args.output_dir,
            delta=args.delta,
            max_model_size_mb=args.max_model_size_mb,
        )
        print(f"Synthephus result saved to: {result_path}")
        if log_path:
            print(f"Detailed log: {log_path}")


if __name__ == "__main__":
    _main()
