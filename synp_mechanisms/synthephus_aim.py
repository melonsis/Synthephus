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
from hdmm.matrix import Identity
from scipy.special import softmax

"""Synthephus streaming synthesis implementation based on AIM algorithm."""


# =============================
# Utility Functions
# =============================


EPS_FLOOR = 1e-12
DEFAULT_DELTA = 1e-9


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    """Compute the downward closure of workload"""
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    """Calculate the hypothetical model size in MB"""
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2 ** 20


def compile_workload(workload):
    """Compile workload into a scoring dictionary for candidate cliques"""
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    def score(cl):
        return sum(
            weights[workload_cl] * len(set(cl) & set(workload_cl))
            for workload_cl in workload_cliques
        )

    return {cl: score(cl) for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    """Filter candidate cliques that satisfy the size limit"""
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
            hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


def _compute_workload_error(
    data: Dataset, est: GraphicalModel, workload: Iterable[Tuple[str, ...]]
) -> float:
    """Compute workload error using the original AIM evaluation method"""
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
    """Return the size of the graphical model corresponding to the clique set"""
    cliques_list = list(cliques)
    if not cliques_list:
        return 0.0
    gm = GraphicalModel(domain, cliques_list)
    return float(gm.size)


def cdp_rho(epsilon: float, delta: float) -> float:
    """Convert (ε,δ)-DP to ρ-zCDP"""
    if epsilon <= 0 or delta <= 0:
        return EPS_FLOOR
    if delta >= 1:
        return EPS_FLOOR
    return epsilon ** 2 / (2 * np.log(1 / delta))


# =============================
# AIM Runner
# =============================


@dataclass
class AIMState:
    """Encapsulate AIM iteration state for stage switching and replay"""
    measurements: List[Any]
    cliques: List[Tuple[str, ...]]
    est: GraphicalModel
    rho_used: float


class AIMRunner:
    """AIM data synthesis controller for a single timestamp, providing split-run capability"""

    def __init__(
        self,
        data: Dataset,
        workload: List[Tuple[str, ...]],
        max_model_size: float,
        max_iters: int = 1000,
        delta: float = DEFAULT_DELTA,
        verbose: bool = False,
        rounds: Optional[int] = None,
    ) -> None:
        self.data = data
        self.workload = workload
        self.max_model_size = max_model_size
        self.max_iters = max_iters
        self.delta = delta if delta > 0 else DEFAULT_DELTA
        self.domain = data.domain
        self.verbose = verbose
        self._logs: List[Dict[str, Any]] = []
        # Set rounds parameter, default to 16 * number of attributes, used for budget allocation
        self.rounds = rounds if rounds is not None else 16 * len(data.domain)

        # Compile workload and get answers
        self.workload_weighted = [(cl, 1.0) for cl in workload]
        self.candidates = compile_workload(self.workload_weighted)
        self.answers = {cl: data.project(cl).datavector() for cl in self.candidates}

        # Initialize inference engine
        self.engine = FactoredInference(
            data.domain, iters=max_iters, warm_start=True, structural_zeros={}
        )

    def _log(self, iteration: int, clique: Optional[Tuple[str, ...]], rho_used: float, est: GraphicalModel) -> None:
        """Record detailed logs"""
        if not self.verbose:
            return
        model_mb = float(est.size * 8 / (2 ** 20)) if hasattr(est, "size") else 0.0
        self._logs.append(
            {
                "iteration": iteration,
                "clique": list(clique) if clique else None,
                "rho_used": float(rho_used),
                "model_size_mb": model_mb,
                "workload_error": _compute_workload_error(self.data, est, self.workload),
            }
        )

    def consume_logs(self) -> List[Dict[str, Any]]:
        """Consume and clear logs"""
        logs = self._logs
        self._logs = []
        return logs

    def _exponential_mechanism(self, errors, eps, sensitivity):
        """Select clique using exponential mechanism"""
        errors_array = np.array(list(errors.values()))
        cliques_list = list(errors.keys())
        if len(errors_array) == 0:
            return None
        probs = softmax(0.5 * eps / sensitivity * (errors_array - errors_array.max()))
        idx = np.random.choice(len(errors_array), p=probs)
        return cliques_list[idx]

    def _worst_approximated(self, candidates, model, eps, sigma):
        """Find the clique with the worst approximation error"""
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = self.answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(sensitivity.values()) if sensitivity else 1.0
        return self._exponential_mechanism(errors, eps, max_sensitivity)

    def _gaussian_noise(self, sigma, size):
        """Generate Gaussian noise"""
        return np.random.normal(loc=0.0, scale=sigma, size=size)

    def run_oneway(
        self,
        rho_total: float,
        state: Optional[AIMState] = None,
    ) -> Tuple[AIMState, List[Tuple[str, ...]], float]:
        """
        Stage 1: Run one-way marginal observations
        
        Args:
            rho_total: Total rho budget
            state: Existing state (if any)
            
        Returns:
            (new state, newly added cliques, actual consumed rho)
        """
        if state is None:
            measurements = []
            cliques = []
            rho_used = 0.0
            est = self.engine.estimate(measurements)
        else:
            measurements = list(state.measurements)
            cliques = list(state.cliques)
            rho_used = state.rho_used
            est = state.est

        # Check if budget is valid
        if rho_total <= 0:
            return AIMState(measurements, cliques, est, rho_used), [], 0.0

        # Get one-way marginals
        oneway = [cl for cl in self.candidates if len(cl) == 1]
        
        # Calculate noise parameters using rounds and total budget
        # Note: rho_total here should be the total budget for the entire algorithm
        sigma = np.sqrt(self.rounds / (2 * 0.9 * rho_total))
        
        # Add one-way marginal observations
        new_cliques = []
        for cl in oneway:
            x = self.data.project(cl).datavector()
            y = x + self._gaussian_noise(sigma, x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))
            new_cliques.append(cl)
            cliques.append(cl)

        rho_oneway = len(oneway) * 0.5 / (sigma ** 2)
        rho_used += rho_oneway

        # Estimate model
        est = self.engine.estimate(measurements)

        return AIMState(measurements, cliques, est, rho_used), new_cliques, rho_oneway

    def run_adaptive(
        self,
        rho_available: float,
        state: AIMState,
        total_rho: float,
        max_rounds: Optional[int] = None,
        temp_save_path: Optional[str] = None,
        temp_load_path: Optional[str] = None,
    ) -> Tuple[AIMState, List[Tuple[str, ...]], float, int]:
        """
        Stage 2: Run adaptive iterations
        
        Args:
            rho_available: Available rho budget for this stage
            state: State from previous stage
            total_rho: Total rho budget for the entire algorithm (used to compute initial sigma/epsilon)
            max_rounds: Maximum iteration rounds (None means adaptively determined)
            temp_save_path: Path to save temporary model
            temp_load_path: Path to load temporary model
            
        Returns:
            (new state, newly added cliques, actual consumed rho, actual iteration rounds)
        """
        measurements = list(state.measurements)
        cliques = list(state.cliques)
        rho_used_before = state.rho_used
        est = state.est

        # Check if budget is valid
        if rho_available <= 0 or total_rho <= 0:
            return AIMState(measurements, cliques, est, rho_used_before), [], 0.0, 0

        # If need to load model from file
        if temp_load_path and os.path.exists(temp_load_path):
            with open(temp_load_path, "rb") as f:
                est = pickle.load(f)

        rho_consumed = 0.0
        new_cliques = []
        iteration_count = 0
        
        # Initialize adaptive parameters using rounds and total budget
        sigma = np.sqrt(self.rounds / (2 * 0.9 * total_rho))
        epsilon = np.sqrt(8 * 0.1 * total_rho / self.rounds)
        if temp_load_path and os.path.exists(temp_load_path):
            # If model was loaded, assume annealing has occurred once
            sigma /= 2
            epsilon *= 2

        terminate = False
        while not terminate:
            iteration_count += 1
            
            # Check if max rounds reached
            if max_rounds is not None and iteration_count > max_rounds:
                break
            
            # Check remaining budget
            rho_remaining = rho_available - rho_consumed
            if rho_remaining <= 0:
                break
            if rho_remaining < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2):
                # Use remaining budget for last round
                if rho_remaining > EPS_FLOOR:
                    sigma = np.sqrt(1 / (2 * 0.9 * rho_remaining))
                    epsilon = np.sqrt(8 * 0.1 * rho_remaining)
                else:
                    break
                terminate = True

            rho_step = 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
            size_limit = self.max_model_size * (rho_used_before + rho_consumed + rho_step) / total_rho

            # 过滤候选 clique
            small_candidates = filter_candidates(self.candidates, est, size_limit)
            if not small_candidates:
                break

            # 选择最差近似的 clique
            cl = self._worst_approximated(small_candidates, est, epsilon, sigma)
            if cl is None:
                break

            n = self.domain.size(cl)
            Q = Identity(n)
            x = self.data.project(cl).datavector()
            y = x + self._gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            new_cliques.append(cl)
            cliques.append(cl)
            z = est.project(cl).datavector()

            # 更新模型
            est = self.engine.estimate(measurements)
            w = est.project(cl).datavector()

            rho_consumed += rho_step

            if self.verbose:
                print(f'Selected {cl}, Size {n}, Rho Used {(rho_used_before + rho_consumed):.4f}')

            # Check if sigma should be reduced
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                if self.verbose:
                    print(f"Reducing sigma from {sigma} to {sigma / 2}")
                sigma /= 2
                epsilon *= 2

        # Save temporary model
        if temp_save_path:
            with open(temp_save_path, "wb") as f:
                pickle.dump(est, f)

        final_state = AIMState(measurements, cliques, est, rho_used_before + rho_consumed)
        return final_state, new_cliques, rho_consumed, iteration_count

    def run_adaptive_until_first_annealing(
        self,
        rho_available: float,
        state: AIMState,
        total_rho: float,
        temp_save_path: Optional[str] = None,
    ) -> Tuple[AIMState, List[Tuple[str, ...]], float, int]:
        """
        Run adaptive iterations until first sigma reduction (annealing)
        
        Args:
            rho_available: Available rho budget for this stage
            state: State from previous stage
            total_rho: Total rho budget for the entire algorithm (used to compute initial sigma/epsilon)
            temp_save_path: Path to save temporary model
            
        Returns:
            (new state, newly added cliques, actual consumed rho, actual iteration rounds T_trunc)
        """
        measurements = list(state.measurements)
        cliques = list(state.cliques)
        rho_used_before = state.rho_used
        est = state.est

        # Check if budget is valid
        if rho_available <= 0 or total_rho <= 0:
            return AIMState(measurements, cliques, est, rho_used_before), [], 0.0, 0

        rho_consumed = 0.0
        new_cliques = []
        iteration_count = 0
        
        # Initialize adaptive parameters
        sigma = np.sqrt(self.rounds / (2 * 0.9 * total_rho))
        epsilon = np.sqrt(8 * 0.1 * total_rho / self.rounds)
        initial_sigma = sigma  # Record initial sigma

        while True:
            iteration_count += 1
            
            # Check remaining budget
            rho_remaining = rho_available - rho_consumed
            if rho_remaining <= 0:
                break
            if rho_remaining < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2):
                # Insufficient budget, exit
                break

            rho_step = 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
            size_limit = self.max_model_size * (rho_used_before + rho_consumed + rho_step) / total_rho

            # Filter clique
            small_candidates = filter_candidates(self.candidates, est, size_limit)
            if not small_candidates:
                break

            # Select the worst approximated clique
            cl = self._worst_approximated(small_candidates, est, epsilon, sigma)
            if cl is None:
                break

            n = self.domain.size(cl)
            Q = Identity(n)
            x = self.data.project(cl).datavector()
            y = x + self._gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            new_cliques.append(cl)
            cliques.append(cl)
            z = est.project(cl).datavector()

            # Update model
            est = self.engine.estimate(measurements)
            w = est.project(cl).datavector()

            rho_consumed += rho_step

            if self.verbose:
                print(f'Selected {cl}, Size {n}, Rho Used {(rho_used_before + rho_consumed):.4f}')

            # Check if sigma should be reduced (first annealing)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                if self.verbose:
                    print(f"First annealing: Reducing sigma from {sigma} to {sigma / 2}, stopping at T_trunc={iteration_count}")
                sigma /= 2
                epsilon *= 2
                # Exit loop immediately
                break

        # Save temporary model
        if temp_save_path:
            with open(temp_save_path, "wb") as f:
                pickle.dump(est, f)

        final_state = AIMState(measurements, cliques, est, rho_used_before + rho_consumed)
        return final_state, new_cliques, rho_consumed, iteration_count


# =============================
# Main Algorithm and Baseline
# =============================


def aim_plain(
    input_folder: str,
    epsilon: float,
    w: int,
    timestamp_exp: int,
    domain_path: Optional[str] = None,
    workload: Optional[List[Tuple[str, ...]]] = None,
    delta: float = DEFAULT_DELTA,
    max_model_size_mb: float = 80.0,
    max_iters: int = 1000,
    verbose: bool = False,
    output_dir: str = "results",
) -> str:
    """Baseline approach: fixed budget eps/w for each timestamp, no dynamic adjustment"""

    domain_path = domain_path or os.path.join(os.path.dirname(input_folder), "domain.json")
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Missing domain.json: {domain_path}")

    first_file = os.path.join(input_folder, "real_1.csv")
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"Missing input file: {first_file}")

    first_ds = Dataset.load(first_file, domain_path)
    workload = workload or list(itertools.combinations(first_ds.domain, 2))

    os.makedirs(output_dir, exist_ok=True)
    utc_tag = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result_path = os.path.join(output_dir, f"synp_aim_plain_result_{utc_tag}.csv")

    per_timestamp_eps = epsilon / float(max(w, 1))
    per_timestamp_rho = cdp_rho(per_timestamp_eps, delta)

    active_window: Deque[Tuple[int, float]] = deque()
    active_sum = 0.0

    results: List[Dict[str, Any]] = []
    model_history: Dict[int, GraphicalModel] = {}
    cl_history: Dict[int, List[Tuple[str, ...]]] = {}
    actual_consumed: Dict[int, float] = {}

    for t in tqdm(range(1, timestamp_exp + 1), desc="Baseline timestamp progress"):
        # Window management
        while active_window and (t - active_window[0][0]) >= w:
            _, removed = active_window.popleft()
            active_sum -= removed
        eps_remain = max(epsilon - active_sum, 0.0)

        data_path = os.path.join(input_folder, f"real_{t}.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing input file: {data_path}")
        data = Dataset.load(data_path, domain_path)

        runner = AIMRunner(
            data,
            workload,
            max_model_size=max_model_size_mb,
            max_iters=max_iters,
            delta=delta,
            verbose=verbose,
        )

        # Execute complete AIM algorithm
        state_oneway, _, rho_oneway = runner.run_oneway(per_timestamp_rho)
        state_final, _, rho_adaptive, _ = runner.run_adaptive(
            per_timestamp_rho - rho_oneway,
            state_oneway,
            total_rho=per_timestamp_rho,
        )

        # Convert rho back to epsilon (approximation)
        consumed_eps = per_timestamp_eps
        actual_consumed[t] = consumed_eps
        cl_history[t] = list(state_final.cliques)
        model_history[t] = state_final.est

        active_window.append((t, consumed_eps))
        active_sum += consumed_eps
        eps_after = max(epsilon - active_sum, 0.0)

        error_val = _compute_workload_error(data, state_final.est, workload)
        results.append(
            {
                "timestamp": t,
                "allocated_budget": per_timestamp_eps,
                "actual_consumed_budget": consumed_eps,
                "eps_remain": eps_after,
                "workload_error": error_val,
                "cliques_count": len(state_final.cliques),
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


def synthephus_aim(
    input_folder: str,
    epsilon: float,
    w: int,
    timestamp_exp: int,
    domain_path: Optional[str] = None,
    workload: Optional[List[Tuple[str, ...]]] = None,
    delta: float = DEFAULT_DELTA,
    max_model_size_mb: float = 80.0,
    max_iters: int = 1000,
    verbose: bool = False,
    output_dir: str = "results",
) -> Tuple[str, Optional[str]]:
    """Synthephus main workflow: dynamic budget, window management, and quality fallback"""

    domain_path = domain_path or os.path.join(os.path.dirname(input_folder), "domain.json")
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Missing domain.json: {domain_path}")

    first_file = os.path.join(input_folder, "real_1.csv")
    if not os.path.exists(first_file):
        raise FileNotFoundError(f"Missing input file: {first_file}")

    first_ds = Dataset.load(first_file, domain_path)
    workload = workload or list(itertools.combinations(first_ds.domain, 2))

    os.makedirs(output_dir, exist_ok=True)
    utc_tag = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result_path = os.path.join(output_dir, f"synp_aim_result_{utc_tag}.csv")
    log_path = (
        os.path.join(output_dir, f"synp_aim_log_{utc_tag}.txt") if verbose else None
    )

    active_window: Deque[Tuple[int, float]] = deque()
    active_sum = 0.0

    model_history: Dict[int, GraphicalModel] = {}
    cl_history: Dict[int, List[Tuple[str, ...]]] = {}
    actual_consumed: Dict[int, float] = {}
    allocated: Dict[int, float] = {}
    T_trunc_sizes: Dict[int, float] = {}  # S_{t'}^{T_trunc}
    T_final_sizes: Dict[int, float] = {}  # S_{t'}
    T_trunc_history: Dict[int, int] = {}  # T_trunc per timestamp
    T_final_history: Dict[int, int] = {}  # T final per timestamp
    T_init_value: Optional[int] = None

    results: List[Dict[str, Any]] = []

    for t in tqdm(range(1, timestamp_exp + 1), desc="Synthephus AIM timestamp progress"):
        # Window management
        while active_window and (t - active_window[0][0]) >= w:
            _, removed = active_window.popleft()
            active_sum -= removed
        eps_remain_pre = max(epsilon - active_sum, 0.0)

        data_path = os.path.join(input_folder, f"real_{t}.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing input file: {data_path}")
        data = Dataset.load(data_path, domain_path)

        runner = AIMRunner(
            data,
            workload,
            max_model_size=max_model_size_mb,
            max_iters=max_iters,
            delta=delta,
            verbose=verbose,
        )

        prev_model = model_history.get(t - 1)
        prev_cliques = cl_history.get(t - 1, [])

        # ========== Stage A: Timestamp 1 ==========
        if t == 1:
            alloc_eps = epsilon / float(max(w, 1))
            alloc_rho = cdp_rho(alloc_eps, delta)
            allocated[t] = alloc_eps

            state_oneway, _, rho_oneway = runner.run_oneway(alloc_rho)
            state_final, _, rho_adaptive, T_init_value = runner.run_adaptive(
                alloc_rho - rho_oneway,
                state_oneway,
                total_rho=alloc_rho,
            )

            cl_history[t] = list(state_final.cliques)
            model_history[t] = state_final.est
            actual_consumed[t] = alloc_eps
            allocated_total = alloc_eps
            actual_total = alloc_eps
            
            # For first timestamp, trunc and final are the same
            size_final = _clique_set_size(first_ds.domain, state_final.cliques)
            T_trunc_sizes[t] = size_final
            T_final_sizes[t] = size_final
            T_trunc_history[t] = T_init_value or 0
            T_final_history[t] = T_init_value or 0

        # ========== Stage B: Timestamp 2 to w ==========
        elif 2 <= t <= w:
            denom = max(w - t, 1)
            base_eps = eps_remain_pre / float(denom)
            base_rho = cdp_rho(base_eps, delta)

            # 1. One-way marginal observations
            state_oneway, _, rho_oneway = runner.run_oneway(base_rho)
            eps_oneway = base_eps * (rho_oneway / base_rho) if base_rho > 0 else 0.0

            # 2. Truncate at half of previous timestamp's total rounds (T_{i-1} / 2)
            temp_file = os.path.join(output_dir, f"temp_{uuid.uuid4().hex}.pkl")
            
            prev_T_final = T_final_history.get(t - 1, T_init_value or 10)
            T_trunc = prev_T_final // 2
            
            rho_front = base_rho - rho_oneway
            state_temp, cl_temp, rho_temp, actual_trunc = runner.run_adaptive(
                rho_front,
                state_oneway,
                total_rho=base_rho,
                max_rounds=T_trunc,
                temp_save_path=temp_file,
            )
            eps_front = base_eps * (rho_temp / base_rho) if base_rho > 0 else 0.0
            size_trunc = _clique_set_size(first_ds.domain, cl_temp)

            # 3. Arbitrary additional rounds
            available_after_front = max(eps_remain_pre - eps_oneway - eps_front, 0.0)

            # Calculate global budget upper bound: [2 * size(cl_temp) / size(cl_{i-1})] * (eps_remain / (w - i) - eps_one_way)
            prev_size = T_final_sizes.get(t - 1, size_trunc)
            size_ratio = size_trunc / prev_size if prev_size > EPS_FLOOR else 1.0

            base_back_eps = 2.0 * size_ratio * (eps_remain_pre / float(denom) - eps_oneway)
            base_back_eps = min(base_back_eps, available_after_front)
            base_back_rho = cdp_rho(base_back_eps, delta) if base_back_eps > 0 else 0.0

            if base_back_rho > 0:
                # Use eps_remain / (w - i) as reference to calculate initial sigma
                state_final, _, rho_back, T_final = runner.run_adaptive(
                    base_back_rho,
                    state_temp,
                    total_rho=cdp_rho(eps_remain_pre / float(denom), delta),
                    temp_load_path=temp_file,
                )
                eps_back = base_back_eps
            else:
                state_final = state_temp
                eps_back = 0.0
                T_final = actual_trunc

            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

            allocated_total = eps_oneway + eps_front + eps_back
            actual_total = allocated_total
            
            T_trunc_sizes[t] = size_trunc
            T_final_sizes[t] = _clique_set_size(first_ds.domain, state_final.cliques)
            T_trunc_history[t] = actual_trunc
            T_final_history[t] = T_final

        # ========== Stage C: Timestamp w+1 to timestamp_exp ==========
        else:  # t > w
            # 0. Preliminary calculation
            window_start = max(1, t - w)
            R_values: List[float] = []
            S_values: List[float] = []
            
            for hist_t in range(window_start, t):
                S_trunc = T_trunc_sizes.get(hist_t)
                S_final = T_final_sizes.get(hist_t)
                if S_trunc is None or S_final is None:
                    continue
                if S_trunc <= EPS_FLOOR:
                    R_values.append(1.0)
                else:
                    R_values.append(S_final / S_trunc)
                S_values.append(S_final)
            
            hat_R = sum(R_values) / len(R_values) if R_values else 1.0
            bar_S = sum(S_values) / len(S_values) if S_values else max(
                _clique_set_size(first_ds.domain, prev_cliques), 1.0
            )

            # 1. Budget recovery (completed in window management)

            # 2. Truncate at average of half the total rounds of previous w timestamps
            # T_trunc = sum(T_{t-w+1}...T_{t-1}) / (2*w)
            base_eps = eps_remain_pre
            base_rho = cdp_rho(base_eps, delta)

            state_oneway, _, rho_oneway = runner.run_oneway(base_rho)
            eps_oneway = base_eps * (rho_oneway / base_rho) if base_rho > 0 else 0.0

            temp_file = os.path.join(output_dir, f"temp_{uuid.uuid4().hex}.pkl")
            
            # Use average rounds from window
            window_T_start = max(1, t - w + 1)
            total_T_in_window = sum(
                T_final_history.get(idx, T_init_value or 10) 
                for idx in range(window_T_start, t)
            )
            T_trunc = total_T_in_window // (2 * w)
            # Ensure T_trunc is at least 1
            T_trunc = max(T_trunc, 1)
            
            rho_front = base_rho - rho_oneway
            state_temp, cl_temp, rho_temp, actual_trunc = runner.run_adaptive(
                rho_front,
                state_oneway,
                total_rho=base_rho,
                max_rounds=T_trunc,
                temp_save_path=temp_file,
            )
            eps_front = base_eps * (rho_temp / base_rho) if base_rho > 0 else 0.0
            size_trunc = _clique_set_size(first_ds.domain, cl_temp)

            # 3. Arbitrary additional rounds
            available_after_front = max(eps_remain_pre - eps_oneway - eps_front, 0.0)

            # Predict domain size
            hat_S_i = hat_R * size_trunc

            # Calculate budget correction factor gamma
            gamma = hat_S_i / bar_S if bar_S > EPS_FLOOR else 1.0

            # Revised global budget upper bound: gamma * (eps_remain - eps_one_way)
            base_back_eps = gamma * (eps_remain_pre - eps_oneway)
            base_back_eps = min(base_back_eps, available_after_front)
            base_back_rho = cdp_rho(base_back_eps, delta) if base_back_eps > 0 else 0.0

            if base_back_rho > 0:
                # Use eps_remain as reference to calculate initial sigma
                state_final, _, rho_back, T_final = runner.run_adaptive(
                    base_back_rho,
                    state_temp,
                    total_rho=cdp_rho(eps_remain_pre, delta),
                    temp_load_path=temp_file,
                )
                eps_back = base_back_eps
            else:
                state_final = state_temp
                eps_back = 0.0
                T_final = actual_trunc

            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

            allocated_total = eps_oneway + eps_front + eps_back
            actual_total = allocated_total
            
            T_trunc_sizes[t] = size_trunc
            T_final_sizes[t] = _clique_set_size(first_ds.domain, state_final.cliques)
            T_trunc_history[t] = actual_trunc
            T_final_history[t] = T_final

        # ========== Quality assurance mechanism ==========
        if t > 1:
            if prev_model is None:
                err_prev = float("inf")
            else:
                err_prev = _compute_workload_error(data, prev_model, workload)
            err_curr = _compute_workload_error(data, state_final.est, workload)

            if prev_model is not None and err_prev < err_curr:
                # Model fallback
                actual_total = 0.0
                final_cliques = prev_cliques
                final_model = prev_model
                # Round records also fall back
                T_trunc_used = T_trunc_history.get(t - 1, actual_trunc if t > 1 else 0)
                T_final_used = T_final_history.get(t - 1, T_final)
                T_trunc_size_used = T_trunc_sizes.get(t - 1, size_trunc if t > 1 else 0.0)
                T_final_size_used = T_final_sizes.get(t - 1, T_final_sizes.get(t, 0.0))
            else:
                # No fallback, use current model
                final_cliques = list(state_final.cliques)
                final_model = state_final.est
                T_trunc_used = actual_trunc if t > 1 else T_trunc_history.get(t, 0)
                T_final_used = T_final
                T_trunc_size_used = size_trunc if t > 1 else T_trunc_sizes.get(t, 0.0)
                T_final_size_used = T_final_sizes.get(t, 0.0)
        else:  # t == 1
            final_cliques = list(state_final.cliques)
            final_model = state_final.est
            T_trunc_used = T_trunc_history.get(t, 0)
            T_final_used = T_final_history.get(t, 0)
            T_trunc_size_used = T_trunc_sizes.get(t, 0.0)
            T_final_size_used = T_final_sizes.get(t, 0.0)

        cl_history[t] = final_cliques
        model_history[t] = final_model
        actual_consumed[t] = actual_total if t > 1 else actual_consumed.get(t, actual_total)
        allocated[t] = allocated.get(t, allocated_total)
        T_trunc_sizes[t] = T_trunc_size_used
        T_final_sizes[t] = T_final_size_used
        T_trunc_history[t] = T_trunc_used
        T_final_history[t] = T_final_used

        active_window.append((t, actual_consumed[t]))
        active_sum += actual_consumed[t]
        eps_after = max(epsilon - active_sum, 0.0)

        error_val = _compute_workload_error(data, final_model, workload) if final_model else 0.0

        # Release old history records
        if t > w:
            cl_history.pop(t - w, None)
            model_history.pop(t - w, None)
            T_trunc_sizes.pop(t - w, None)
            T_final_sizes.pop(t - w, None)
            T_trunc_history.pop(t - w, None)
            T_final_history.pop(t - w, None)

        results.append(
            {
                "timestamp": t,
                "allocated_budget": allocated[t],
                "actual_consumed_budget": actual_consumed[t],
                "eps_remain": eps_after,
                "workload_error": error_val,
                "cliques_count": len(final_cliques),
            }
        )

        if verbose and log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                for entry in runner.consume_logs():
                    f.write(
                        f"t={t}, iter={entry['iteration']}, rho_used={entry['rho_used']:.6f}, "
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
    parser = argparse.ArgumentParser(description="Synthephus AIM streaming data synthesis")
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--timestamp_exp", type=int, required=True)
    parser.add_argument("--domain", help="path to domain.json, defaults to input_folder if not provided")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_model_size_mb", type=float, default=80.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["synthephus", "baseline"],
        default="synthephus",
        help="Choose to run Synthephus or baseline aim_plain",
    )
    return parser


def _main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    if args.mode == "baseline":
        path = aim_plain(
            input_folder=args.input_folder,
            epsilon=args.epsilon,
            w=args.w,
            timestamp_exp=args.timestamp_exp,
            domain_path=args.domain,
            delta=args.delta,
            max_model_size_mb=args.max_model_size_mb,
            max_iters=args.max_iters,
            verbose=args.verbose,
            output_dir=args.output_dir,
        )
        print(f"Baseline result saved to: {path}")
    else:
        result_path, log_path = synthephus_aim(
            input_folder=args.input_folder,
            epsilon=args.epsilon,
            w=args.w,
            timestamp_exp=args.timestamp_exp,
            domain_path=args.domain,
            delta=args.delta,
            max_model_size_mb=args.max_model_size_mb,
            max_iters=args.max_iters,
            verbose=args.verbose,
            output_dir=args.output_dir,
        )
        print(f"Synthephus AIM result saved to: {result_path}")
        if log_path:
            print(f"Detailed log: {log_path}")


if __name__ == "__main__":
    _main()

