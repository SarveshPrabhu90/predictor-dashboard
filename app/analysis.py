"""
Analysis module — data collection, prediction, and statistics.

Handles all communication with the TCP server and runs the modelless
predictor training / evaluation pipeline.
"""

import base64
import io
import json
import os
import socket
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from .manifest import write_manifest

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

# ── Ground-truth coefficients (for validation only) ────────────────────────
YIELD_W = np.array([0.45, 0.30, 0.80])
YIELD_I = 12.0
PURITY_W = np.array([-0.15, 0.55, 0.35])
PURITY_I = 85.0
NOISE_STD = 0.5

INPUT_NAMES = ["temperature", "flow_rate", "concentration"]
OUTPUT_NAMES = ["yield", "purity"]
INPUT_RANGES = [(20.0, 80.0), (1.0, 10.0), (0.1, 5.0)]


def ground_truth(inputs: np.ndarray) -> np.ndarray:
    """Noise-free deterministic outputs from known coefficients."""
    y = inputs @ YIELD_W + YIELD_I
    p = inputs @ PURITY_W + PURITY_I
    return np.column_stack([y, p])


def collect_observations(n: int, host: str = "127.0.0.1", port: int = 9100):
    """Fetch n observations from the TCP plant server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect((host, port))
        s.sendall((json.dumps({"command": "observe", "n": n}) + "\n").encode())
        data = b""
        while b"\n" not in data:
            chunk = s.recv(262144)
            if not chunk:
                break
            data += chunk
    resp = json.loads(data)
    return np.array(resp["inputs"]), np.array(resp["outputs"])


def server_is_available(host: str = "127.0.0.1", port: int = 9100) -> bool:
    """Check if the TCP server is reachable."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect((host, port))
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ── Explicit model analysis ────────────────────────────────────────────────

def run_explicit_analysis(n: int = 500):
    """Run full explicit model validation. Returns a dict of results."""
    t_start = time.time()
    inputs, noisy_outputs = collect_observations(n)
    gt_outputs = ground_truth(inputs)
    residuals = noisy_outputs - gt_outputs

    # Metrics
    stats = {}
    for i, name in enumerate(OUTPUT_NAMES):
        stats[name] = {
            "r2": float(r2_score(gt_outputs[:, i], noisy_outputs[:, i])),
            "mae": float(mean_absolute_error(gt_outputs[:, i], noisy_outputs[:, i])),
            "noise_mean": float(residuals[:, i].mean()),
            "noise_std": float(residuals[:, i].std()),
        }

    input_stats = {}
    for i, name in enumerate(INPUT_NAMES):
        col = inputs[:, i]
        input_stats[name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "min": float(col.min()), "max": float(col.max()),
        }

    output_stats = {}
    for i, name in enumerate(OUTPUT_NAMES):
        col = noisy_outputs[:, i]
        output_stats[name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "min": float(col.min()), "max": float(col.max()),
        }

    # Generate charts
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Explicit Model — Validation Dashboard", fontsize=13, fontweight="bold")

    # Yield scatter
    ax = axes[0, 0]
    ax.scatter(gt_outputs[:, 0], noisy_outputs[:, 0], s=10, alpha=0.5, c="#1f77b4")
    lo, hi = gt_outputs[:, 0].min(), gt_outputs[:, 0].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Ground Truth Yield")
    ax.set_ylabel("Observed Yield")
    ax.set_title(f"Yield: R²={stats['yield']['r2']:.4f}")

    # Purity scatter
    ax = axes[0, 1]
    ax.scatter(gt_outputs[:, 1], noisy_outputs[:, 1], s=10, alpha=0.5, c="#ff7f0e")
    lo, hi = gt_outputs[:, 1].min(), gt_outputs[:, 1].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Ground Truth Purity")
    ax.set_ylabel("Observed Purity")
    ax.set_title(f"Purity: R²={stats['purity']['r2']:.4f}")

    # Input coverage
    ax = axes[0, 2]
    sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=inputs[:, 2], s=10, alpha=0.5, cmap="viridis")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Flow Rate (L/min)")
    ax.set_title("Input Space Coverage")
    fig.colorbar(sc, ax=ax, label="Conc (%)")

    # Yield noise histogram
    ax = axes[1, 0]
    ax.hist(residuals[:, 0], bins=30, color="#1f77b4", alpha=0.7, edgecolor="black", lw=0.5, density=True)
    x_n = np.linspace(-2, 2, 200)
    y_n = (1 / (NOISE_STD * np.sqrt(2 * np.pi))) * np.exp(-x_n ** 2 / (2 * NOISE_STD ** 2))
    ax.plot(x_n, y_n, "r-", lw=2, label=f"N(0,{NOISE_STD}²)")
    ax.set_title(f"Yield Noise: μ={stats['yield']['noise_mean']:+.4f} σ={stats['yield']['noise_std']:.4f}")
    ax.legend(fontsize=7)

    # Purity noise histogram
    ax = axes[1, 1]
    ax.hist(residuals[:, 1], bins=30, color="#ff7f0e", alpha=0.7, edgecolor="black", lw=0.5, density=True)
    ax.plot(x_n, y_n, "r-", lw=2, label=f"N(0,{NOISE_STD}²)")
    ax.set_title(f"Purity Noise: μ={stats['purity']['noise_mean']:+.4f} σ={stats['purity']['noise_std']:.4f}")
    ax.legend(fontsize=7)

    # Yield-purity tradeoff
    ax = axes[1, 2]
    ax.scatter(noisy_outputs[:, 0], noisy_outputs[:, 1], s=10, alpha=0.4, c="#2ca02c")
    ax.axhline(88, color="red", ls="--", lw=1.5, label="Purity ≥ 88%")
    ax.set_xlabel("Yield (%)")
    ax.set_ylabel("Purity (%)")
    ax.set_title("Yield–Purity Trade-off")
    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart = _fig_to_base64(fig)

    # Validation checks
    checks = [
        ("Yield noise mean ≈ 0 (|μ| < 0.1)", abs(stats["yield"]["noise_mean"]) < 0.1),
        ("Purity noise mean ≈ 0 (|μ| < 0.1)", abs(stats["purity"]["noise_mean"]) < 0.1),
        ("Yield noise σ ≈ 0.5 (within 20%)", abs(stats["yield"]["noise_std"] - NOISE_STD) < 0.1),
        ("Purity noise σ ≈ 0.5 (within 20%)", abs(stats["purity"]["noise_std"] - NOISE_STD) < 0.1),
        ("Yield R² ≥ 0.99", stats["yield"]["r2"] >= 0.99),
        ("Purity R² ≥ 0.95", stats["purity"]["r2"] >= 0.95),
    ]

    all_pass = all(ok for _, ok in checks)
    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="explicit",
        data_type="synthetic",
        explicit_model_source="TCP server (127.0.0.1:9100)",
        explicit_model_version="1.0.0",
        modelless_model_type=None,
        sample_size=n,
        train_test_split=None,
        random_seed=None,
        noise_level=NOISE_STD,
        constraints_used={"min_purity": 88.0},
        metrics=stats,
        all_checks_pass=all_pass,
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )

    return {
        "n_samples": n,
        "stats": stats,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "chart": chart,
        "checks": checks,
        "all_pass": all_pass,
    }


# ── Modelless predictor analysis ──────────────────────────────────────────

def run_predictor_analysis(n_train: int = 300, n_test: int = 100):
    """Train the modelless predictor and evaluate it. Returns a dict of results."""
    t_start = time.time()
    np.random.seed(42)

    train_in, train_out = collect_observations(n_train)
    test_in, test_out = collect_observations(n_test)
    gt_test = ground_truth(test_in)

    # Train
    models = []
    for i in range(2):
        m = LinearRegression()
        m.fit(train_in, train_out[:, i])
        models.append(m)

    learned_coeffs = {
        OUTPUT_NAMES[i]: {
            "weights": models[i].coef_.tolist(),
            "intercept": float(models[i].intercept_),
        }
        for i in range(2)
    }

    # Predict
    preds = np.column_stack([m.predict(test_in) for m in models])

    # Metrics vs ground truth
    gt_metrics = {}
    for i, name in enumerate(OUTPUT_NAMES):
        gt_metrics[name] = {
            "r2": float(r2_score(gt_test[:, i], preds[:, i])),
            "mae": float(mean_absolute_error(gt_test[:, i], preds[:, i])),
        }

    # Metrics vs noisy
    noisy_metrics = {}
    for i, name in enumerate(OUTPUT_NAMES):
        noisy_metrics[name] = {
            "r2": float(r2_score(test_out[:, i], preds[:, i])),
            "mae": float(mean_absolute_error(test_out[:, i], preds[:, i])),
        }

    residuals = preds - gt_test

    # Coefficient recovery
    true_w = {"yield": YIELD_W.tolist(), "purity": PURITY_W.tolist()}
    true_i = {"yield": YIELD_I, "purity": PURITY_I}
    coeff_errors = {}
    for name in OUTPUT_NAMES:
        w_err = [abs(a - b) for a, b in zip(learned_coeffs[name]["weights"], true_w[name])]
        i_err = abs(learned_coeffs[name]["intercept"] - true_i[name])
        coeff_errors[name] = {"weight_errors": w_err, "intercept_error": i_err}

    # Optimization
    yield_w = np.array(learned_coeffs["yield"]["weights"])
    purity_w = np.array(learned_coeffs["purity"]["weights"])
    purity_i = learned_coeffs["purity"]["intercept"]

    c = -yield_w
    A_ub = [-purity_w]
    b_ub = [-(88.0 - purity_i)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=INPUT_RANGES, method="highs")

    opt_result = None
    if result.success:
        x = result.x
        pred_at_opt = np.column_stack([m.predict(x.reshape(1, -1)) for m in models])[0]
        gt_at_opt = ground_truth(x.reshape(1, -1))[0]
        opt_result = {
            "inputs": x.tolist(),
            "predicted_yield": float(pred_at_opt[0]),
            "predicted_purity": float(pred_at_opt[1]),
            "gt_yield": float(gt_at_opt[0]),
            "gt_purity": float(gt_at_opt[1]),
            "yield_error": float(abs(pred_at_opt[0] - gt_at_opt[0])),
        }

    # Charts
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Modelless Predictor — Validation Dashboard", fontsize=13, fontweight="bold")

    # Yield: predicted vs truth
    ax = axes[0, 0]
    ax.scatter(gt_test[:, 0], preds[:, 0], s=15, alpha=0.6, c="#1f77b4")
    lo, hi = gt_test[:, 0].min(), gt_test[:, 0].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Ground Truth Yield")
    ax.set_ylabel("Predicted Yield")
    ax.set_title(f"Yield: R²={gt_metrics['yield']['r2']:.6f}")

    # Purity: predicted vs truth
    ax = axes[0, 1]
    ax.scatter(gt_test[:, 1], preds[:, 1], s=15, alpha=0.6, c="#ff7f0e")
    lo, hi = gt_test[:, 1].min(), gt_test[:, 1].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Ground Truth Purity")
    ax.set_ylabel("Predicted Purity")
    ax.set_title(f"Purity: R²={gt_metrics['purity']['r2']:.6f}")

    # Coefficient comparison
    ax = axes[0, 2]
    labels = ["T(y)", "F(y)", "C(y)", "T(p)", "F(p)", "C(p)"]
    true_vals = YIELD_W.tolist() + PURITY_W.tolist()
    learned_vals = learned_coeffs["yield"]["weights"] + learned_coeffs["purity"]["weights"]
    x_pos = np.arange(len(labels))
    w = 0.35
    ax.bar(x_pos - w / 2, true_vals, w, label="True", color="#2ca02c", alpha=0.8)
    ax.bar(x_pos + w / 2, learned_vals, w, label="Learned", color="#d62728", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Coefficient Recovery")
    ax.legend(fontsize=7)
    ax.axhline(0, color="gray", lw=0.5)

    # Yield residuals
    ax = axes[1, 0]
    ax.hist(residuals[:, 0], bins=25, color="#1f77b4", alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", ls="--", lw=1.5)
    r_y = residuals[:, 0]
    ax.set_title(f"Yield Residuals: μ={r_y.mean():+.4f} σ={r_y.std():.4f}")

    # Purity residuals
    ax = axes[1, 1]
    ax.hist(residuals[:, 1], bins=25, color="#ff7f0e", alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(0, color="red", ls="--", lw=1.5)
    r_p = residuals[:, 1]
    ax.set_title(f"Purity Residuals: μ={r_p.mean():+.4f} σ={r_p.std():.4f}")

    # Residual normality
    ax = axes[1, 2]
    for i, (name, color) in enumerate(zip(["Yield", "Purity"], ["#1f77b4", "#ff7f0e"])):
        r = np.sort(residuals[:, i])
        theoretical = np.linspace(-2.5, 2.5, len(r))
        ax.scatter(theoretical, r, s=8, alpha=0.6, c=color, label=name)
    ax.plot([-3, 3], [-3, 3], "r--", lw=1, alpha=0.5)
    ax.set_title("Residual Normality Check")
    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart = _fig_to_base64(fig)

    # Validation checks
    checks = [
        ("Yield R² ≥ 0.999 (vs ground truth)", gt_metrics["yield"]["r2"] >= 0.999),
        ("Purity R² ≥ 0.999 (vs ground truth)", gt_metrics["purity"]["r2"] >= 0.999),
        ("Yield MAE < 0.1", gt_metrics["yield"]["mae"] < 0.1),
        ("Purity MAE < 0.1", gt_metrics["purity"]["mae"] < 0.1),
        ("Yield residual mean ≈ 0", abs(residuals[:, 0].mean()) < 0.1),
        ("Purity residual mean ≈ 0", abs(residuals[:, 1].mean()) < 0.1),
        ("Optimizer feasible", opt_result is not None),
        ("Optimizer yield error < 0.5%", opt_result is not None and opt_result["yield_error"] < 0.5),
    ]

    all_pass = all(ok for _, ok in checks)
    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="predictor",
        data_type="synthetic",
        explicit_model_source="TCP server (127.0.0.1:9100)",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"train": n_train, "test": n_test},
        train_test_split={"train": n_train, "test": n_test},
        random_seed=42,
        noise_level=NOISE_STD,
        constraints_used={"min_purity": 88.0},
        metrics={
            "yield_r2_vs_truth": gt_metrics["yield"]["r2"],
            "purity_r2_vs_truth": gt_metrics["purity"]["r2"],
            "yield_mae_vs_truth": gt_metrics["yield"]["mae"],
            "purity_mae_vs_truth": gt_metrics["purity"]["mae"],
        },
        learned_coefficients=learned_coeffs,
        optimization=opt_result,
        all_checks_pass=all_pass,
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )

    return {
        "n_train": n_train,
        "n_test": n_test,
        "learned_coeffs": learned_coeffs,
        "true_coeffs": {"yield": {"weights": true_w["yield"], "intercept": true_i["yield"]},
                        "purity": {"weights": true_w["purity"], "intercept": true_i["purity"]}},
        "coeff_errors": coeff_errors,
        "gt_metrics": gt_metrics,
        "noisy_metrics": noisy_metrics,
        "optimization": opt_result,
        "chart": chart,
        "checks": checks,
        "all_pass": all_pass,
    }


# ── Comparison analysis ───────────────────────────────────────────────────

def run_comparison_analysis(n_train: int = 300, n_test: int = 100):
    """Run both analyses and produce a comparison. Returns a dict."""
    t_start = time.time()
    np.random.seed(42)

    train_in, train_out = collect_observations(n_train)
    test_in, test_out = collect_observations(n_test)
    gt_test = ground_truth(test_in)

    # Explicit model predictions (noise-free)
    explicit_preds = gt_test  # deterministic

    # Modelless predictor
    models = []
    for i in range(2):
        m = LinearRegression()
        m.fit(train_in, train_out[:, i])
        models.append(m)
    learned_preds = np.column_stack([m.predict(test_in) for m in models])

    # Metrics
    comparison = {}
    for i, name in enumerate(OUTPUT_NAMES):
        comparison[name] = {
            "learned_r2": float(r2_score(gt_test[:, i], learned_preds[:, i])),
            "learned_mae": float(mean_absolute_error(gt_test[:, i], learned_preds[:, i])),
        }

    # Optimization comparison
    # Explicit
    c_exp = -YIELD_W
    A_exp = [-PURITY_W]
    b_exp = [-(88.0 - PURITY_I)]
    r_exp = linprog(c_exp, A_ub=A_exp, b_ub=b_exp, bounds=INPUT_RANGES, method="highs")

    # Learned
    yield_w = np.array(models[0].coef_)
    purity_w = np.array(models[1].coef_)
    purity_i = float(models[1].intercept_)
    c_lrn = -yield_w
    A_lrn = [-purity_w]
    b_lrn = [-(88.0 - purity_i)]
    r_lrn = linprog(c_lrn, A_ub=A_lrn, b_ub=b_lrn, bounds=INPUT_RANGES, method="highs")

    opt_comparison = None
    if r_exp.success and r_lrn.success:
        gt_exp = ground_truth(r_exp.x.reshape(1, -1))[0]
        gt_lrn = ground_truth(r_lrn.x.reshape(1, -1))[0]
        lrn_pred = np.column_stack([m.predict(r_lrn.x.reshape(1, -1)) for m in models])[0]
        opt_comparison = {
            "explicit": {"inputs": r_exp.x.tolist(), "yield": float(gt_exp[0]), "purity": float(gt_exp[1])},
            "learned": {"inputs": r_lrn.x.tolist(), "yield": float(lrn_pred[0]), "purity": float(lrn_pred[1]),
                        "gt_yield": float(gt_lrn[0]), "gt_purity": float(gt_lrn[1])},
            "yield_diff": float(abs(gt_exp[0] - lrn_pred[0])),
        }

    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Explicit vs. Modelless — Comparison", fontsize=13, fontweight="bold")

    # Yield comparison
    ax = axes[0]
    ax.scatter(explicit_preds[:, 0], learned_preds[:, 0], s=15, alpha=0.6, c="#9467bd")
    lo, hi = explicit_preds[:, 0].min(), explicit_preds[:, 0].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect")
    ax.set_xlabel("Explicit Model Yield")
    ax.set_ylabel("Modelless Predictor Yield")
    ax.set_title(f"Yield Agreement\nR²={comparison['yield']['learned_r2']:.6f}")
    ax.legend()

    # Purity comparison
    ax = axes[1]
    ax.scatter(explicit_preds[:, 1], learned_preds[:, 1], s=15, alpha=0.6, c="#8c564b")
    lo, hi = explicit_preds[:, 1].min(), explicit_preds[:, 1].max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect")
    ax.set_xlabel("Explicit Model Purity")
    ax.set_ylabel("Modelless Predictor Purity")
    ax.set_title(f"Purity Agreement\nR²={comparison['purity']['learned_r2']:.6f}")
    ax.legend()

    # Error distribution
    ax = axes[2]
    diff = learned_preds - explicit_preds
    ax.hist(diff[:, 0], bins=25, alpha=0.6, label="Yield error", color="#1f77b4")
    ax.hist(diff[:, 1], bins=25, alpha=0.6, label="Purity error", color="#ff7f0e")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Prediction Error (learned − explicit)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    chart = _fig_to_base64(fig)

    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="comparison",
        data_type="synthetic",
        explicit_model_source="TCP server (127.0.0.1:9100)",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"train": n_train, "test": n_test},
        train_test_split={"train": n_train, "test": n_test},
        random_seed=42,
        noise_level=NOISE_STD,
        constraints_used={"min_purity": 88.0},
        metrics=comparison,
        optimization=opt_comparison,
        all_checks_pass=None,
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )

    return {
        "n_train": n_train,
        "n_test": n_test,
        "comparison": comparison,
        "optimization": opt_comparison,
        "chart": chart,
    }


# ── Sample-size sensitivity ───────────────────────────────────────────────

SENSITIVITY_SIZES = [5, 10, 25, 50, 100, 200]
SENSITIVITY_TEST = 100


def run_sensitivity_analysis():
    """Sweep over sample sizes and measure prediction accuracy. Returns a dict."""
    from sklearn.metrics import mean_squared_error

    t_start = time.time()
    np.random.seed(42)

    max_train = max(SENSITIVITY_SIZES)
    pool_in, pool_out = collect_observations(max_train)
    test_in, test_out = collect_observations(SENSITIVITY_TEST)
    gt_test = ground_truth(test_in)

    rows = []
    for n in SENSITIVITY_SIZES:
        train_in = pool_in[:n]
        train_out = pool_out[:n]

        models = []
        for i in range(2):
            m = LinearRegression()
            m.fit(train_in, train_out[:, i])
            models.append(m)
        preds = np.column_stack([m.predict(test_in) for m in models])

        row = {"sample_size": n}
        for i, name in enumerate(OUTPUT_NAMES):
            row[f"{name}_mae"] = float(mean_absolute_error(gt_test[:, i], preds[:, i]))
            row[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(gt_test[:, i], preds[:, i])))
            row[f"{name}_r2"] = float(r2_score(gt_test[:, i], preds[:, i]))
        rows.append(row)

    # Build chart
    sizes = [r["sample_size"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sample-Size Sensitivity", fontsize=13, fontweight="bold")

    for ax, metric, label in zip(
        axes,
        ["r2", "mae", "rmse"],
        ["R² (higher is better)", "MAE (lower is better)", "RMSE (lower is better)"],
    ):
        for name, color in [("yield", "#1f77b4"), ("purity", "#ff7f0e")]:
            vals = [r[f"{name}_{metric}"] for r in rows]
            ax.plot(sizes, vals, "o-", color=color, label=name.capitalize(), linewidth=2, markersize=6)
        ax.set_xlabel("Training Samples")
        ax.set_ylabel(metric.upper())
        ax.set_title(label)
        ax.legend()
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    chart = _fig_to_base64(fig)

    # Determine where the model becomes "useful" (R² ≥ 0.99)
    thresholds = {}
    for name in OUTPUT_NAMES:
        for r in rows:
            if r[f"{name}_r2"] >= 0.99:
                thresholds[name] = r["sample_size"]
                break
        else:
            thresholds[name] = None

    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="sample_size_sensitivity",
        data_type="synthetic",
        explicit_model_source="TCP server (127.0.0.1:9100)",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"pool": max_train, "test": SENSITIVITY_TEST},
        train_test_split={"sample_sizes": SENSITIVITY_SIZES, "test": SENSITIVITY_TEST},
        random_seed=42,
        noise_level=NOISE_STD,
        constraints_used=None,
        metrics={str(r["sample_size"]): {k: v for k, v in r.items() if k != "sample_size"} for r in rows},
        all_checks_pass=None,
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )

    return {
        "sample_sizes": SENSITIVITY_SIZES,
        "test_size": SENSITIVITY_TEST,
        "rows": rows,
        "chart": chart,
        "thresholds": thresholds,
    }


# ── Residual / error analysis ─────────────────────────────────────────────

RESIDUAL_TRAIN = 300
RESIDUAL_TEST = 200


def run_residual_analysis():
    """Full residual analysis: per-observation errors, summary stats, plots."""
    from sklearn.metrics import mean_squared_error

    t_start = time.time()
    np.random.seed(42)

    train_in, train_out = collect_observations(RESIDUAL_TRAIN)
    test_in, test_out = collect_observations(RESIDUAL_TEST)
    gt_test = ground_truth(test_in)

    # Train predictor
    models = []
    for i in range(2):
        m = LinearRegression()
        m.fit(train_in, train_out[:, i])
        models.append(m)
    preds = np.column_stack([m.predict(test_in) for m in models])

    # Residuals against noise-free ground truth
    residuals = gt_test - preds

    # Per-observation rows (for the table)
    obs_rows = []
    for j in range(len(test_in)):
        row = {"obs": j}
        for k, name in enumerate(INPUT_NAMES):
            row[f"input_{name}"] = round(float(test_in[j, k]), 4)
        for k, name in enumerate(OUTPUT_NAMES):
            row[f"actual_{name}"] = round(float(gt_test[j, k]), 4)
            row[f"predicted_{name}"] = round(float(preds[j, k]), 4)
            row[f"residual_{name}"] = round(float(residuals[j, k]), 4)
        obs_rows.append(row)

    # Summary statistics
    summary = {}
    for k, name in enumerate(OUTPUT_NAMES):
        r = residuals[:, k]
        abs_r = np.abs(r)
        summary[name] = {
            "mean_residual": round(float(r.mean()), 6),
            "mean_abs_residual": round(float(abs_r.mean()), 6),
            "max_abs_residual": round(float(abs_r.max()), 6),
            "p50_abs_error": round(float(np.percentile(abs_r, 50)), 6),
            "p90_abs_error": round(float(np.percentile(abs_r, 90)), 6),
            "p95_abs_error": round(float(np.percentile(abs_r, 95)), 6),
            "rmse": round(float(np.sqrt(mean_squared_error(gt_test[:, k], preds[:, k]))), 6),
            "r2": round(float(r2_score(gt_test[:, k], preds[:, k])), 6),
        }

    # ── Charts (3 × 2 grid) ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("Residual / Error Analysis", fontsize=13, fontweight="bold")

    colors = {"yield": "#1f77b4", "purity": "#ff7f0e"}
    for col, name in enumerate(OUTPUT_NAMES):
        k = col
        c = colors[name]
        r = residuals[:, k]

        # Row 0: Actual vs Predicted
        ax = axes[0, col]
        ax.scatter(gt_test[:, k], preds[:, k], s=12, alpha=0.5, c=c)
        lo, hi = gt_test[:, k].min(), gt_test[:, k].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"Actual {name.capitalize()}")
        ax.set_ylabel(f"Predicted {name.capitalize()}")
        ax.set_title(f"{name.capitalize()}: Actual vs Predicted (R²={summary[name]['r2']:.6f})")

        # Row 1: Residual vs Predicted
        ax = axes[1, col]
        ax.scatter(preds[:, k], r, s=12, alpha=0.5, c=c)
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xlabel(f"Predicted {name.capitalize()}")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title(f"{name.capitalize()}: Residual vs Predicted")

    # Row 2: Residual vs each input variable (combined in 2 subplots)
    for col, name in enumerate(OUTPUT_NAMES):
        ax = axes[2, col]
        k = col
        r = residuals[:, k]
        for i_idx, i_name in enumerate(INPUT_NAMES):
            ax.scatter(test_in[:, i_idx], r, s=8, alpha=0.35,
                       label=i_name.replace("_", " ").title())
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xlabel("Input Variable Value")
        ax.set_ylabel("Residual")
        ax.set_title(f"{name.capitalize()} Residual vs Inputs")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart = _fig_to_base64(fig)

    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="residual_analysis",
        data_type="synthetic",
        explicit_model_source="TCP server (127.0.0.1:9100)",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"train": RESIDUAL_TRAIN, "test": RESIDUAL_TEST},
        train_test_split={"train": RESIDUAL_TRAIN, "test": RESIDUAL_TEST},
        random_seed=42,
        noise_level=NOISE_STD,
        constraints_used=None,
        metrics=summary,
        all_checks_pass=None,
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=["(base64 embedded)"],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )

    return {
        "n_train": RESIDUAL_TRAIN,
        "n_test": RESIDUAL_TEST,
        "summary": summary,
        "obs_rows": obs_rows,
        "chart": chart,
    }


# ── Optimization agreement ─────────────────────────────────────────────────

OPT_TRAIN = 300
OPT_MIN_PURITY = 88.0


def run_optimization_analysis():
    """Compare explicit vs learned optimizer recommendations."""
    t_start = time.time()
    np.random.seed(42)

    train_in, train_out = collect_observations(OPT_TRAIN)

    # Train modelless predictor
    models = []
    for i in range(2):
        m = LinearRegression()
        m.fit(train_in, train_out[:, i])
        models.append(m)

    learned_coeffs = {
        OUTPUT_NAMES[i]: {
            "weights": models[i].coef_.tolist(),
            "intercept": float(models[i].intercept_),
        }
        for i in range(2)
    }

    # Explicit optimizer (known coefficients)
    c_exp = -YIELD_W
    A_exp = [-PURITY_W]
    b_exp = [-(OPT_MIN_PURITY - PURITY_I)]
    r_exp = linprog(c_exp, A_ub=A_exp, b_ub=b_exp, bounds=INPUT_RANGES, method="highs")

    # Learned optimizer
    lrn_yield_w = np.array(learned_coeffs["yield"]["weights"])
    lrn_purity_w = np.array(learned_coeffs["purity"]["weights"])
    lrn_purity_i = learned_coeffs["purity"]["intercept"]
    c_lrn = -lrn_yield_w
    A_lrn = [-lrn_purity_w]
    b_lrn = [-(OPT_MIN_PURITY - lrn_purity_i)]
    r_lrn = linprog(c_lrn, A_ub=A_lrn, b_ub=b_lrn, bounds=INPUT_RANGES, method="highs")

    if not r_exp.success or not r_lrn.success:
        return {"error": "One or both optimizers failed."}

    # Predictions at optimal points
    exp_x = r_exp.x
    lrn_x = r_lrn.x
    exp_yield = float(exp_x @ YIELD_W + YIELD_I)
    exp_purity = float(exp_x @ PURITY_W + PURITY_I)
    lrn_pred = np.column_stack([m.predict(lrn_x.reshape(1, -1)) for m in models])[0]
    lrn_yield = float(lrn_pred[0])
    lrn_purity = float(lrn_pred[1])

    # Ground truth at learned optimum (to check real-world performance)
    gt_at_lrn = ground_truth(lrn_x.reshape(1, -1))[0]

    # Build comparison rows
    rows = []
    for i, name in enumerate(INPUT_NAMES):
        rows.append({
            "variable": name,
            "explicit": round(exp_x[i], 4),
            "learned": round(lrn_x[i], 4),
            "diff": round(abs(exp_x[i] - lrn_x[i]), 4),
        })
    rows.append({"variable": "predicted_yield", "explicit": round(exp_yield, 4),
                 "learned": round(lrn_yield, 4), "diff": round(abs(exp_yield - lrn_yield), 4)})
    rows.append({"variable": "predicted_purity", "explicit": round(exp_purity, 4),
                 "learned": round(lrn_purity, 4), "diff": round(abs(exp_purity - lrn_purity), 4)})

    # Match score: 1.0 = perfect, penalised by normalised input distance
    ranges = [b[1] - b[0] for b in INPUT_RANGES]
    norm_diffs = [abs(exp_x[i] - lrn_x[i]) / r for i, r in enumerate(ranges)]
    match_score = round(max(0.0, 1.0 - sum(norm_diffs) / len(norm_diffs)), 6)

    # Bar chart comparing the two recommendations
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Optimization Agreement  (match score = {match_score:.4f})",
                 fontsize=13, fontweight="bold")

    # Input comparison
    ax = axes[0]
    x_pos = np.arange(len(INPUT_NAMES))
    w = 0.35
    ax.bar(x_pos - w / 2, exp_x, w, label="Explicit", color="#2ca02c", alpha=0.8)
    ax.bar(x_pos + w / 2, lrn_x, w, label="Learned", color="#d62728", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace("_", "\n") for n in INPUT_NAMES], fontsize=8)
    ax.set_title("Optimal Inputs")
    ax.legend(fontsize=8)

    # Output comparison
    ax = axes[1]
    labels = ["Yield", "Purity"]
    exp_vals = [exp_yield, exp_purity]
    lrn_vals = [lrn_yield, lrn_purity]
    x_pos = np.arange(2)
    ax.bar(x_pos - w / 2, exp_vals, w, label="Explicit", color="#2ca02c", alpha=0.8)
    ax.bar(x_pos + w / 2, lrn_vals, w, label="Learned", color="#d62728", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title("Predicted Outputs at Optimum")
    ax.legend(fontsize=8)
    ax.axhline(OPT_MIN_PURITY, color="orange", ls="--", lw=1.5, label=f"Min purity = {OPT_MIN_PURITY}")

    # Normalised difference radar-style bar
    ax = axes[2]
    all_labels = INPUT_NAMES + ["yield", "purity"]
    all_diffs = [abs(exp_x[i] - lrn_x[i]) for i in range(3)] + [
        abs(exp_yield - lrn_yield), abs(exp_purity - lrn_purity)]
    colors = ["#1f77b4"] * 3 + ["#ff7f0e"] * 2
    ax.barh(range(len(all_labels)), all_diffs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels([n.replace("_", " ").title() for n in all_labels], fontsize=9)
    ax.set_xlabel("Absolute Difference")
    ax.set_title("Explicit vs Learned Differences")
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    chart = _fig_to_base64(fig)

    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="optimization_agreement",
        data_type="synthetic",
        explicit_model_source="known coefficients + TCP server",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"train": OPT_TRAIN},
        train_test_split=None,
        random_seed=42,
        noise_level=NOISE_STD,
        constraints_used={"min_purity": OPT_MIN_PURITY},
        metrics={"optimization_match_score": match_score},
        all_checks_pass=match_score >= 0.95,
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=["(base64 embedded)"],
        duration_seconds=round(duration, 2),
    )

    return {
        "n_train": OPT_TRAIN,
        "min_purity": OPT_MIN_PURITY,
        "rows": rows,
        "match_score": match_score,
        "chart": chart,
        "explicit": {"inputs": exp_x.tolist(), "yield": exp_yield, "purity": exp_purity},
        "learned": {"inputs": lrn_x.tolist(), "yield": lrn_yield, "purity": lrn_purity,
                     "gt_yield": float(gt_at_lrn[0]), "gt_purity": float(gt_at_lrn[1])},
    }


# ── Constraint verification ───────────────────────────────────────────────

CONSTRAINT_TRAIN = 300
CONSTRAINT_DEFS = [
    {"name": "purity_min", "kind": "output", "output": "purity", "op": ">=", "threshold": 88.0},
    {"name": "yield_positive", "kind": "output", "output": "yield", "op": ">", "threshold": 0.0},
    {"name": "temperature_range", "kind": "input", "input": "temperature", "idx": 0,
     "low": 20.0, "high": 80.0},
    {"name": "flow_rate_range", "kind": "input", "input": "flow_rate", "idx": 1,
     "low": 1.0, "high": 10.0},
    {"name": "concentration_range", "kind": "input", "input": "concentration", "idx": 2,
     "low": 0.1, "high": 5.0},
]


def run_constraint_analysis():
    """Verify modelless optimizer recommendations against explicit baseline."""
    t_start = time.time()
    np.random.seed(42)

    train_in, train_out = collect_observations(CONSTRAINT_TRAIN)

    # Train modelless predictor
    models = []
    for i in range(2):
        m = LinearRegression()
        m.fit(train_in, train_out[:, i])
        models.append(m)

    # Run learned optimizer
    lrn_yield_w = np.array(models[0].coef_)
    lrn_purity_w = np.array(models[1].coef_)
    lrn_purity_i = float(models[1].intercept_)
    c_lrn = -lrn_yield_w
    A_lrn = [-lrn_purity_w]
    b_lrn = [-(88.0 - lrn_purity_i)]
    r_lrn = linprog(c_lrn, A_ub=A_lrn, b_ub=b_lrn, bounds=INPUT_RANGES, method="highs")

    if not r_lrn.success:
        return {"error": "Modelless optimizer failed."}

    rec = r_lrn.x
    ml_pred = np.column_stack([m.predict(rec.reshape(1, -1)) for m in models])[0]
    ml_yield = float(ml_pred[0])
    ml_purity = float(ml_pred[1])

    # Baseline verification (noise-free ground truth)
    gt = ground_truth(rec.reshape(1, -1))[0]
    bl_yield = float(gt[0])
    bl_purity = float(gt[1])
    baseline_outputs = {"yield": bl_yield, "purity": bl_purity}

    # Check constraints against baseline
    checks = []
    for c in CONSTRAINT_DEFS:
        if c["kind"] == "output":
            val = baseline_outputs[c["output"]]
            th = c["threshold"]
            if c["op"] == ">=":
                ok = val >= th
            else:
                ok = val > th
            detail = f"{c['output']}={val:.4f} {c['op']} {th}"
        else:
            val = float(rec[c["idx"]])
            ok = c["low"] <= val <= c["high"]
            detail = f"{c['input']}={val:.4f} in [{c['low']}, {c['high']}]"
        checks.append({
            "name": c["name"],
            "detail": detail,
            "value": round(val, 4),
            "passed": ok,
        })

    recommendation_safe = all(ch["passed"] for ch in checks)

    # Recommended inputs
    rec_inputs = {INPUT_NAMES[i]: round(float(rec[i]), 4) for i in range(3)}

    # Chart: bar comparison + pass/fail indicators
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    safe_color = "#2ca02c" if recommendation_safe else "#d62728"
    fig.suptitle(f"Constraint Verification  —  recommendation_safe = {recommendation_safe}",
                 fontsize=13, fontweight="bold", color=safe_color)

    # Output comparison
    ax = axes[0]
    labels = ["Yield", "Purity"]
    ml_vals = [ml_yield, ml_purity]
    bl_vals = [bl_yield, bl_purity]
    x_pos = np.arange(2)
    w = 0.35
    ax.bar(x_pos - w / 2, ml_vals, w, label="Modelless Predicted", color="#d62728", alpha=0.8)
    ax.bar(x_pos + w / 2, bl_vals, w, label="Baseline Verified", color="#2ca02c", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title("Predicted vs Baseline Outputs")
    ax.legend(fontsize=8)
    ax.axhline(88.0, color="orange", ls="--", lw=1.5, label="Purity ≥ 88%")

    # Constraint pass/fail
    ax = axes[1]
    names = [ch["name"].replace("_", " ").title() for ch in checks]
    colors = ["#2ca02c" if ch["passed"] else "#d62728" for ch in checks]
    ax.barh(range(len(checks)), [1] * len(checks), color=colors, alpha=0.8,
            edgecolor="white", linewidth=2)
    ax.set_yticks(range(len(checks)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    ax.set_title("Constraint Pass / Fail")
    for i, ch in enumerate(checks):
        label = "PASS" if ch["passed"] else "FAIL"
        ax.text(0.5, i, label, ha="center", va="center", fontweight="bold",
                fontsize=11, color="white")
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    chart = _fig_to_base64(fig)

    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="constraint_verification",
        data_type="synthetic",
        explicit_model_source="known coefficients + TCP server",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"train": CONSTRAINT_TRAIN},
        train_test_split=None,
        random_seed=42,
        noise_level=NOISE_STD,
        constraints_used={c["name"]: c for c in CONSTRAINT_DEFS},
        metrics={"recommendation_safe": recommendation_safe},
        all_checks_pass=recommendation_safe,
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=["(base64 embedded)"],
        duration_seconds=round(duration, 2),
    )

    return {
        "n_train": CONSTRAINT_TRAIN,
        "rec_inputs": rec_inputs,
        "modelless": {"yield": ml_yield, "purity": ml_purity},
        "baseline": {"yield": bl_yield, "purity": bl_purity},
        "checks": checks,
        "recommendation_safe": recommendation_safe,
        "chart": chart,
    }


# ── Noise / process variation sensitivity ──────────────────────────────────

NOISE_TRAIN = 300
NOISE_TEST = 200
NOISE_LEVELS = [
    ("none",    0.0),
    ("low",     0.25),
    ("medium",  0.5),
    ("high",    1.0),
    ("extreme", 2.0),
]


def _generate_local(n, noise_std, rng):
    """Generate n observations client-side with specified noise."""
    inputs = np.column_stack([
        rng.uniform(lo, hi, n) for lo, hi in INPUT_RANGES
    ])
    gt_y = inputs @ YIELD_W + YIELD_I
    gt_p = inputs @ PURITY_W + PURITY_I
    noisy_y = gt_y + rng.normal(0, noise_std, n) if noise_std > 0 else gt_y.copy()
    noisy_p = gt_p + rng.normal(0, noise_std, n) if noise_std > 0 else gt_p.copy()
    return inputs, np.column_stack([noisy_y, noisy_p]), np.column_stack([gt_y, gt_p])


def run_noise_analysis():
    """Sweep noise levels, train predictors, evaluate stability."""
    from sklearn.metrics import mean_squared_error

    t_start = time.time()
    rng = np.random.default_rng(42)

    # Fixed noise-free test set
    test_in, _, gt_test = _generate_local(NOISE_TEST, 0.0, rng)

    # Explicit optimum (reference)
    c_exp = -YIELD_W
    A_exp = [-PURITY_W]
    b_exp = [-(88.0 - PURITY_I)]
    r_exp = linprog(c_exp, A_ub=A_exp, b_ub=b_exp, bounds=INPUT_RANGES, method="highs")
    exp_x = r_exp.x if r_exp.success else None

    rows = []
    for label, sigma in NOISE_LEVELS:
        train_in, train_out, _ = _generate_local(NOISE_TRAIN, sigma, rng)

        models = []
        for i in range(2):
            m = LinearRegression()
            m.fit(train_in, train_out[:, i])
            models.append(m)
        preds = np.column_stack([m.predict(test_in) for m in models])

        row = {"noise_label": label, "noise_std": sigma}
        for i, name in enumerate(OUTPUT_NAMES):
            row[f"{name}_mae"] = round(float(mean_absolute_error(gt_test[:, i], preds[:, i])), 6)
            row[f"{name}_rmse"] = round(float(np.sqrt(mean_squared_error(gt_test[:, i], preds[:, i]))), 6)
            row[f"{name}_r2"] = round(float(r2_score(gt_test[:, i], preds[:, i])), 6)

        # Optimization agreement
        yield_w = np.array(models[0].coef_)
        purity_w = np.array(models[1].coef_)
        purity_i = float(models[1].intercept_)
        c_lrn = -yield_w
        A_lrn = [-purity_w]
        b_lrn = [-(88.0 - purity_i)]
        r_lrn = linprog(c_lrn, A_ub=A_lrn, b_ub=b_lrn, bounds=INPUT_RANGES, method="highs")

        if r_lrn.success and exp_x is not None:
            ranges = [b[1] - b[0] for b in INPUT_RANGES]
            norm_diffs = [abs(exp_x[j] - r_lrn.x[j]) / r for j, r in enumerate(ranges)]
            row["opt_match_score"] = round(max(0.0, 1.0 - sum(norm_diffs) / len(norm_diffs)), 6)
            gt_purity_at_rec = float(r_lrn.x @ PURITY_W + PURITY_I)
            row["constraint_safe"] = gt_purity_at_rec >= 88.0
        else:
            row["opt_match_score"] = 0.0
            row["constraint_safe"] = False

        rows.append(row)

    # Charts: 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Noise / Process Variation Sensitivity", fontsize=13, fontweight="bold")
    sigmas = [r["noise_std"] for r in rows]
    labels_x = [r["noise_label"] for r in rows]

    # R²
    ax = axes[0, 0]
    for name, color in [("yield", "#1f77b4"), ("purity", "#ff7f0e")]:
        vals = [r[f"{name}_r2"] for r in rows]
        ax.plot(range(len(sigmas)), vals, "o-", color=color, label=name.capitalize(), lw=2, ms=7)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("R²")
    ax.set_title("R² vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[0, 1]
    for name, color in [("yield", "#1f77b4"), ("purity", "#ff7f0e")]:
        vals = [r[f"{name}_mae"] for r in rows]
        ax.plot(range(len(sigmas)), vals, "o-", color=color, label=name.capitalize(), lw=2, ms=7)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Match score
    ax = axes[1, 0]
    match_vals = [r["opt_match_score"] for r in rows]
    colors_bar = ["#2ca02c" if v >= 0.95 else "#ff7f0e" if v >= 0.80 else "#d62728" for v in match_vals]
    ax.bar(range(len(sigmas)), match_vals, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("Match Score")
    ax.set_title("Optimization Agreement")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color="gray", ls="--", lw=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Constraint safety
    ax = axes[1, 1]
    safe_vals = [1 if r["constraint_safe"] else 0 for r in rows]
    bar_colors = ["#2ca02c" if s else "#d62728" for s in safe_vals]
    ax.bar(range(len(sigmas)), safe_vals, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("Safe (1) / Unsafe (0)")
    ax.set_title("Constraint Satisfaction")
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart = _fig_to_base64(fig)

    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="noise_sensitivity",
        data_type="synthetic_variable_noise",
        explicit_model_source="known coefficients (client-side)",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"train": NOISE_TRAIN, "test": NOISE_TEST},
        train_test_split={"train": NOISE_TRAIN, "test": NOISE_TEST},
        random_seed=42,
        noise_level={label: sigma for label, sigma in NOISE_LEVELS},
        constraints_used={"min_purity": 88.0},
        metrics={r["noise_label"]: {k: v for k, v in r.items()
                 if k not in ("noise_label",)} for r in rows},
        all_checks_pass=all(r["constraint_safe"] for r in rows),
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )

    return {
        "n_train": NOISE_TRAIN,
        "n_test": NOISE_TEST,
        "noise_levels": NOISE_LEVELS,
        "rows": rows,
        "chart": chart,
        "all_safe": all(r["constraint_safe"] for r in rows),
    }


# ── POV Summary (unified evaluation) ──────────────────────────────────────

POV_TRAIN = 300
POV_TEST = 200
POV_SAMPLE_SIZES = [5, 10, 25, 50, 100, 200]
POV_NOISE_LEVELS = [
    ("none", 0.0), ("low", 0.25), ("medium", 0.5),
    ("high", 1.0), ("extreme", 2.0),
]


def run_pov_summary():
    """Single-pass proof-of-value evaluation across all analysis dimensions."""
    from sklearn.metrics import mean_squared_error

    t_start = time.time()
    rng = np.random.default_rng(42)
    np.random.seed(42)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # ── 1. Collect data ──────────────────────────────────────────────────
    train_in, train_out = collect_observations(POV_TRAIN)
    test_in, test_out = collect_observations(POV_TEST)
    gt_test = ground_truth(test_in)

    # ── 2. Train modelless predictor ─────────────────────────────────────
    models = []
    for i in range(2):
        m = LinearRegression()
        m.fit(train_in, train_out[:, i])
        models.append(m)
    preds = np.column_stack([m.predict(test_in) for m in models])

    learned_coeffs = {
        OUTPUT_NAMES[i]: {
            "weights": models[i].coef_.tolist(),
            "intercept": float(models[i].intercept_),
        }
        for i in range(2)
    }

    # ── Section 1: Run Summary ───────────────────────────────────────────
    run_summary = {
        "run_id": run_id,
        "data_type": "synthetic (TCP server + client-side noise sweep)",
        "sample_size": POV_TRAIN,
        "model_type": "LinearRegression (scikit-learn)",
        "noise_level": f"σ = {NOISE_STD} (server default)",
        "train_test_split": f"{POV_TRAIN} train / {POV_TEST} test",
    }

    # ── Section 2: Prediction Accuracy ───────────────────────────────────
    residuals = gt_test - preds
    accuracy = {}
    for i, name in enumerate(OUTPUT_NAMES):
        accuracy[name] = {
            "r2": round(float(r2_score(gt_test[:, i], preds[:, i])), 6),
            "mae": round(float(mean_absolute_error(gt_test[:, i], preds[:, i])), 6),
            "rmse": round(float(np.sqrt(mean_squared_error(gt_test[:, i], preds[:, i]))), 6),
        }

    # Chart: actual vs predicted (1×2)
    fig_acc, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig_acc.suptitle("Prediction Accuracy", fontsize=12, fontweight="bold")
    for col, (name, color) in enumerate(zip(OUTPUT_NAMES, ["#1f77b4", "#ff7f0e"])):
        ax = axes[col]
        ax.scatter(gt_test[:, col], preds[:, col], s=10, alpha=0.5, c=color)
        lo, hi = gt_test[:, col].min(), gt_test[:, col].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"Actual {name.capitalize()}")
        ax.set_ylabel(f"Predicted {name.capitalize()}")
        ax.set_title(f"R² = {accuracy[name]['r2']:.6f}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    chart_accuracy = _fig_to_base64(fig_acc)

    # ── Section 3: Sample-Size Sensitivity ───────────────────────────────
    # Use a large pool from the already-collected training data
    max_pool = max(POV_SAMPLE_SIZES)
    pool_in, pool_out = collect_observations(max_pool)
    sens_test_in, _ = collect_observations(100)
    gt_sens = ground_truth(sens_test_in)

    sensitivity_rows = []
    for n in POV_SAMPLE_SIZES:
        sub_in = pool_in[:n]
        sub_out = pool_out[:n]
        sub_models = []
        for i in range(2):
            m = LinearRegression()
            m.fit(sub_in, sub_out[:, i])
            sub_models.append(m)
        sub_preds = np.column_stack([m.predict(sens_test_in) for m in sub_models])
        row = {"n": n}
        for i, name in enumerate(OUTPUT_NAMES):
            row[f"{name}_r2"] = round(float(r2_score(gt_sens[:, i], sub_preds[:, i])), 6)
            row[f"{name}_mae"] = round(float(mean_absolute_error(gt_sens[:, i], sub_preds[:, i])), 6)
        sensitivity_rows.append(row)

    # Thresholds
    sens_thresholds = {}
    for name in OUTPUT_NAMES:
        for row in sensitivity_rows:
            if row[f"{name}_r2"] >= 0.99:
                sens_thresholds[name] = row["n"]
                break
        else:
            sens_thresholds[name] = None

    # Chart: R² trend (1×1)
    fig_sens, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_title("R² by Sample Size", fontsize=12, fontweight="bold")
    sizes = [r["n"] for r in sensitivity_rows]
    for name, color in [("yield", "#1f77b4"), ("purity", "#ff7f0e")]:
        vals = [r[f"{name}_r2"] for r in sensitivity_rows]
        ax.plot(sizes, vals, "o-", color=color, label=name.capitalize(), lw=2, ms=6)
    ax.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("R²")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    chart_sensitivity = _fig_to_base64(fig_sens)

    # ── Section 4: Residual Analysis ─────────────────────────────────────
    residual_summary = {}
    for i, name in enumerate(OUTPUT_NAMES):
        r = residuals[:, i]
        abs_r = np.abs(r)
        residual_summary[name] = {
            "mean": round(float(r.mean()), 6),
            "mean_abs": round(float(abs_r.mean()), 6),
            "max_abs": round(float(abs_r.max()), 6),
            "p50": round(float(np.percentile(abs_r, 50)), 6),
            "p90": round(float(np.percentile(abs_r, 90)), 6),
            "p95": round(float(np.percentile(abs_r, 95)), 6),
        }

    # Chart: residual vs predicted (1×2)
    fig_res, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig_res.suptitle("Residuals (actual − predicted)", fontsize=12, fontweight="bold")
    for col, (name, color) in enumerate(zip(OUTPUT_NAMES, ["#1f77b4", "#ff7f0e"])):
        ax = axes[col]
        ax.scatter(preds[:, col], residuals[:, col], s=10, alpha=0.5, c=color)
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xlabel(f"Predicted {name.capitalize()}")
        ax.set_ylabel("Residual")
        ax.set_title(f"{name.capitalize()}: mean={residual_summary[name]['mean']:+.4f}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    chart_residuals = _fig_to_base64(fig_res)

    # ── Section 5: Optimization Agreement ────────────────────────────────
    # Explicit optimizer
    c_exp = -YIELD_W
    A_exp = [-PURITY_W]
    b_exp = [-(88.0 - PURITY_I)]
    r_exp = linprog(c_exp, A_ub=A_exp, b_ub=b_exp, bounds=INPUT_RANGES, method="highs")

    # Learned optimizer
    lrn_yield_w = np.array(learned_coeffs["yield"]["weights"])
    lrn_purity_w = np.array(learned_coeffs["purity"]["weights"])
    lrn_purity_i = learned_coeffs["purity"]["intercept"]
    c_lrn = -lrn_yield_w
    A_lrn = [-lrn_purity_w]
    b_lrn = [-(88.0 - lrn_purity_i)]
    r_lrn = linprog(c_lrn, A_ub=A_lrn, b_ub=b_lrn, bounds=INPUT_RANGES, method="highs")

    opt_rows = []
    match_score = 0.0
    if r_exp.success and r_lrn.success:
        exp_x, lrn_x = r_exp.x, r_lrn.x
        exp_yield = float(exp_x @ YIELD_W + YIELD_I)
        exp_purity = float(exp_x @ PURITY_W + PURITY_I)
        lrn_pred = np.column_stack([m.predict(lrn_x.reshape(1, -1)) for m in models])[0]
        lrn_yield, lrn_purity = float(lrn_pred[0]), float(lrn_pred[1])

        for i, name in enumerate(INPUT_NAMES):
            opt_rows.append({"variable": name,
                             "explicit": round(exp_x[i], 4), "learned": round(lrn_x[i], 4),
                             "diff": round(abs(exp_x[i] - lrn_x[i]), 4)})
        opt_rows.append({"variable": "predicted_yield",
                         "explicit": round(exp_yield, 4), "learned": round(lrn_yield, 4),
                         "diff": round(abs(exp_yield - lrn_yield), 4)})
        opt_rows.append({"variable": "predicted_purity",
                         "explicit": round(exp_purity, 4), "learned": round(lrn_purity, 4),
                         "diff": round(abs(exp_purity - lrn_purity), 4)})

        ranges = [b[1] - b[0] for b in INPUT_RANGES]
        norm_diffs = [abs(exp_x[j] - lrn_x[j]) / r for j, r in enumerate(ranges)]
        match_score = round(max(0.0, 1.0 - sum(norm_diffs) / len(norm_diffs)), 6)

    # ── Section 6: Constraint Verification ───────────────────────────────
    constraint_checks = []
    if r_lrn.success:
        rec = r_lrn.x
        gt_at_rec = ground_truth(rec.reshape(1, -1))[0]
        bl_yield, bl_purity = float(gt_at_rec[0]), float(gt_at_rec[1])

        constraint_checks.append({
            "name": "Purity ≥ 88%",
            "value": round(bl_purity, 4),
            "threshold": "≥ 88.0",
            "passed": bl_purity >= 88.0,
        })
        constraint_checks.append({
            "name": "Yield > 0",
            "value": round(bl_yield, 4),
            "threshold": "> 0.0",
            "passed": bl_yield > 0.0,
        })
        for i, name in enumerate(INPUT_NAMES):
            lo, hi = INPUT_RANGES[i]
            val = round(float(rec[i]), 4)
            constraint_checks.append({
                "name": f"{name} in [{lo}, {hi}]",
                "value": val,
                "threshold": f"[{lo}, {hi}]",
                "passed": lo <= rec[i] <= hi,
            })
    recommendation_safe = len(constraint_checks) > 0 and all(c["passed"] for c in constraint_checks)

    # ── Section 7 (input): Noise robustness ──────────────────────────────
    noise_rows = []
    for label, sigma in POV_NOISE_LEVELS:
        n_train_in, n_train_out, _ = _generate_local(POV_TRAIN, sigma, rng)
        n_test_in, _, n_gt_test = _generate_local(POV_TEST, 0.0, rng)

        n_models = []
        for i in range(2):
            m = LinearRegression()
            m.fit(n_train_in, n_train_out[:, i])
            n_models.append(m)
        n_preds = np.column_stack([m.predict(n_test_in) for m in n_models])

        nr = {"label": label, "sigma": sigma}
        for i, name in enumerate(OUTPUT_NAMES):
            nr[f"{name}_r2"] = round(float(r2_score(n_gt_test[:, i], n_preds[:, i])), 6)
            nr[f"{name}_mae"] = round(float(mean_absolute_error(n_gt_test[:, i], n_preds[:, i])), 6)

        # Constraint check at this noise level
        n_yield_w = np.array(n_models[0].coef_)
        n_purity_w = np.array(n_models[1].coef_)
        n_purity_i = float(n_models[1].intercept_)
        c_n = -n_yield_w
        A_n = [-n_purity_w]
        b_n = [-(88.0 - n_purity_i)]
        r_n = linprog(c_n, A_ub=A_n, b_ub=b_n, bounds=INPUT_RANGES, method="highs")
        if r_n.success:
            gt_p = float(r_n.x @ PURITY_W + PURITY_I)
            nr["safe"] = gt_p >= 88.0
        else:
            nr["safe"] = False
        noise_rows.append(nr)

    # ── Section 7: POV Gate Summary ──────────────────────────────────────
    gates = []

    # Accuracy gate
    acc_pass = (accuracy["yield"]["r2"] >= 0.999 and accuracy["purity"]["r2"] >= 0.999)
    gates.append({
        "name": "Prediction Accuracy",
        "criterion": "R² ≥ 0.999 for both outputs vs ground truth",
        "status": "pass" if acc_pass else "fail",
        "detail": f"yield R²={accuracy['yield']['r2']:.6f}, purity R²={accuracy['purity']['r2']:.6f}",
    })

    # Sample efficiency gate
    eff_pass = all(sens_thresholds.get(n) is not None and sens_thresholds[n] <= 25
                   for n in OUTPUT_NAMES)
    gates.append({
        "name": "Sample Efficiency",
        "criterion": "R² ≥ 0.99 by n = 25 for both outputs",
        "status": "pass" if eff_pass else ("review" if any(
            sens_thresholds.get(n) is not None and sens_thresholds[n] <= 50
            for n in OUTPUT_NAMES) else "fail"),
        "detail": ", ".join(f"{n}: n={sens_thresholds.get(n, '>200')}" for n in OUTPUT_NAMES),
    })

    # Decision agreement gate
    dec_pass = match_score >= 0.95
    gates.append({
        "name": "Decision Agreement",
        "criterion": "Optimization match score ≥ 0.95",
        "status": "pass" if dec_pass else "review" if match_score >= 0.80 else "fail",
        "detail": f"match score = {match_score:.4f}",
    })

    # Constraint safety gate
    gates.append({
        "name": "Constraint Safety",
        "criterion": "All constraints pass under baseline verification",
        "status": "pass" if recommendation_safe else "fail",
        "detail": f"{sum(1 for c in constraint_checks if c['passed'])}/{len(constraint_checks)} passed",
    })

    # Robustness gate
    medium_row = next((r for r in noise_rows if r["label"] == "medium"), None)
    low_row = next((r for r in noise_rows if r["label"] == "low"), None)
    robust_r2 = (medium_row and medium_row["yield_r2"] >= 0.99
                 and medium_row["purity_r2"] >= 0.99)
    robust_safe = low_row["safe"] if low_row else False
    gates.append({
        "name": "Robustness",
        "criterion": "R² ≥ 0.99 at medium noise; constraints safe at low noise",
        "status": "pass" if (robust_r2 and robust_safe) else
                  "review" if robust_r2 else "fail",
        "detail": (f"medium R²: y={medium_row['yield_r2']:.6f} p={medium_row['purity_r2']:.6f}; "
                   f"low noise safe={low_row['safe']}" if medium_row and low_row else "N/A"),
    })

    overall = ("pass" if all(g["status"] == "pass" for g in gates) else
               "fail" if any(g["status"] == "fail" for g in gates) else "review")

    duration = time.time() - t_start
    write_manifest(
        OUTPUT_DIR,
        analysis_type="pov_summary",
        data_type="synthetic",
        explicit_model_source="TCP server + known coefficients",
        explicit_model_version="1.0.0",
        modelless_model_type="LinearRegression",
        sample_size={"train": POV_TRAIN, "test": POV_TEST},
        train_test_split={"train": POV_TRAIN, "test": POV_TEST},
        random_seed=42,
        noise_level=NOISE_STD,
        constraints_used={"min_purity": 88.0},
        metrics={
            "accuracy": accuracy,
            "optimization_match_score": match_score,
            "recommendation_safe": recommendation_safe,
            "pov_gate_overall": overall,
        },
        all_checks_pass=overall == "pass",
        plot_files=["(base64 embedded)"],
        metric_files=[],
        prediction_files=[],
        residual_files=[],
        optimization_files=[],
        duration_seconds=round(duration, 2),
    )

    return {
        "run_summary": run_summary,
        "accuracy": accuracy,
        "chart_accuracy": chart_accuracy,
        "sensitivity_rows": sensitivity_rows,
        "sens_thresholds": sens_thresholds,
        "chart_sensitivity": chart_sensitivity,
        "residual_summary": residual_summary,
        "chart_residuals": chart_residuals,
        "opt_rows": opt_rows,
        "match_score": match_score,
        "constraint_checks": constraint_checks,
        "recommendation_safe": recommendation_safe,
        "noise_rows": noise_rows,
        "gates": gates,
        "overall": overall,
        "duration": round(duration, 2),
    }
