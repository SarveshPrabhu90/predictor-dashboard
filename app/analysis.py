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
