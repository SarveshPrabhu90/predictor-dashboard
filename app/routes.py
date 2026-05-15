"""Flask routes — pages and API endpoints."""

from flask import Blueprint, render_template, jsonify

from . import analysis

bp = Blueprint("main", __name__)


@bp.route("/")
def home():
    server_up = analysis.server_is_available()
    return render_template("home.html", server_up=server_up)


@bp.route("/explicit")
def explicit():
    if not analysis.server_is_available():
        return render_template("error.html",
                               message="TCP server is not running. Start explicit-model on port 9100.")
    results = analysis.run_explicit_analysis(500)
    return render_template("explicit.html", r=results)


@bp.route("/predictor")
def predictor():
    if not analysis.server_is_available():
        return render_template("error.html",
                               message="TCP server is not running. Start explicit-model on port 9100.")
    results = analysis.run_predictor_analysis(300, 100)
    return render_template("predictor.html", r=results)


@bp.route("/comparison")
def comparison():
    if not analysis.server_is_available():
        return render_template("error.html",
                               message="TCP server is not running. Start explicit-model on port 9100.")
    results = analysis.run_comparison_analysis(300, 100)
    return render_template("comparison.html", r=results)


@bp.route("/sensitivity")
def sensitivity():
    if not analysis.server_is_available():
        return render_template("error.html",
                               message="TCP server is not running. Start explicit-model on port 9100.")
    results = analysis.run_sensitivity_analysis()
    return render_template("sensitivity.html", r=results)


@bp.route("/residuals")
def residuals():
    if not analysis.server_is_available():
        return render_template("error.html",
                               message="TCP server is not running. Start explicit-model on port 9100.")
    results = analysis.run_residual_analysis()
    return render_template("residuals.html", r=results)


@bp.route("/optimization")
def optimization():
    if not analysis.server_is_available():
        return render_template("error.html",
                               message="TCP server is not running. Start explicit-model on port 9100.")
    results = analysis.run_optimization_analysis()
    if "error" in results:
        return render_template("error.html", message=results["error"])
    return render_template("optimization.html", r=results)


@bp.route("/constraints")
def constraints():
    if not analysis.server_is_available():
        return render_template("error.html",
                               message="TCP server is not running. Start explicit-model on port 9100.")
    results = analysis.run_constraint_analysis()
    if "error" in results:
        return render_template("error.html", message=results["error"])
    return render_template("constraints.html", r=results)


# ── JSON API endpoints ─────────────────────────────────────────────────────

@bp.route("/api/status")
def api_status():
    return jsonify({"server_available": analysis.server_is_available()})


@bp.route("/api/explicit")
def api_explicit():
    if not analysis.server_is_available():
        return jsonify({"error": "TCP server not available"}), 503
    results = analysis.run_explicit_analysis(500)
    results.pop("chart", None)  # exclude large image from JSON
    return jsonify(results)


@bp.route("/api/predictor")
def api_predictor():
    if not analysis.server_is_available():
        return jsonify({"error": "TCP server not available"}), 503
    results = analysis.run_predictor_analysis(300, 100)
    results.pop("chart", None)
    return jsonify(results)


@bp.route("/api/comparison")
def api_comparison():
    if not analysis.server_is_available():
        return jsonify({"error": "TCP server not available"}), 503
    results = analysis.run_comparison_analysis(300, 100)
    results.pop("chart", None)
    return jsonify(results)


@bp.route("/api/sensitivity")
def api_sensitivity():
    if not analysis.server_is_available():
        return jsonify({"error": "TCP server not available"}), 503
    results = analysis.run_sensitivity_analysis()
    results.pop("chart", None)
    return jsonify(results)


@bp.route("/api/residuals")
def api_residuals():
    if not analysis.server_is_available():
        return jsonify({"error": "TCP server not available"}), 503
    results = analysis.run_residual_analysis()
    results.pop("chart", None)
    results.pop("obs_rows", None)  # large; use CSV endpoint if needed
    return jsonify(results)


@bp.route("/api/optimization")
def api_optimization():
    if not analysis.server_is_available():
        return jsonify({"error": "TCP server not available"}), 503
    results = analysis.run_optimization_analysis()
    results.pop("chart", None)
    return jsonify(results)


@bp.route("/api/constraints")
def api_constraints():
    if not analysis.server_is_available():
        return jsonify({"error": "TCP server not available"}), 503
    results = analysis.run_constraint_analysis()
    results.pop("chart", None)
    return jsonify(results)
