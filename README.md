# Predictor Dashboard

A Flask web application for validating, analyzing, and comparing the
model-free predictor proof-of-value. Connects live to the
[explicit-model](https://github.com/SarveshPrabhu90/explicit-model) TCP server
(port 9100), trains the
[modelless-predictor](https://github.com/SarveshPrabhu90/modelless-predictor)
from streamed observations, and presents statistical dashboards proving the two
approaches produce near-identical results.

## Dashboards

| Page | Route | Description |
|------|-------|-------------|
| **Home** | `/` | Architecture overview, live server status, feature cards, links to related repos |
| **POV Summary** | `/pov` | Unified evaluation story — accuracy, sensitivity, residuals, optimization, constraints, noise, gate verdicts |
| **Explicit Model** | `/explicit` | 500-sample validation — input/output distributions, noise analysis (σ ≈ 0.5), R² checks |
| **Modelless Predictor** | `/predictor` | Coefficient recovery, test-set R² > 0.999, LP optimization (max yield, purity ≥ 88%) |
| **Comparison** | `/comparison` | Side-by-side predictions & optimization — yield difference typically < 0.2% |
| **Sensitivity** | `/sensitivity` | Sample-size sweep — R², MAE, RMSE at n = 5, 10, 25, 50, 100, 200 |
| **Residuals** | `/residuals` | Per-observation error analysis — summary stats, actual vs predicted, residual plots |
| **Optimization** | `/optimization` | Explicit vs learned optimizer agreement — match score, input/output comparison |
| **Constraints** | `/constraints` | Verify modelless recommendations against baseline — pass/fail per constraint |
| **Noise** | `/noise` | Performance across 5 noise levels — R², MAE, opt match, constraint safety (no server needed) |

### API Endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/status` | TCP server connectivity check |
| `GET /api/explicit` | Full explicit-model validation results (JSON) |
| `GET /api/predictor` | Modelless predictor training & evaluation results (JSON) |
| `GET /api/comparison` | Side-by-side comparison results (JSON) |
| `GET /api/sensitivity` | Sample-size sensitivity sweep results (JSON) |
| `GET /api/residuals` | Residual summary statistics (JSON) |
| `GET /api/optimization` | Optimization agreement results (JSON) |
| `GET /api/constraints` | Constraint verification results (JSON) |
| `GET /api/noise` | Noise sensitivity sweep results (JSON) |
| `GET /api/pov` | Full POV summary with all gate verdicts (JSON) |

## Architecture

```
┌──────────────────────┐         TCP/IP          ┌──────────────────────┐
│   Explicit Model     │ ◄──────────────────────► │   This Dashboard     │
│   TCP Server         │   JSON over sockets      │   Flask (port 5000)  │
│   (port 9100)        │                          │                      │
└──────────────────────┘                          └──────────┬───────────┘
                                                             │
                                                  ┌──────────▼───────────┐
                                                  │  Modelless Predictor │
                                                  │  (Linear Regression) │
                                                  └──────────┬───────────┘
                                                             │
                                                  ┌──────────▼───────────┐
                                                  │   LP Optimizer       │
                                                  │  (scipy.linprog)     │
                                                  └──────────────────────┘
```

## Key Results

- **Explicit Model**: Noise σ ≈ 0.50 for both yield and purity (as expected), R² > 0.99
- **Modelless Predictor**: Learned coefficients match ground truth within 1%, R² > 0.999 vs noise-free truth
- **Optimizer**: Learned model finds the same optimal operating point (yield difference < 0.2%)
- **Conclusion**: A data-driven learner can fully replace the hand-crafted explicit model

## Quick Start

1. Start the explicit-model TCP server (port 9100):
   ```bash
   cd ../explicit-model
   python run_server.py
   ```

2. Start the Flask dashboard:
   ```bash
   cd predictor-dashboard
   pip install -r requirements.txt
   python run.py
   ```

3. Open http://127.0.0.1:5000 in your browser.

## Tests

```bash
python -m pytest tests/ -v
```

**20 tests** covering analysis logic and all routes:

| Suite | Tests | Covers |
|-------|-------|--------|
| `test_analysis.py` | 10 | Ground truth, constants, coefficient shapes, base64 encoding |
| `test_routes.py` | 10 | Home page, server-down error pages, API endpoint responses |

## Tech Stack

| Area | Tool |
|------|------|
| Web framework | Flask 3.0+ |
| ML / Learning | scikit-learn (LinearRegression) |
| Optimization | scipy (linprog) |
| Visualization | matplotlib (base64-embedded PNGs) |
| Networking | TCP/IP sockets, JSON protocol |
| Theme | Custom dark UI (CSS variables) |

## Project Structure

```
predictor-dashboard/
├── README.md
├── requirements.txt
├── run.py                       # Entry point (port 5000, debug mode)
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── analysis.py              # TCP data collection, training, validation logic
│   ├── routes.py                # Page routes + JSON API endpoints
│   ├── static/
│   │   └── style.css            # Dark-themed responsive styles
│   └── templates/
│       ├── base.html            # Nav layout (all pages extend this)
│       ├── home.html            # Overview + architecture + feature cards
│       ├── explicit.html        # Explicit model validation dashboard
│       ├── predictor.html       # Modelless predictor dashboard
│       ├── comparison.html      # Side-by-side comparison
│       └── error.html           # Server-unavailable fallback
└── tests/
    ├── __init__.py
    ├── test_analysis.py         # Unit tests for analysis module
    └── test_routes.py           # Route tests with mocked analysis
```

## Run Manifest

Each analysis run (explicit, predictor, comparison) saves a `run_manifest.json`
to the `output/` folder. The manifest records run metadata (timestamp, analysis
type, sample sizes, metrics, checks passed, duration). Manifests are written
automatically when dashboards are loaded or API endpoints are called. See
`docs/SHARED_OUTPUT_CONTRACT.md` in the workspace root for the full schema.

## Related Repositories

| Repository | Description |
|------------|-------------|
| [explicit-model](https://github.com/SarveshPrabhu90/explicit-model) | Ground-truth pharmaceutical process model + TCP server |
| [modelless-predictor](https://github.com/SarveshPrabhu90/modelless-predictor) | Data-driven predictor, data collector, LP optimizer |
| [model-free-predictor-poc](https://github.com/SarveshPrabhu90/model-free-predictor-poc) | Original monorepo (proof of concept) |
