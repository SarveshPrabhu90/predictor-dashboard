# Model-Free Predictor — Reporting & Analysis Dashboard

A Flask web application for visualizing, analyzing, and documenting the
model-free predictor proof-of-value. Connects to the
[explicit-model](https://github.com/SarveshPrabhu90/explicit-model) TCP server,
trains the [modelless-predictor](https://github.com/SarveshPrabhu90/modelless-predictor),
and presents interactive dashboards comparing the two approaches.

## Features

- **Home / Overview** — Project summary and architecture diagram
- **Explicit Model** — Validation dashboard for the ground-truth model
- **Modelless Predictor** — Training, prediction accuracy, and coefficient recovery
- **Comparison** — Side-by-side analysis of both models with optimization results
- **API** — JSON endpoints for programmatic access to results

## Architecture

```
┌──────────────────────┐         TCP/IP          ┌──────────────────────┐
│   Explicit Model     │ ◄──────────────────────► │   Flask App          │
│   TCP Server         │   JSON over sockets      │   (this project)     │
│   (port 9100)        │                          │   (port 5000)        │
└──────────────────────┘                          └──────────────────────┘
```

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

## Project Structure

```
predictor-dashboard/
├── README.md
├── requirements.txt
├── run.py                       # Entry point
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── routes.py                # Page and API routes
│   ├── analysis.py              # Data collection & analysis logic
│   ├── templates/
│   │   ├── base.html            # Base layout with nav
│   │   ├── home.html            # Overview page
│   │   ├── explicit.html        # Explicit model dashboard
│   │   ├── predictor.html       # Modelless predictor dashboard
│   │   └── comparison.html      # Side-by-side comparison
│   └── static/
│       └── style.css            # Custom styles
└── tests/
    ├── __init__.py
    ├── test_analysis.py
    └── test_routes.py
```

## Next Steps

- See [GitHub Issues](../../issues) for planned work.
