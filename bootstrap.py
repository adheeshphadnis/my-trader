#!/usr/bin/env python3
"""
bootstrap.py
-------------
Step 1: Project scaffolding for a quantitative strategy backtesting + simulation lab.

Usage:
  python bootstrap.py --project-name quant-sim-lab
  python bootstrap.py --project-name quant-sim-lab --force   # overwrite existing files

What this does:
- Creates a clean repo structure with `src/` layout, tests, data buckets, docs, and scripts.
- Writes README.md and docs/concepts.md with the financial concepts we’ll use.
- Adds pyproject.toml and requirements.txt for future steps.
"""

import argparse
import os
from pathlib import Path
from textwrap import dedent

PYTHON_GITIGNORE = dedent("""
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.venv/
venv/

# Distribution / packaging
build/
dist/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# Data caches
data/interim/
data/processed/
reports/
""").strip() + "\n"

README_TEMPLATE = """# {name}

Backtesting + Monte Carlo simulation lab for comparing trading/investment strategies over a 3-year horizon (and beyond).

## What’s inside

- `src/` — Python package (strategies, backtests, risk, simulation, utils)
- `data/` — `{name}` keeps **raw**, **interim**, **processed** buckets (you commit raw only if safe)
- `tests/` — unit tests (we’ll add real tests as we build)
- `docs/` — project docs; start with **financial concepts** you’ll use
- `scripts/` — handy CLIs we’ll add as we go (e.g., fetch data, run backtests)
- `reports/` — figures & backtest outputs
- `notebooks/` — optional exploratory notebooks

## Next steps (we’ll do these together)
1. Create a virtual env and install dependencies.
2. Implement data loader (SPY via yfinance) and a basic sanity check.
3. Implement two strategies:
   - Buy & Hold (SPY)
   - 200-day MA trend-following
4. Backtest both.
5. Monte Carlo simulations to produce return distributions.
6. Compare metrics (CAGR, volatility, Sharpe, Sortino, max drawdown, percentiles).

See `docs/concepts.md` for the finance concepts we’ll rely on.
"""

CONCEPTS_MD = dedent("""
# Core Financial Concepts (What We'll Measure & Why)

This project evaluates strategies using **historical backtests** and **Monte Carlo simulations** to understand the *distribution* of outcomes (not just a single point estimate). Below are the key concepts, with simple math you can sanity-check.

---

## Returns

**Simple return** (daily or monthly):
- r_t = P_t / P_{t-1} - 1

**Log return** (additive over time):
- g_t = ln(P_t / P_{t-1})

We’ll use simple returns for interpretability; log returns are convenient for math but not required.

**Cumulative return** over T periods (simple returns):
- (1 + r_1) * (1 + r_2) * ... * (1 + r_T) - 1

**CAGR** (Compounded Annual Growth Rate) for N years:
- CAGR = (Ending / Beginning)^{1/N} - 1

---

## Volatility (Risk) & Drawdowns

**Volatility (annualized)** from daily returns r_t:
- σ_annual ≈ std(r_daily) * √252
(from monthly returns: √12)

**Max Drawdown (MDD)**:
- Largest peak-to-trough drop of cumulative equity curve:
  MDD = max over time of (peak_to_date - equity) / peak_to_date

This captures the “worst pain” you’d have felt holding the strategy.

---

## Risk-Adjusted Returns

**Sharpe ratio** (excess return per unit of volatility):
- Sharpe = (E[R] - R_f) / σ
  - R_f = risk-free rate (often set ≈ 0 for short horizons; we’ll make this configurable)
  - Higher is better; sensitive to non-normality and downside asymmetry.

**Sortino ratio** (penalizes downside only):
- Sortino = (E[R] - R_f) / σ_downside
  - σ_downside = std of negative returns only
  - More appropriate when upside volatility isn’t “bad.”

---

## Monte Carlo Simulation

We estimate the *distribution* of 3-year outcomes by simulating many future paths:

1) **Bootstrap** historical returns (randomly sample with replacement).
2) Optionally, fit a simple distribution or use block bootstrap to preserve some autocorrelation.
3) For each path, compound returns over the horizon to get an ending value.
4) Aggregate thousands of paths → summarize **median**, **mean**, **5th/95th percentiles**, **worst/best**, and **probability of loss**.

Why this matters: point forecasts hide tail risks. Distributions help you see *how often* you might get a bad outcome.

---

## 200-Day Moving Average (Trend) Strategy

**Rule**: Invest in SPY when price > 200-day simple moving average; otherwise stay in cash (≈ risk-free).
- Intuition: avoid major downtrends; accept whipsaws.
- Implementation details:
  - Compute 200-day SMA on **prior data** (no lookahead).
  - Use next-day close or same-day close after signal generation *consistently*.
  - Account for transaction costs (we’ll parameterize; default can be 0 at first).
  - Cash return can be 0% initially; we can add T-bill yield later.

**Trade-offs**:
- Pros: historically reduces severe drawdowns.
- Cons: can underperform during sharp V-shaped recoveries and sideways markets.

---

## What “Good” Looks Like

We’ll compare strategies on:
- **CAGR**
- **Volatility**
- **Sharpe / Sortino**
- **Max Drawdown**
- **Distribution percentiles** from Monte Carlo

We’ll choose based on your risk tolerance (7/10), focusing on drawdown control vs expected return.

> **Note**: Past performance is not a guarantee of future results. Our simulations are tools to reason about risk, not promises.
""").strip() + "\n"

REQUIREMENTS = dedent("""
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
yfinance>=0.2.40
scipy>=1.11
""").strip() + "\n"

def make_pyproject(name: str) -> str:
    return f"""[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
description = "Backtesting and Monte Carlo simulation framework for strategy comparison"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "yfinance>=0.2.40",
    "scipy>=1.11",
]

[tool.setuptools]
package-dir = {{ "": "src" }}

[tool.setuptools.packages.find]
where = ["src"]
"""



TEST_SMOKE = dedent("""
def test_imports():
    # Basic smoke test to ensure package imports work once installed (-e .)
    # We will populate these modules in later steps.
    assert True
""").strip() + "\n"

INIT_PY = "__all__ = []\n"

SIMPLE_SCRIPT = dedent("""
if __name__ == "__main__":
    print("Placeholder for future CLI entry points.")
""").strip() + "\n"

def safe_write(path: Path, content: str, force: bool):
    if path.exists() and not force:
        print(f"[skip] {path} (exists)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[write] {path}")

def main():
    parser = argparse.ArgumentParser(description="Scaffold a quant research project.")
    parser.add_argument("--project-name", required=True, help="Package/repo name (e.g., quant-sim-lab)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    root = Path.cwd()
    name = args.project_name

    # Top-level files
    safe_write(root / ".gitignore", PYTHON_GITIGNORE, args.force)
    safe_write(root / "README.md", README_TEMPLATE.format(name=name), args.force)
    safe_write(root / "requirements.txt", REQUIREMENTS, args.force)
    safe_write(root / "pyproject.toml", make_pyproject(name), args.force)

    # Docs
    safe_write(root / "docs" / "concepts.md", CONCEPTS_MD, args.force)

    # Data buckets
    for sub in ["raw", "interim", "processed"]:
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)
        safe_write(root / "data" / sub / ".gitkeep", "", args.force)

    # Reports
    for sub in ["figures", "backtests"]:
        (root / "reports" / sub).mkdir(parents=True, exist_ok=True)
        safe_write(root / "reports" / sub / ".gitkeep", "", args.force)

    # Notebooks
    (root / "notebooks").mkdir(parents=True, exist_ok=True)
    safe_write(root / "notebooks" / ".gitkeep", "", args.force)

    # Scripts
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    safe_write(root / "scripts" / "run_placeholder.py", SIMPLE_SCRIPT, args.force)

    # Tests
    (root / "tests").mkdir(parents=True, exist_ok=True)
    safe_write(root / "tests" / "test_smoke.py", TEST_SMOKE, args.force)

    # src package layout with future modules
    base = root / "src" / name.replace("-", "_")
    for subpkg in ["config", "data", "strategies", "backtest", "risk", "sim", "utils"]:
        pkg_path = base / subpkg
        pkg_path.mkdir(parents=True, exist_ok=True)
        safe_write(pkg_path / "__init__.py", INIT_PY, args.force)

    # Top-level __init__.py
    safe_write(base / "__init__.py", INIT_PY, args.force)

    print("\n✅ Scaffolding complete.\n")
    print("Project tree:")
    for dirpath, dirnames, filenames in os.walk(root):
        # only print subtree of current root; limit depth for readability
        rel = Path(dirpath).relative_to(root)
        depth = 0 if rel == Path(".") else len(rel.parts)
        indent = "  " * depth
        if depth <= 3:  # keep it readable
            print(f"{indent}{rel if rel != Path('.') else '.'}/")
            for f in sorted(filenames):
                print(f"{indent}  {f}")

if __name__ == "__main__":
    main()
