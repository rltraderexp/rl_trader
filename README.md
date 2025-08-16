# RL Trading Project

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive, modular framework for developing and backtesting algorithmic trading strategies using Reinforcement Learning. Built with PyTorch and Gymnasium, this project provides the necessary components to research, train, and evaluate sophisticated RL agents in realistic, simulated financial environments.

The framework is designed for flexibility, allowing users to easily swap out agents, environments, feature sets, and execution models to explore different trading ideas.

## Key Features

-   **Multiple RL Agents**: Includes implementations of:
    -   **Dueling DQN**: A powerful value-based agent for discrete or discretized action spaces.
    -   **Proximal Policy Optimization (PPO)**: A robust policy-gradient agent with support for continuous, multi-dimensional actions and advanced policy networks (MLP, Conv1D, Transformer).

-   **Diverse Trading Environments**:
    -   `SimpleEnv`: A fast, single-asset environment for quick experiments.
    -   `PortfolioEnv`: A multi-asset environment simulating portfolio management with leverage, margin calls, and risk-adjusted rewards.
    -   `OptionsEnv`: A specialized environment for trading options and their underlying assets, featuring delta hedging and Black-Scholes pricing.

-   **Realistic Execution Simulation**:
    -   Includes a simple order book model (`OrderbookEnv`) that simulates market impact and slippage, providing more realistic fills than a fixed percentage.

-   **Advanced Options Modeling**:
    -   Built-in Black-Scholes pricers for European options and their Greeks (Delta, Gamma, Vega, Theta).
    -   Tools for building and interpolating Implied Volatility (IV) surfaces from market data, including SVI (Stochastic Volatility Inspired) and spline-based models.

-   **Powerful Backtesting Suite**:
    -   A `Backtester` for running policies against historical data and generating performance metrics (Sharpe ratio, max drawdown, etc.).
    -   A `WalkForwardBacktester` for robust out-of-sample validation.
    -   Utilities for generating PDF reports with equity curves and P&L distributions.

-   **Pluggable Forecasting**:
    -   An adapter for integrating external forecasting models, demonstrated with a stub for the `TiRex` time-series transformer.

## Project Structure

The project is organized into modular components to facilitate research and development.

```
rl_trading_v3/
├── rl_trading_project/ # The main source code for the trading framework.
│   ├── agents/         # RL agent implementations (DQN, PPO).
│   ├── data/           # Data loaders and schema validation.
│   ├── envs/           # Trading environments (single-asset, portfolio, options).
│   ├── execution/      # Market execution simulators (e.g., order book).
│   ├── features/       # Feature engineering utilities (RSI, ATR, etc.).
│   ├── forecasting/    # Adapters for external forecasting models.
│   ├── options/        # Tools for options pricing, IV surfaces, and hedging.
│   ├── preprocessing/  # Data scaling and normalization tools.
│   ├── trainers/       # Backtesting, training loops, and reporting tools.
│   └── utils/          # Experiment logging and other utilities.
│
└── tests/              # Unit tests and runnable demo scripts.
    ├── test_core_logic.py
    └── run_env_demo.py
    ...
```

-   **`rl_trading_project/`**: Contains the core, importable Python package with all the framework's logic.
-   **`tests/`**: Contains scripts for validation and demonstration. This includes formal unit tests (`test_core_logic.py`) to ensure components work as expected, and runnable demo scripts (`run_*.py`) that showcase key functionalities.

## Installation

This project uses `uv` for fast and reliable package management.

#### 1. Clone the Repository

First, clone this project from GitHub to your local machine.

```bash
git clone https://github.com/rltraderexp/rl_trader.git
cd rl_trading_project
```
*(Replace `your-username/rl_trading_project.git` with the actual URL of your repository)*

#### 2. Prerequisites

Next, ensure you have `uv` installed. If you don't, run the appropriate command for your system:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

#### 3. Create and Activate a Virtual Environment

From the project's root directory (`rl_trading_project`), create a virtual environment.

```bash
uv venv
```

Activate the newly created environment.

```bash
# On macOS and Linux
source .venv/bin/activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

#### 4. Install Dependencies

Install all required packages from the `requirements.txt` file.

```bash
uv pip install -r requirements.txt
```

#### 5. Install the Project in Editable Mode

Install the `rl_trading_project` package itself in editable mode. This allows you to make changes to the source code that are immediately available to your scripts.

```bash
uv pip install -e .
```

### Optional Installations

-   **TiRex Forecasting Library**: To use the forecasting adapter with the actual `tirex` model, install it from its repository:
    ```bash
    uv pip install "tirex@git+https://github.com/NX-AI/tirex.git"
    ```

-   **PyTorch with GPU (CUDA) Support**: For accelerated training, install a CUDA-enabled version of PyTorch. First, find the correct command for your system on the [PyTorch website](https://pytorch.org/get-started/locally/). Then, uninstall the CPU version and install the GPU version.
    ```bash
    # Example for CUDA 12.1
    uv pip uninstall torch
    uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
    ```

## Getting Started: Running the Demos

The project includes several demo scripts to showcase its capabilities. You can run them from the project's root directory.

#### 1. Basic Environment Demo

This script, located in the `tests` folder, runs a simple buy/sell strategy in the `SimpleEnv` to verify that the core environment logic is working.

```bash
python -m tests.run_env_demo
```

#### 2. Multi-Asset Portfolio Demo

See the `PortfolioEnv` in action, managing a portfolio of three synthetic assets with a simple long/short policy.

```bash
python -m tests.run_portfolio_demo
```

#### 3. Options Delta Hedging Demo

Run a strategy that sells call options while the environment automatically delta-hedges the position.

```bash
python -m rl_trading_project.options.delta_hedge_demo
```

#### 4. Strategy Comparison and Reporting

This demo backtests two different option-selling strategies ("Hedged" vs. "Unhedged") using the `Backtester`, saves the results to CSV files, and prints a comparison.

```bash
python -m rl_trading_project.trainers.demo_compare_strategies
```
After running, you can inspect the generated `Hedged_history.csv` and `Unhedged_history.csv` files.

#### 5. Train a PPO Agent

Run a complete training loop for the PPO agent on the multi-asset `PortfolioEnv`. This script demonstrates feature engineering, data scaling, vectorized environments, and experiment logging.

```bash
python -m rl_trading_project.trainers.train_ppo
```
Training logs and model checkpoints will be saved to a new directory inside `runs/`.

## Core Components

-   **Agents (`agents/`)**: The "brains" of the trading system. The `PPOAgent` is highly configurable, with different network architectures suitable for various observation types (flat vectors, time-series, etc.).

-   **Environments (`envs/`)**: The simulation engine. They define the state, actions, and rewards. The `PortfolioEnv` is the most advanced, handling multi-asset dynamics, while `OptionsEnv` is tailored for derivatives trading. All environments follow the Gymnasium API.

-   **Trainers (`trainers/`)**: These are high-level harnesses for running experiments.
    -   `backtester.py`: Evaluates a fixed policy.
    -   `walkforward.py`: Performs robust, out-of-sample testing.
    -   `train_ppo.py`: Contains a full training pipeline.
    -   `reporting.py`: Creates plots and PDF reports from backtest results.

## Extending the Project

This framework is designed to be a foundation for your own research. Here are some ways you can extend it:

-   **Integrate Real Data**: Modify `data/csv_loader.py` or add new loaders to work with your own market data sources.
-   **Implement a New Agent**: Create a new agent class that inherits from `agents/base.py`.
-   **Design a Custom Reward Function**: Implement new reward shaping functions in `trainers/reward_shapers.py` to guide the agent toward different objectives (e.g., maximizing Sharpe ratio, minimizing drawdown).
-   **Add New Features**: Add new technical indicators or alternative features to `features/basic_features.py`.
-   **Connect a Real Forecasting Model**: Customize the `_load_model` method in `forecasting/tirex_adapter.py` to load your own trained forecasting model.

## License

This project is licensed under the MIT License.