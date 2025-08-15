"""Reporting utilities: load histories, plot equity curve, drawdown, P&L hist, costs, and assemble PDF report."""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_history_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def plot_equity_curve(history_df: pd.DataFrame, title: str = 'Equity Curve', outpath: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_df.get('t', history_df.index), history_df['total_value'])
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Total Value ($)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    if outpath:
        fig.savefig(outpath, bbox_inches='tight'); plt.close(fig)
    return fig

def plot_drawdown(history_df: pd.DataFrame, title: str = 'Drawdown', outpath: str = None) -> plt.Figure:
    vals = history_df['total_value'].values
    peak = np.maximum.accumulate(vals)
    drawdowns = (peak - vals) / (peak + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(history_df.get('t', history_df.index), 0, drawdowns, color='r', alpha=0.3)
    ax.plot(history_df.get('t', history_df.index), drawdowns, color='r')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Drawdown')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    if outpath:
        fig.savefig(outpath, bbox_inches='tight'); plt.close(fig)
    return fig

def plot_pnl_hist(history_df: pd.DataFrame, title: str = 'Step P&L Distribution', outpath: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(history_df['pnl'].dropna(), bins=50, density=True)
    ax.set_xlabel('P&L per Step')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    if outpath:
        fig.savefig(outpath, bbox_inches='tight'); plt.close(fig)
    return fig

def generate_pdf_report(history_paths: List[str], out_pdf: str, meta: Dict[str,Any]=None):
    with PdfPages(out_pdf) as pp:
        # First page: Summary
        summary_data = []
        for path in history_paths:
            df = pd.read_csv(path)
            if not df.empty:
                name = os.path.basename(path).replace('_history.csv','')
                ret = (df['total_value'].iloc[-1] / df['total_value'].iloc[0] - 1.0)
                summary_data.append({'Strategy': name, 'Return': f"{ret:.2%}", 'Steps': len(df)})

        fig, ax = plt.subplots(figsize=(8, len(summary_data) * 0.5 + 3))
        ax.axis('off')
        ax.set_title('Backtest Report Summary', fontsize=16)
        if meta:
            meta_text = '\n'.join([f'{k}: {v}' for k, v in meta.items()])
            ax.text(0.01, 0.95, meta_text, va='top', transform=ax.transAxes)
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Subsequent pages: Individual strategy reports
        for path in history_paths:
            df = load_history_csv(path)
            if df.empty: continue
            name = os.path.basename(path).replace('_history.csv', '')
            
            fig = plot_equity_curve(df, title=f'{name}: Equity Curve')
            pp.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            fig = plot_drawdown(df, title=f'{name}: Drawdown')
            pp.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            fig = plot_pnl_hist(df, title=f'{name}: P&L Distribution')
            pp.savefig(fig, bbox_inches='tight')
            plt.close(fig)