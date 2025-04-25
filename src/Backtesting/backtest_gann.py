#!/usr/bin/env python3
import os
# Force matplotlib to use a non-interactive backend for Streamlit
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("agg")

import streamlit as st
from datetime import datetime
import logging
import backtrader as bt
import pandas as pd
import numpy as np
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Data fetcher
from src.Data_Retrieval.data_fetcher import DataFetcher
# Agents for CrewAI
from src.UI.gann_main import GannAnalysisAgents

import crewai
from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# =============================================================================
# Gann Hi-Lo Activator calculation (self-contained)
# =============================================================================
def calculate_gann_hi_lo_activator(df: pd.DataFrame, smoothing_period: int = 0) -> pd.DataFrame:
    """
    Calculates the Gann Hi-Lo Activator:
      - If Close > previous Activator: activator = min(Low, prev_activator)
      - Else:                      activator = max(High, prev_activator)
    Then optionally EMA-smooths over `smoothing_period`.
    Adds 'Gann Hi Lo' and 'Gann Hi Lo Smoothed' columns.
    """
    df = df.copy()
    # ensure capitalized columns
    df.rename(columns={c: c.capitalize() for c in df.columns}, inplace=True)
    activator = [float(df['Low'].iloc[0])]
    for i in range(1, len(df)):
        close, low, high = df['Close'].iat[i], df['Low'].iat[i], df['High'].iat[i]
        prev = activator[i-1]
        if close > prev:
            activator.append(min(low, prev))
        else:
            activator.append(max(high, prev))

    df['Gann Hi Lo'] = activator
    if smoothing_period > 1:
        df['Gann Hi Lo Smoothed'] = pd.Series(activator, index=df.index)\
                                      .ewm(span=smoothing_period, adjust=False).mean()
    else:
        df['Gann Hi Lo Smoothed'] = df['Gann Hi Lo']
    return df

# ----------------------------------------
# Backtrader Indicator wrapping Gann logic
# ----------------------------------------
class GannHiLoActivatorBT(bt.Indicator):
    lines = ('activator_raw', 'activator_smoothed')
    params = (('smoothing_period', 0),)

    def __init__(self):
        self.addminperiod(1)

    def once(self, start, end):
        size = self.data.buflen()
        df = pd.DataFrame({
            'high':   [self.data.high[i]   for i in range(size)],
            'low':    [self.data.low[i]    for i in range(size)],
            'close':  [self.data.close[i]  for i in range(size)],
            'volume': [self.data.volume[i] for i in range(size)],
        })
        df['date'] = pd.date_range(end=datetime.today(), periods=size, freq='D')

        res = calculate_gann_hi_lo_activator(df, smoothing_period=self.p.smoothing_period)

        for i in range(size):
            self.lines.activator_raw[i]      = res['Gann Hi Lo'].iat[i]
            self.lines.activator_smoothed[i] = res['Gann Hi Lo Smoothed'].iat[i]

# ----------------------------------------
# Strategy: price crossing the raw Activator
# ----------------------------------------
class GannStrategy(bt.Strategy):
    params = (
        ('smoothing_period', 0),
        ('allocation',        1.0),
    )

    def __init__(self):
        self.trade_log = []
        self.gann = GannHiLoActivatorBT(
            self.data, 
            smoothing_period=self.p.smoothing_period
        )

    def next(self):
        dt    = self.datas[0].datetime.date(0)
        close = self.data.close[0]
        act   = self.gann.activator_raw[0]

        if not self.position and close > act:
            size = int((self.broker.getcash() * self.p.allocation) // close)
            self.buy(size=size)
            msg = f"{dt}: BUY  {size} @ {close:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)

        elif self.position and close < act:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {close:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)

# ----------------------------------------
# Backtest runner
# ----------------------------------------
def run_backtest(strategy_cls, feed, cash=10_000, commission=0.001, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls, **kwargs)
    cerebro.adddata(feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    strat = cerebro.run()[0]
    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    summary = {
        "Sharpe Ratio":         strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
        "Total Return (%)":     r.get('rtot', 0) * 100,
        "Avg Daily Return (%)": r.get('ravg', 0) * 100,
        "Max Drawdown (%)":     d.get('drawdown', 0) * 100,
    }
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Shared LLM (unused directly)
# ----------------------------------------
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Streamlit + CrewAI UI
# ----------------------------------------
def main():
    st.title("Gann Hi-Lo Activator Backtest With CrewAI Signals")

    st.sidebar.header("Parameters")
    ticker           = st.sidebar.text_input("Ticker", "SPY")
    sd               = st.sidebar.date_input("Start", datetime(2020, 1, 1))
    ed               = st.sidebar.date_input("End",   datetime.today().date())
    cash             = st.sidebar.number_input("Cash",       10000)
    comm             = st.sidebar.number_input("Commission",  0.001, step=0.0001)
    smoothing_period = st.sidebar.number_input(
        "Gann Smoothing Period",
        min_value=0,
        value=0,
        step=1
    )

    if st.sidebar.button("Run Backtest"):
        # 1) fetch
        df = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        df.columns = [c.lower() for c in df.columns]  # ensure 'close' exists

        # 2) calc Gann
        df_gann = calculate_gann_hi_lo_activator(df, smoothing_period=smoothing_period)

        # 3) CrewAI
        agents  = GannAnalysisAgents()
        advisor = agents.gann_investment_advisor()
        price   = df['close'].iloc[-1]
        task    = agents.gann_analysis(advisor, df_gann, price)
        crew    = Crew(agents=[advisor], tasks=[task], verbose=True, process=Process.sequential)
        _       = crew.kickoff()

        # 4) backtest
        feed = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)
        perf, trades, fig = run_backtest(
            GannStrategy, feed,
            cash=cash, commission=comm,
            smoothing_period=smoothing_period
        )

        st.subheader("Performance Summary")
        st.write(perf)

        st.subheader("Trade Log")
        for t in trades:
            st.write(t)

        st.subheader("Equity Curve")
        st.pyplot(fig)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
