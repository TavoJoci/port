# bonifaz_dashboard_v_2_fixed.py
# Streamlit dashboard estilo Bonifaz (tendencia + ciclos)
# Requisitos: pip install streamlit yfinance pandas numpy plotly

import warnings
warnings.filterwarnings("ignore")

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# ------------------------- Utilidades básicas -------------------------

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def regime_filter(df: pd.DataFrame) -> pd.Series:
    # Tendencia válida cuando SMA20 > SMA50
    return (df["SMA20"] > df["SMA50"]).astype(int)

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA20"] = sma(out["Close"], 20)
    out["SMA50"] = sma(out["Close"], 50)
    out["ATR14"] = atr(out, 14)
    out["Regime"] = regime_filter(out)
    return out

def fetch(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    data = data.rename(columns={
        "Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"
    })
    data.dropna(how="any", inplace=True)
    return data

# ------------------------- Señales (pullback + breakout) -------------------------

@dataclass
class EntryParams:
    res_lookback: int = 20          # ventana para resistencia local
    confirm_close_above: float = 0.001  # 0.1% por encima
    vol_confirm_ma: int = 20
    atr_mult_stop: float = 1.5
    tp_atr_mult: float = 1.0

def recent_resistance(high: pd.Series, lookback: int) -> float:
    return high.rolling(lookback).max().shift(1)

def recent_support(low: pd.Series, lookback: int) -> float:
    return low.rolling(lookback).min().shift(1)

def entry_zones(df: pd.DataFrame, p: EntryParams) -> pd.DataFrame:
    out = df.copy()
    out["Res"] = recent_resistance(out["High"], p.res_lookback)
    out["Sup"] = recent_support(out["Low"], p.res_lookback)
    # Breakout válido si Cierre > Res*(1+confirm) y volumen > media
    vol_ma = out["Volume"].rolling(p.vol_confirm_ma).mean()
    out["Breakout_OK"] = (out["Close"] > out["Res"] * (1 + p.confirm_close_above)) & (out["Volume"] > vol_ma)
    # Pullback válido si Close retrocede hacia SMA20 pero SMA20>SMA50
    out["Pullback_OK"] = (out["Close"] >= out["SMA20"]*0.995) & (out["Close"] <= out["SMA20"]*1.01) & (out["SMA20"] > out["SMA50"])
    return out

# ------------------------- Backtest simple y probabilidad calibrada -------------------------

def label_future(df: pd.DataFrame, tp_atr: float, sl_atr: float, horizon: int = 10) -> pd.Series:
    """Etiqueta 1 si en las próximas N velas toca TP antes que SL; 0 si SL antes que TP; NaN si no ocurre.
       Usa solo info a partir de t+1 (evita look-ahead)."""
    close = df["Close"]
    atr14 = df["ATR14"]
    entry = close  # entrada al cierre de la vela de señal
    tp = entry + tp_atr * atr14
    sl = entry - sl_atr * atr14

    # Mirar hacia adelante a partir de t+1
    future_high = df["High"].shift(-1).rolling(horizon).max()
    future_low  = df["Low"].shift(-1).rolling(horizon).min()

    tp_hit = (future_high >= tp)
    sl_hit = (future_low  <= sl)

    y = np.where(tp_hit & ~sl_hit, 1,
        np.where(sl_hit & ~tp_hit, 0, np.nan))
    return pd.Series(y, index=df.index)

def rolling_success_prob(df: pd.DataFrame, mask_signal: pd.Series, horizon: int = 10, tp_mult: float = 1.0, sl_mult: float = 1.5) -> pd.Series:
    y = label_future(df, tp_mult, sl_mult, horizon)
    # Tasa de acierto sobre ventana móvil de señales pasadas (sin mezclar futuro)
    probs = []
    hits = 0; total = 0
    window = 100  # últimas 100 señales
    indices = df.index
    for i in range(len(df)):
        idx = indices[i]
        # agregamos el resultado de señales de pasos previos
        if i>0 and mask_signal.iloc[i-1] and not math.isnan(y.iloc[i-1]):
            total += 1; hits += int(y.iloc[i-1] == 1)
        if total > window:
            # remover muy antiguo: aproximación simple (no estricta por velocidad)
            total = window
        probs.append(hits/total if total>0 else np.nan)
    return pd.Series(probs, index=df.index)

# ------------------------- Vista y lógica principal -------------------------

def analyze(ticker: str, params: EntryParams) -> Dict:
    df = fetch(ticker, period="1y", interval="1d")
    df = calc_indicators(df)
    df = entry_zones(df, params)

    # Señales válidas con régimen
    mask_break = df["Breakout_OK"] & (df["Regime"]==1)
    mask_pull  = df["Pullback_OK"] & (df["Regime"]==1)
    mask_signal = mask_break | mask_pull

    # Probabilidad empírica rodante (calibrada por frecuencia real pasada)
    prob = rolling_success_prob(df, mask_signal, horizon=10, tp_mult=params.tp_atr_mult, sl_mult=params.atr_mult_stop)
    df["Prob_Success"] = prob

    # Sugerencia de entrada (última fila)
    last = df.iloc[-1]
    sugerida = None
    entrada_tipo = None
    if last["Breakout_OK"]:
        sugerida = float(last["Close"])
        entrada_tipo = "breakout"
    elif last["Pullback_OK"]:
        sugerida = float(last["Close"])
        entrada_tipo = "pullback"

    # Stop/TP
    stop = tp1 = tp2 = np.nan
    if not math.isnan(last["ATR14"]):
        stop = round(float(last["Close"] - params.atr_mult_stop * last["ATR14"]), 4)
        tp1  = round(float(last["Close"] + params.tp_atr_mult * last["ATR14"]), 4)
        tp2  = round(float(last["Close"] + 2 * params.tp_atr_mult * last["ATR14"]), 4)

    out = {
        "df": df,
        "entrada_sugerida": sugerida,
        "entrada_tipo": entrada_tipo,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "prob_exito": float(last["Prob_Success"]) if not math.isnan(last["Prob_Success"]) else np.nan,
        "resistencia": float(last["Res"]) if not math.isnan(last["Res"]) else np.nan,
        "soporte": float(last["Sup"]) if not math.isnan(last["Sup"]) else np.nan
    }
    return out

# ------------------------- Streamlit UI -------------------------

def main():
    st.set_page_config(page_title="Bonifaz Dashboard (fix)", layout="wide")
    st.title("Bonifaz – Tendencia + Ciclos (versión FIX)")
    st.caption("Evita *look-ahead*, confirma volumen y usa régimen SMA20>50.")

    ticker = st.text_input("Ticker", "PPL").upper().strip()
    params = EntryParams(
        res_lookback=st.slider("Ventana resistencia/soporte (días)", 10, 60, 20, 1),
        confirm_close_above=st.slider("Confirmación sobre resistencia (%)", 0.0, 0.5, 0.1, 0.05)/100.0,
        vol_confirm_ma=st.slider("Media de volumen (barras)", 10, 40, 20, 1),
        atr_mult_stop=st.slider("Stop ATR×", 0.5, 3.0, 1.5, 0.1),
        tp_atr_mult=st.slider("TP1 ATR×", 0.5, 2.5, 1.0, 0.1)
    )

    results = analyze(ticker, params)
    df = results["df"]

    col1, col2 = st.columns([2,1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Precio"
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["Res"], name="Resistencia"))
        fig.add_trace(go.Scatter(x=df.index, y=df["Sup"], name="Soporte"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Señal actual")
        st.write(f"**Entrada sugerida**: {results['entrada_sugerida']} ({results['entrada_tipo']})" if results['entrada_sugerida'] else "Sin señal vigente")
        st.write(f"**Stop**: {results['stop']}  •  **TP1**: {results['tp1']}  •  **TP2**: {results['tp2']}")
        st.write(f"**Prob éxito empírica**: {round(results['prob_exito']*100,1)}%") if not math.isnan(results['prob_exito']) else st.write("Prob. éxito: s/d")
        st.write(f"**Resistencia**: {results['resistencia']}  •  **Soporte**: {results['soporte']}")

    with st.expander("Datos (últimas 20 velas)"):
        st.dataframe(df.tail(20))

if __name__ == "__main__":
    main()
