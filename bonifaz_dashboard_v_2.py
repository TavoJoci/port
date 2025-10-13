
# bonifaz_dashboard_v_2.py — Bonifaz–Claro Dashboard (zonas corregidas)
# Dependencias: pip install streamlit yfinance pandas numpy openpyxl plotly

import io
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception as e:
    st.error("Necesitas instalar yfinance: `pip install yfinance`")
    raise e

# =============================
# CONFIG por defecto
# =============================
DEFAULT_TICKERS = ["ANET", "WDAY", "CCJ"]
DEFAULT_ENTRADAS = {"ANET": np.nan, "WDAY": np.nan, "CCJ": np.nan}

DEFAULTS = dict(
    buffer_pct=1.5,
    atr_mult=1.0,
    days=180,
    tp_mode="STRUCT",  # "R", "ATR", "AUTO", "STRUCT"
    auto_vol_threshold_low=3.5,
    auto_vol_threshold_high=5.0,
    min_tp1_pct=1.5,
    min_tp2_pct=3.0,
    min_tp1_R=0.8,
    min_tp2_R=1.6,
    enable_tp3=True,
    tp3_R=3.0,
    auto_ladder=True,
    lookback=120,
    pivot_span=3,
    resistance_buffer_pct=0.3,
    measured_move_frac=0.6,
    breakout_cap_pct=25.0,  # tope +% sobre precio para breakout
    # Zonas de compra (defaults) — ancho y tolerancia más realistas
    pullback_zone_atr_low=0.10,   # 0.10×ATR por sobre SMA20
    pullback_zone_atr_high=0.30,  # 0.30×ATR por sobre SMA20
    breakout_zone_pct=0.6,        # 0.6% por sobre el gatillo de ruptura
    entry_tolerance_pct=0.25,     # ±0.25% para marcar "en zona"
)

# =============================
# Descarga y técnicos
# =============================

def download_data(tickers: List[str], days: int) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=f"{days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if isinstance(df.columns, pd.MultiIndex):
        tickers = sorted(list(set(df.columns.get_level_values(-1))))
        for t in tickers:
            sub = pd.DataFrame(
                {
                    "Open": df[("Open", t)],
                    "High": df[("High", t)],
                    "Low": df[("Low", t)],
                    "Close": df[("Adj Close", t)] if ("Adj Close", t) in df.columns else df[("Close", t)],
                }
            ).dropna()
            rows.append((t, _ind_row(sub)))
    else:
        rows.append(("TICKER", _ind_row(df.rename(columns=str.title))))

    res = pd.DataFrame(rows, columns=["Ticker", "Metrics"])
    return pd.concat([res[["Ticker"]], pd.json_normalize(res["Metrics"])], axis=1)

def _ind_row(px: pd.DataFrame) -> dict:
    px = px.copy()
    if px is None or px.empty or px['Close'].dropna().empty:
        return {
            "Price": np.nan, "SMA20": np.nan, "SMA50": np.nan, "SMA200": np.nan, "ATR14": np.nan,
            "Slope20": np.nan, "Slope50": np.nan,
            "TAIL_High": [], "TAIL_Low": [], "TAIL_Close": [],
            "TAIL_SMA20": [], "TAIL_SMA50": [], "TAIL_SMA200": [], "TAIL_Index": []
        }

    px["TR"] = np.maximum(
        px["High"] - px["Low"],
        np.maximum(abs(px["High"] - px["Close"].shift(1)), abs(px["Low"] - px["Close"].shift(1))),
    )
    px["ATR14"] = px["TR"].rolling(14).mean()
    px["SMA20"] = px["Close"].rolling(20).mean()
    px["SMA50"] = px["Close"].rolling(50).mean()
    px["SMA200"] = px["Close"].rolling(200).mean()

    def slope(s, window=5):
        y = s.dropna().tail(window).values
        if len(y) < window:
            return np.nan
        x = np.arange(len(y))
        return float(np.polyfit(x, y, 1)[0])

    slope20 = slope(px["SMA20"], 5)
    slope50 = slope(px["SMA50"], 5)

    last = px.dropna(subset=["Close", "SMA20", "SMA50", "ATR14"]).iloc[-1]

    sma20_full = px["Close"].rolling(20).mean()
    sma50_full = px["Close"].rolling(50).mean()
    sma200_full = px["Close"].rolling(200).mean()

    tail = px.tail(DEFAULTS["lookback"]).copy()

    return {
        "Price": round(float(last["Close"]), 4),
        "SMA20": round(float(last["SMA20"]), 4),
        "SMA50": round(float(last["SMA50"]), 4),
        "SMA200": round(float(sma200_full.dropna().iloc[-1]) if not sma200_full.dropna().empty else np.nan, 4),
        "ATR14": round(float(last["ATR14"]), 4),
        "Slope20": round(float(slope20), 6) if not np.isnan(slope20) else np.nan,
        "Slope50": round(float(slope50), 6) if not np.isnan(slope50) else np.nan,
        "TAIL_High": tail["High"].tolist(),
        "TAIL_Low": tail["Low"].tolist(),
        "TAIL_Close": tail["Close"].tolist(),
        "TAIL_SMA20": sma20_full.tail(DEFAULTS["lookback"]).tolist(),
        "TAIL_SMA50": sma50_full.tail(DEFAULTS["lookback"]).tolist(),
        "TAIL_SMA200": sma200_full.tail(DEFAULTS["lookback"]).tolist(),
        "TAIL_Index": tail.index.astype(str).tolist(),
    }

# =============================
# Fase/Stops/TPs
# =============================

def fase_bonifaz(row: pd.Series) -> Tuple[str, str]:
    price, s20, s50, m20 = row["Price"], row["SMA20"], row["SMA50"], row["Slope20"]
    if price > s20 > s50 and (pd.notna(m20) and m20 > 0):
        return "ciclo alcista activo", "SMA20"
    if price > s50 and (abs(s20 - s50) / s50 < 0.01 or (pd.notna(m20) and m20 <= 0)):
        return "base / incipiente", "SMA50"
    if price < s50:
        return "defensivo (bajo SMA50)", "SMA50"
    return "estructural", "SMA50"

def recomendar_stop(row: pd.Series, buffer_pct: float, atr_mult: float, auto_low: float, auto_high: float) -> dict:
    """Stop de contexto (AJUSTADO/EXTENDIDO) sin usar aún la ENTRADA; solo referencia."""
    fase, base_tipo = fase_bonifaz(row)
    base_sma = row["SMA20"] if base_tipo == "SMA20" else row["SMA50"]
    price, atr = row["Price"], row["ATR14"]

    stop_ajustado = base_sma * (1 - buffer_pct / 100.0)
    stop_extendido = stop_ajustado - atr_mult * atr

    vol_rel = atr / price * 100.0 if price > 0 else np.nan
    if vol_rel < auto_low and "ciclo alcista activo" in fase:
        sugerido = "Ajustado"
        recomendacion = "Usar Stop Ajustado (sin ATR)."
    elif vol_rel > auto_high or "defensivo" in fase:
        sugerido = "Extendido"
        recomendacion = "Usar Stop Extendido (con ATR)."
    else:
        if "base / incipiente" in fase:
            sugerido = "Extendido"
            recomendacion = "Preferir Stop Extendido (con ATR)."
        else:
            sugerido = "Ajustado"
            recomendacion = "Preferir Stop Ajustado (sin ATR)."

    return {
        "FaseBonifaz": fase,
        "BaseStop": base_tipo,
        "Volatilidad_%": round(vol_rel, 2) if pd.notna(vol_rel) else np.nan,
        "Stop_Ajustado": round(stop_ajustado, 4) if pd.notna(stop_ajustado) else np.nan,
        "Stop_Extendido": round(stop_extendido, 4) if pd.notna(stop_extendido) else np.nan,
        "Stop_Sugerido": sugerido,
        "Riesgo_%_Ajustado": round((price - stop_ajustado) / price * 100.0, 2) if pd.notna(stop_ajustado) and price>0 else np.nan,
        "Riesgo_%_Extendido": round((price - stop_extendido) / price * 100.0, 2) if pd.notna(stop_extendido) and price>0 else np.nan,
        "RecomendacionStop": recomendacion,
    }

# -------- Estructura (pivotes) --------

def pivot_highs(highs: List[float], span=3) -> List[float]:
    highs = list(highs)
    pivots = []
    n = len(highs)
    for i in range(span, n - span):
        h = highs[i]
        if h > max(highs[i - span : i]) and h > max(highs[i + 1 : i + 1 + span]):
            pivots.append(round(float(h), 4))
    return sorted(set(pivots))

def pivot_lows(lows: List[float], span=3) -> List[float]:
    lows = list(lows)
    pivots = []
    n = len(lows)
    for i in range(span, n - span):
        l = lows[i]
        if l < min(lows[i - span : i]) and l < min(lows[i + 1 : i + 1 + span]):
            pivots.append(round(float(l), 4))
    return sorted(set(pivots))

# -------- Take Profits --------

def round_numbers_near(price: float, steps=None) -> List[float]:
    steps = steps or [50, 25, 10, 5, 2, 1, 0.5]
    levels = set()
    for s in steps:
        k = round(price / s)
        for delta in (-1, 0, 1):
            levels.add(round((k + delta) * s, 4))
    return sorted(levels)

def measured_move(entry: float, highs: List[float], lows: List[float], frac=0.6) -> float:
    if not highs or not lows:
        return np.nan
    rng = (max(highs) - min(lows)) * float(frac)
    return entry + rng

def snap_to_structure(entry: float, base_tp: float, highs: List[float], lows: List[float], buffer_pct=0.3, steps=None):
    buf = buffer_pct / 100.0
    piv = pivot_highs(highs, DEFAULTS["pivot_span"]) if highs else []
    rnums = round_numbers_near(base_tp, steps or [50, 25, 10, 5, 2, 1, 0.5])

    res_above_entry = [lvl for lvl in (piv + rnums) if lvl > entry]
    candidates = [lvl for lvl in res_above_entry if lvl >= base_tp] or res_above_entry
    if candidates:
        chosen = min(candidates)
        snapped = chosen * (1 - buf)
        return round(float(snapped), 4), "resistencia", chosen
    return round(float(base_tp), 4), "base", np.nan

def enforce_tp_floors(entrada: float, R: float, base_tp1: float, base_tp2: float,
                      min_tp1_pct: float, min_tp2_pct: float, min_tp1_R: float, min_tp2_R: float):
    min_tp1_pct_val = entrada * (1 + min_tp1_pct / 100.0)
    min_tp2_pct_val = entrada * (1 + min_tp2_pct / 100.0)
    min_tp1_R_val = entrada + min_tp1_R * max(R, 0)
    min_tp2_R_val = entrada + min_tp2_R * max(R, 0)
    tp1 = max(base_tp1, min_tp1_pct_val, min_tp1_R_val)
    tp2 = max(base_tp2, min_tp2_pct_val, min_tp2_R_val)
    return tp1, tp2

def tp_targets(row: pd.Series, entrada: float, stop_sugerido: float, atr: float, mode: str, params: dict) -> dict:
    price = row["Price"]
    vol_rel = row.get("Volatilidad_%", np.nan)
    sugerido = row.get("Stop_Sugerido", "Ajustado")

    mode_upper = mode.upper()
    if mode_upper in ("AUTO", "STRUCT"):
        if sugerido == "Ajustado" and (pd.notna(vol_rel) and vol_rel < params["auto_vol_threshold_low"]):
            base_mode = "R"
        elif sugerido == "Extendido" or (pd.notna(vol_rel) and vol_rel >= params["auto_vol_threshold_low"]):
            base_mode = "ATR"
        else:
            base_mode = "R"
    else:
        base_mode = mode_upper

    tp1 = tp2 = tp3 = np.nan
    nota = ""
    status = "sin entrada"
    sugerencia_accion = ""

    if pd.notna(entrada) and entrada > 0 and pd.notna(stop_sugerido):
        R = max(entrada - stop_sugerido, 0)
        if base_mode == "R":
            base_tp1 = entrada + 1.0 * R
            base_tp2 = entrada + 2.0 * R
            nota = "TP base por R: TP1=E+1R, TP2=E+2R."
            if params["enable_tp3"]:
                tp3 = entrada + params["tp3_R"] * R
        else:
            base_tp1 = entrada + 1.0 * atr
            base_tp2 = entrada + 2.0 * atr
            nota = "TP base por ATR: TP1=E+1×ATR, TP2=E+2×ATR."
            if params["enable_tp3"]:
                tp3 = entrada + params["tp3_R"] * atr

        tp1, tp2 = enforce_tp_floors(
            entrada, R, base_tp1, base_tp2,
            params["min_tp1_pct"], params["min_tp2_pct"], params["min_tp1_R"], params["min_tp2_R"]
        )

        if mode_upper == "STRUCT":
            highs = row["TAIL_High"]
            lows = row["TAIL_Low"]
            mm = measured_move(entrada, highs, lows, params["measured_move_frac"])
            if not np.isnan(mm):
                tp2 = max(tp2, mm * 0.8)
            tp1, _, _ = snap_to_structure(entrada, tp1, highs, lows, params["resistance_buffer_pct"]) 
            tp2, _, _ = snap_to_structure(entrada, tp2, highs, lows, params["resistance_buffer_pct"]) 
            nota += " | STRUCT: TP ajustados a estructura."

        if price >= tp2:
            status = "TP2 alcanzado"
            sugerencia_accion = "Toma ganancias fuertes y trailing por SMA20 si el ciclo sigue."
        elif price >= tp1:
            status = "TP1 alcanzado"
            sugerencia_accion = "Vende ~50% y mueve stop a break-even o trailing SMA20."
        else:
            status = "en curso"
            sugerencia_accion = "Mantén el plan; no muevas el stop antes de tiempo."
    else:
        nota = "Sin precio de entrada: no se calculan TP."

    return {
        "TP_Mode": mode_upper,
        "TP_BaseMode": base_mode,
        "TP1": round(tp1, 4) if pd.notna(tp1) else np.nan,
        "TP2": round(tp2, 4) if pd.notna(tp2) else np.nan,
        "TP3": round(tp3, 4) if pd.notna(tp3) else np.nan,
        "TP_Status": status,
        "TP_Nota": nota,
        "SugerenciaAccion": sugerencia_accion,
    }

# =============================
# Sugeridor de entrada (candidatos y selección)
# =============================

def entry_candidates(row: pd.Series, params: dict) -> dict:
    price, s20, s50, atr = row["Price"], row["SMA20"], row["SMA50"], row["ATR14"]
    highs = row["TAIL_High"]

    res_buf = params.get("resistance_buffer_pct", DEFAULTS["resistance_buffer_pct"])
    pivot_span = int(DEFAULTS["pivot_span"])  # usa slider global
    piv = pivot_highs(highs, pivot_span) if highs else []
    last_res = max(piv) if piv else np.nan

    atr_safe = float(atr) if pd.notna(atr) else 0.0
    breakout_cap = max(
        price * (1 + float(params.get("breakout_cap_pct", DEFAULTS["breakout_cap_pct"])) / 100.0),
        price + 1.5 * atr_safe,
    )

    # Parámetros de zona
    pb_low_m = float(params.get("pullback_zone_atr_low", DEFAULTS["pullback_zone_atr_low"]))
    pb_high_m = float(params.get("pullback_zone_atr_high", DEFAULTS["pullback_zone_atr_high"]))
    brk_zone_pct = float(params.get("breakout_zone_pct", DEFAULTS["breakout_zone_pct"])) / 100.0

    # Pullback (rango) — SOLO si price > SMA20
    pull_low = pull_high = np.nan
    trig_pull = ""
    if pd.notna(s20):
        base = s20
        if price > base:
            pull_low = base + pb_low_m * atr_safe
            pull_high = base + pb_high_m * atr_safe
            pull_high = min(pull_high, price * 0.995)  # < precio actual
            pull_low = min(pull_low, pull_high)
            trig_pull = f"Espera retroceso a SMA20 (+{pb_low_m:.2f}–{pb_high_m:.2f}×ATR)"
        else:
            pull_low = pull_high = np.nan
            trig_pull = "Sin pullback válido (precio ≤ SMA20)"

    # Breakout (rango) — soporta pre y post-ruptura
    brk_low = brk_high = np.nan
    trig_brk = ""
    if pd.notna(last_res):
        trigger = last_res * (1 + res_buf / 100.0)
        within_cap = trigger <= breakout_cap
        if within_cap:
            brk_low_candidate = trigger
            brk_high_candidate = trigger * (1 + brk_zone_pct)
            # usamos la misma zona tanto si está por debajo (pre) como por encima (post)
            brk_low = brk_low_candidate
            brk_high = brk_high_candidate
            trig_brk = f"Compra si rompe {round(last_res, 2)} (+margen) y confirma"

    # --- Anchos mínimos de zona por cordura ---
    def _ensure_min_width(zlow, zhigh, price, atr, min_pct=0.15, min_atr_frac=0.25):
        if pd.isna(zlow) or pd.isna(zhigh):
            return zlow, zhigh
        width = zhigh - zlow
        min_width = max(price * (min_pct/100.0), (atr if pd.notna(atr) else 0.0) * min_atr_frac)
        if width < min_width:
            center = (zlow + zhigh) / 2
            zlow = center - min_width/2
            zhigh = center + min_width/2
        return round(float(zlow),4), round(float(zhigh),4)

    pull_low, pull_high = _ensure_min_width(pull_low, pull_high, price, atr_safe)
    brk_low, brk_high   = _ensure_min_width(brk_low,  brk_high,  price, atr_safe)

    return {
        "Entrada_Pullback": round(float((pull_low + pull_high)/2), 4) if pd.notna(pull_low) and pd.notna(pull_high) else np.nan,
        "Trigger_Pullback": trig_pull,
        "Zona_Pullback_Low": round(float(pull_low), 4) if pd.notna(pull_low) else np.nan,
        "Zona_Pullback_High": round(float(pull_high), 4) if pd.notna(pull_high) else np.nan,
        "Entrada_Breakout": round(float((brk_low + brk_high)/2), 4) if pd.notna(brk_low) and pd.notna(brk_high) else np.nan,
        "Trigger_Breakout": trig_brk,
        "Zona_Breakout_Low": round(float(brk_low), 4) if pd.notna(brk_low) else np.nan,
        "Zona_Breakout_High": round(float(brk_high), 4) if pd.notna(brk_high) else np.nan,
        "DBG_LastRes": round(float(last_res), 4) if pd.notna(last_res) else np.nan,
    }

def suggest_entry(row: pd.Series, params: dict) -> dict:
    """Selección AUTO usando fase + cercanía a resistencia. Candidatos se calculan en entry_candidates()."""
    cands = entry_candidates(row, params)
    price, s20 = row["Price"], row["SMA20"]
    fase = str(row.get("FaseBonifaz", "")).lower()

    prefer_pct = float(params.get("prefer_breakout_near_res_pct", 2.5)) / 100.0
    last_res = cands.get("DBG_LastRes", np.nan)
    prefer_breakout = pd.notna(last_res) and price >= last_res * (1 - prefer_pct)

    # Regla AUTO
    if "alcista" in fase and pd.notna(s20):
        if prefer_breakout and pd.notna(cands["Entrada_Breakout"]):
            return {"Entrada_Tipo": "breakout", "Entrada_Sugerida": cands["Entrada_Breakout"],
                    "Trigger": cands["Trigger_Breakout"], "Comentario": "Cerca de resistencia; se privilegia ruptura."}
        elif pd.notna(cands["Entrada_Pullback"]):
            return {"Entrada_Tipo": "pullback", "Entrada_Sugerida": cands["Entrada_Pullback"],
                    "Trigger": cands["Trigger_Pullback"], "Comentario": "Ciclo alcista: entrada a valor."}
    if "base" in fase and pd.notna(cands["Entrada_Breakout"]):
        return {"Entrada_Tipo": "breakout", "Entrada_Sugerida": cands["Entrada_Breakout"],
                "Trigger": cands["Trigger_Breakout"], "Comentario": "Base/incipiente: confirmación por ruptura."}
    if pd.notna(cands["Entrada_Pullback"]):
        return {"Entrada_Tipo": "pullback", "Entrada_Sugerida": cands["Entrada_Pullback"],
                "Trigger": cands["Trigger_Pullback"], "Comentario": "Transición/valor."}
    return {"Entrada_Tipo": "esperar", "Entrada_Sugerida": np.nan, "Trigger": "Espera condiciones", "Comentario": "Sin señal."}

# =============================
# Evaluación en lote
# =============================

def evaluate_universe(tickers: List[str], entradas: Dict[str, float], days: int,
                      buffer_pct: float, atr_mult: float,
                      auto_low: float, auto_high: float,
                      lookback: int,
                      tp_params: dict,
                      fallback_price_if_nan: bool = False,
                      strategy_override: str = "auto") -> pd.DataFrame:
    raw = download_data(tickers, days)
    base = compute_indicators(raw)

    # Tails con lookback elegido
    base_rt = base.copy()
    for i in range(len(base_rt)):
        t = base_rt.loc[i, "Ticker"]
        if isinstance(raw.columns, pd.MultiIndex):
            sub = pd.DataFrame(
                {
                    "Open": raw[("Open", t)],
                    "High": raw[("High", t)],
                    "Low": raw[("Low", t)],
                    "Close": raw[("Adj Close", t)] if ("Adj Close", t) in raw.columns else raw[("Close", t)],
                }
            ).dropna()
        else:
            sub = raw.rename(columns=str.title).dropna()
        tail = sub.tail(lookback).copy()
        base_rt.at[i, "TAIL_High"] = tail["High"].tolist()
        base_rt.at[i, "TAIL_Low"] = tail["Low"].tolist()
        base_rt.at[i, "TAIL_Close"] = tail["Close"].tolist()
        base_rt.at[i, "TAIL_SMA20"] = sub["Close"].rolling(20).mean().tail(lookback).tolist()
        base_rt.at[i, "TAIL_SMA50"] = sub["Close"].rolling(50).mean().tail(lookback).tolist()
        base_rt.at[i, "TAIL_SMA200"] = sub["Close"].rolling(200).mean().tail(lookback).tolist()
        base_rt.at[i, "TAIL_Index"] = tail.index.astype(str).tolist()

    # Recs de stop de contexto
    recs = base_rt.apply(lambda r: recomendar_stop(r, buffer_pct, atr_mult, auto_low, auto_high), axis=1)
    recs_df = pd.json_normalize(recs)
    out = pd.concat([base_rt, recs_df], axis=1)
    out.insert(1, "Fecha", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Entradas manuales iniciales
    entradas_vals, stop_sug_vals = [], []
    for _, row in out.iterrows():
        t = row["Ticker"]
        e = entradas.get(t, np.nan)
        if fallback_price_if_nan and (pd.isna(e) or e <= 0):
            e = row["Price"]
        entradas_vals.append(e)
        stop_val = row["Stop_Ajustado"] if row["Stop_Sugerido"] == "Ajustado" else row["Stop_Extendido"]
        stop_sug_vals.append(stop_val)
    out["Entrada"] = entradas_vals
    out["Stop_Sugerido_Nivel"] = stop_sug_vals  # temporal; se recalcula luego según ENTRADA/estrategia

    # Candidatos y sugerencia AUTO
    cands = out.apply(lambda r: entry_candidates(r, tp_params), axis=1)
    cands_df = pd.json_normalize(cands)
    out = pd.concat([out, cands_df], axis=1)

    sug = out.apply(lambda r: suggest_entry(r, tp_params), axis=1)
    sug_df = pd.json_normalize(sug)
    out = pd.concat([out, sug_df], axis=1)

    # Override de estrategia (Pullback/Breakout) si el usuario lo pide
    ov = str(strategy_override).strip().lower()
    if ov in ("pullback", "breakout"):
        def apply_override(r: pd.Series) -> Tuple[str, float, str]:
            if ov == "pullback" and pd.notna(r.get("Entrada_Pullback")):
                return "pullback", r.get("Entrada_Pullback"), r.get("Trigger_Pullback")
            if ov == "breakout" and pd.notna(r.get("Entrada_Breakout")):
                return "breakout", r.get("Entrada_Breakout"), r.get("Trigger_Breakout")
            # si no hay candidato disponible, conserva AUTO
            return r.get("Entrada_Tipo"), r.get("Entrada_Sugerida"), r.get("Trigger")
        ov_vals = out.apply(apply_override, axis=1, result_type="expand")
        out["Entrada_Tipo"], out["Entrada_Sugerida"], out["Trigger"] = ov_vals[0], ov_vals[1], ov_vals[2]

    # Fijar zona según estrategia elegida
    def zona_row(r: pd.Series) -> Tuple[float, float]:
        t = str(r.get("Entrada_Tipo", "")).lower()
        if t == "pullback":
            return r.get("Zona_Pullback_Low", np.nan), r.get("Zona_Pullback_High", np.nan)
        if t == "breakout":
            return r.get("Zona_Breakout_Low", np.nan), r.get("Zona_Breakout_High", np.nan)
        return np.nan, np.nan

    zvals = out.apply(zona_row, axis=1, result_type="expand")
    out["Zona_Low"], out["Zona_High"] = zvals[0], zvals[1]

    # Entrada sugerida = punto medio de la zona
    out["Entrada_Sugerida"] = ((out["Zona_Low"] + out["Zona_High"]) / 2).round(4)

    # Fijar Entrada si no hay manual y hay señal
    tipo = out["Entrada_Tipo"].astype(str).str.strip().str.lower()
    hay_senal = tipo.isin(["pullback", "breakout"])
    sin_manual = out["Entrada"].isna() | (out["Entrada"] <= 0)
    out.loc[sin_manual & hay_senal, "Entrada"] = out.loc[sin_manual & hay_senal, "Entrada_Sugerida"]
    out.loc[tipo == "esperar", "Entrada"] = np.nan

    # Flag "En zona" — tolerancia asimétrica por estrategia
    tol = float(tp_params.get("entry_tolerance_pct", DEFAULTS["entry_tolerance_pct"])) / 100.0
    def en_zona(r: pd.Series) -> str:
        zl, zh, p = r.get("Zona_Low", np.nan), r.get("Zona_High", np.nan), r.get("Price", np.nan)
        if pd.notna(zl) and pd.notna(zh) and pd.notna(p):
            t = str(r.get("Entrada_Tipo","")).lower()
            if t == "breakout":
                low = zl * (1 - tol)
                high = zh * (1 + tol*1.2)
            elif t == "pullback":
                low = zl * (1 - tol*1.2)
                high = zh * (1 + tol)
            else:
                low = zl * (1 - tol)
                high = zh * (1 + tol)
            return "EN_ZONA" if (p >= low and p <= high) else "FUERA"
        return "SIN_ZONA"
    out["Entrada_Zona_Estado"] = out.apply(en_zona, axis=1)

    # Stop definitivo coherente con ENTRADA + regimen
    def stop_from_entry_row(r: pd.Series,
                            buffer_pct_ui: float,
                            atr_mult_ui: float,
                            res_buf_pct: float,
                            pivot_span_local: int) -> float:
        entrada = r.get("Entrada", np.nan)
        if pd.isna(entrada) or entrada <= 0:
            return np.nan
        s20, s50, atr = r.get("SMA20"), r.get("SMA50"), r.get("ATR14")
        lows = r.get("TAIL_Low", [])
        tipo_e = str(r.get("Entrada_Tipo", "")).strip().lower()
        regimen = str(r.get("Stop_Sugerido", "Ajustado")).strip().lower()
        atr_val = float(atr) if pd.notna(atr) else 0.0
        min_gap = max(0.012 * entrada, 0.8 * atr_val)
        max_depth = max(0.03 * entrada, 3.0 * atr_val)
        base = np.nan
        if tipo_e == "pullback" and pd.notna(s20):
            stop_base = s20 * (1 - buffer_pct_ui / 100.0)
            base = stop_base if regimen == "ajustado" else stop_base - atr_mult_ui * atr_val
        elif tipo_e == "breakout":
            plows = pivot_lows(lows, pivot_span_local) if isinstance(lows, list) and len(lows)>0 else []
            ref = plows[-1] if plows else (s50 if pd.notna(s50) else np.nan)
            stop_base = (ref * (1 - res_buf_pct / 100.0)) if pd.notna(ref) else (s50 * (1 - buffer_pct_ui / 100.0) if pd.notna(s50) else np.nan)
            base = stop_base if regimen == "ajustado" else stop_base - atr_mult_ui * atr_val
        else:
            return np.nan
        if pd.notna(base):
            if base >= entrada:
                base = entrada - min_gap
            base = max(entrada - max_depth, base)
        return round(float(base), 4) if pd.notna(base) else np.nan

    out["Stop_Sugerido_Nivel"] = out.apply(
        lambda r: stop_from_entry_row(
            r,
            buffer_pct_ui=buffer_pct,
            atr_mult_ui=atr_mult,
            res_buf_pct=tp_params.get("resistance_buffer_pct", DEFAULTS["resistance_buffer_pct"]),
            pivot_span_local=int(DEFAULTS["pivot_span"]) 
        ), axis=1
    )

    # Probabilidades (ajustadas por estrategia/ubicación)
    def prob_exito(row: pd.Series, horizonte: str) -> float:
        # Base por ciclo/tendencia
        score = 50.0
        fase = row.get("FaseBonifaz", "").lower()
        if "alcista" in fase: score += 15
        elif "base" in fase: score += 5
        elif "defensivo" in fase: score -= 15

        # Pendientes
        if pd.notna(row.get("Slope20")) and row.get("Slope20") > 0: score += 5
        if pd.notna(row.get("Slope50")) and row.get("Slope50") > 0: score += 5

        # Volatilidad (ATR relativa)
        vol = row.get("Volatilidad_%")
        if pd.notna(vol):
            if vol < 3: score += 5
            elif vol > 6: score -= 5

        # Ubicación respecto a medias
        price, s20, s50 = row["Price"], row["SMA20"], row["SMA50"]
        if price >= s20 >= s50: score += 5
        if price < s50: score -= 10

        # --- Ajuste por ESTRATEGIA/ENTRADA seleccionada ---
        entrada = row.get("Entrada", np.nan)
        entrada_tipo = str(row.get("Entrada_Tipo", "")).lower()
        stop_lvl = row.get("Stop_Sugerido_Nivel", np.nan)
        last_res = row.get("DBG_LastRes", np.nan) if "DBG_LastRes" in row else np.nan

        if pd.notna(entrada) and entrada > 0 and entrada_tipo in ("pullback", "breakout"):
            if entrada_tipo == "pullback":
                # Queremos que esté por ENCIMA de la entrada pero cerca (pagas valor)
                delta = (price - entrada) / entrada * 100.0
                if 0.2 <= delta <= 2.0:
                    score += 6
                elif -0.5 <= delta < 0.2:
                    score += 2
                elif delta < -0.5:
                    score -= 6
            else:  # breakout
                # Queremos que esté CERCA del techo; 'entrada' ~ gatillo
                gap = (entrada - price) / price * 100.0
                if 0.1 <= gap <= 2.5:
                    score += 6
                elif 2.5 < gap <= 5.0:
                    score += 2
                elif gap <= 0:
                    score += 1
                if pd.notna(last_res) and entrada > price * 1.35:
                    score -= 6

            # Calidad de R (riesgo)
            if pd.notna(stop_lvl) and stop_lvl < entrada:
                R_pct = (entrada - stop_lvl) / entrada * 100.0
                if R_pct <= 4:
                    score += 3
                elif R_pct > 8:
                    score -= 4

        # Horizonte
        if horizonte == "corto":
            if pd.notna(vol) and vol <= 4: score += 2
            rr2 = row.get("RR_TP2", np.nan)
            if pd.notna(rr2) and rr2 >= 2.0: score += 2
        else:  # medio
            if pd.notna(row.get("SMA200")) and price > row.get("SMA200"): score += 2

        return float(np.clip(score, 5, 90))

    # TPs (desde ENTRADA y Stop_Sugerido_Nivel ya coherentes)
    tps = out.apply(
        lambda r: tp_targets(
            r, r["Entrada"], r["Stop_Sugerido_Nivel"], r["ATR14"], tp_params["tp_mode"], tp_params
        ),
        axis=1,
    )
    tps_df = pd.json_normalize(tps)
    out = pd.concat([out, tps_df], axis=1)

    # RR y métricas
    with np.errstate(divide='ignore', invalid='ignore'):
        out["R_%"] = ((out["Entrada"] - out["Stop_Sugerido_Nivel"]) / out["Entrada"]) * 100
        den = (out["Entrada"] - out["Stop_Sugerido_Nivel"]).replace(0, np.nan)
        out["RR_TP1"] = (out["TP1"] - out["Entrada"]) / den
        out["RR_TP2"] = (out["TP2"] - out["Entrada"]) / den

    out["Prob_Exito_Corto_%"] = out.apply(lambda r: round(prob_exito(r, "corto"), 1), axis=1)
    out["Prob_Exito_Medio_%"] = out.apply(lambda r: round(prob_exito(r, "medio"), 1), axis=1)

    cols = [
        "Ticker","Fecha","Entrada_Zona_Estado","Zona_Low","Zona_High",
        "Entrada","Price","SMA20","SMA50","SMA200","ATR14","Slope20","Slope50",
        "FaseBonifaz","BaseStop","Volatilidad_%",
        "Stop_Ajustado","Stop_Extendido","Stop_Sugerido","Stop_Sugerido_Nivel",
        "Riesgo_%_Ajustado","Riesgo_%_Extendido",
        "Entrada_Tipo","Entrada_Pullback","Zona_Pullback_Low","Zona_Pullback_High","Trigger_Pullback",
        "Entrada_Breakout","Zona_Breakout_Low","Zona_Breakout_High","Trigger_Breakout",
        "Entrada_Sugerida","Trigger","Comentario",
        "TP_Mode","TP_BaseMode","TP1","TP2","TP3",
        "TP_Status","TP_Nota","RecomendacionStop","SugerenciaAccion",
        "Prob_Exito_Corto_%","Prob_Exito_Medio_%","R_%","RR_TP1","RR_TP2"
    ]
    return out[[c for c in cols if c in out.columns]]

# =============================
# UI — Streamlit
# =============================
st.set_page_config(page_title="Bonifaz–Claro Dashboard v2", layout="wide")
st.title("📈 Bonifaz–Claro Dashboard v2")
st.caption("Ciclos + Tendencias • SMA20/50/200 • Stops Ajustados/Extendidos • TP estructurales • Excel • Zonas de compra (corregidas)")

with st.sidebar:
    st.header("⚙️ Parámetros")
    st.subheader("Portafolio actual (Excel opcional)")
    st.caption("Estructura: columnas 'Ticker', 'Entrada'.")
    pf_file = st.file_uploader("Subir Excel de portafolio", type=["xlsx", "xls"])

    st.subheader("Watchlist (Excel opcional)")
    st.caption("Estructura: columna 'Ticker'.")
    wl_file = st.file_uploader("Subir Excel de watchlist", type=["xlsx", "xls"], key="wl")

    tickers_text = st.text_input("Tickers manuales (coma)", ",".join(DEFAULT_TICKERS))
    tickers_manual = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    st.subheader("Entradas manuales")
    entradas_df = pd.DataFrame({"Ticker": tickers_manual, "Entrada": [DEFAULT_ENTRADAS.get(t, np.nan) for t in tickers_manual]})
    entradas_edit = st.data_editor(entradas_df, num_rows="dynamic", key="entradas_editor")
    entradas_manual = {row["Ticker"]: float(row["Entrada"]) if pd.notna(row["Entrada"]) else np.nan for _, row in entradas_edit.iterrows()}

    st.subheader("Stops")
    buffer_pct = st.slider("Buffer (%) bajo la media", 0.5, 3.0, DEFAULTS["buffer_pct"], 0.1)
    atr_mult = st.slider("Multiplicador ATR para Stop Extendido", 0.0, 2.0, DEFAULTS["atr_mult"], 0.1)
    days = st.slider("Días de histórico", 60, 400, DEFAULTS["days"], 10)

    st.subheader("Take Profits")
    tp_mode = st.selectbox("Modo TP", ["R", "ATR", "AUTO", "STRUCT"], index=["R","ATR","AUTO","STRUCT"].index(DEFAULTS["tp_mode"]))
    auto_low = st.number_input("AUTO — Volatilidad baja (%)", value=DEFAULTS["auto_vol_threshold_low"], step=0.1)
    auto_high = st.number_input("AUTO — Volatilidad alta (%)", value=DEFAULTS["auto_vol_threshold_high"], step=0.1)
    min_tp1_pct = st.number_input("Piso TP1 (%)", value=DEFAULTS["min_tp1_pct"], step=0.1)
    min_tp2_pct = st.number_input("Piso TP2 (%)", value=DEFAULTS["min_tp2_pct"], step=0.1)
    min_tp1_R = st.number_input("Piso TP1 (R)", value=DEFAULTS["min_tp1_R"], step=0.1)
    min_tp2_R = st.number_input("Piso TP2 (R)", value=DEFAULTS["min_tp2_R"], step=0.1)
    enable_tp3 = st.checkbox("Activar TP3", value=DEFAULTS["enable_tp3"])
    tp3_R = st.number_input("TP3 (R o ×ATR)", value=DEFAULTS["tp3_R"], step=0.5)
    auto_ladder = st.checkbox("Auto-ladder (promover TP si ya fue superado)", value=DEFAULTS["auto_ladder"])

    st.subheader("Estructura (STRUCT)")
    lookback = st.slider("Lookback resistencias (días)", 30, 250, DEFAULTS["lookback"], 5)
    pivot_span = st.slider("Sensibilidad pivotes (velas a cada lado)", 2, 10, DEFAULTS["pivot_span"], 1)
    resistance_buffer_pct = st.number_input("Margen bajo resistencia (%)", value=DEFAULTS["resistance_buffer_pct"], step=0.1)
    measured_move_frac = st.number_input("Measured move — fracción del rango", value=DEFAULTS["measured_move_frac"], step=0.1)
    breakout_cap_pct = st.number_input("Tope ruptura (% sobre precio)", value=DEFAULTS["breakout_cap_pct"], step=1.0)
    prefer_breakout_near_res_pct = st.number_input("Preferir breakout si precio ≤ X% de resistencia", value=2.5, step=0.1)

    st.subheader("Zonas de compra")
    pullback_zone_atr_low = st.number_input("Pullback: zona baja (×ATR)", value=DEFAULTS["pullback_zone_atr_low"], step=0.05)
    pullback_zone_atr_high = st.number_input("Pullback: zona alta (×ATR)", value=DEFAULTS["pullback_zone_atr_high"], step=0.05)
    breakout_zone_pct = st.number_input("Breakout: ancho de zona (%)", value=DEFAULTS["breakout_zone_pct"], step=0.05)
    entry_tolerance_pct = st.number_input("Tolerancia para 'En zona' (±%)", value=DEFAULTS["entry_tolerance_pct"], step=0.05)

    st.subheader("Estrategia preferida")
    strategy_override = st.radio("Usar estrategia fija (sobre AUTO)", ["AUTO", "Pullback", "Breakout"], index=0)
    strategy_override = strategy_override.strip().lower()  # "auto"/"pullback"/"breakout"

    run_btn = st.button("🔮 Calcular", use_container_width=True)

# Ejecutar
if run_btn:
    try:
        entradas = entradas_manual.copy()
        tickers = set(entradas.keys())
        if pf_file is not None:
            pf_df = pd.read_excel(pf_file)
            if not {"Ticker", "Entrada"}.issubset(pf_df.columns):
                st.error("El Excel de portafolio debe tener columnas 'Ticker' y 'Entrada'.")
            else:
                for _, r in pf_df.iterrows():
                    if pd.notna(r.get("Ticker")):
                        entradas[str(r["Ticker"]).upper()] = float(r.get("Entrada", np.nan)) if pd.notna(r.get("Entrada")) else np.nan
                tickers.update([str(t).upper() for t in pf_df["Ticker"].dropna().tolist()])
        tickers = sorted([t for t in tickers if t])

        tp_params = dict(
            auto_vol_threshold_low=auto_low,
            auto_vol_threshold_high=auto_high,
            min_tp1_pct=min_tp1_pct,
            min_tp2_pct=min_tp2_pct,
            min_tp1_R=min_tp1_R,
            min_tp2_R=min_tp2_R,
            enable_tp3=enable_tp3,
            tp3_R=tp3_R,
            auto_ladder=auto_ladder,
            resistance_buffer_pct=resistance_buffer_pct,
            measured_move_frac=measured_move_frac,
            tp_mode=tp_mode,
            breakout_cap_pct=breakout_cap_pct,
            prefer_breakout_near_res_pct=prefer_breakout_near_res_pct,
            pullback_zone_atr_low=pullback_zone_atr_low,
            pullback_zone_atr_high=pullback_zone_atr_high,
            breakout_zone_pct=breakout_zone_pct,
            entry_tolerance_pct=entry_tolerance_pct,
        )

        # 1) Portafolio / manual
        if tickers:
            st.subheader("Resultados — Portafolio / Manual")
            out = evaluate_universe(tickers, entradas, days, buffer_pct, atr_mult, auto_low, auto_high, lookback, tp_params, strategy_override=strategy_override)
            st.dataframe(out, use_container_width=True)

            b = io.BytesIO()
            with pd.ExcelWriter(b, engine="openpyxl") as writer:
                out.to_excel(writer, sheet_name="PORTAFOLIO", index=False)

            st.download_button(
                "💾 Descargar Excel Portafolio",
                data=b.getvalue(),
                file_name=f"bonifaz_dashboard_portafolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            # Gráficos por ticker
            st.subheader("Gráficos")
            raw = download_data(tickers, days)
            for t in tickers:
                if isinstance(raw.columns, pd.MultiIndex):
                    sub = pd.DataFrame(
                        {
                            "Open": raw[("Open", t)],
                            "High": raw[("High", t)],
                            "Low": raw[("Low", t)],
                            "Close": raw[("Adj Close", t)] if ("Adj Close", t) in raw.columns else raw[("Close", t)],
                        }
                    ).dropna()
                else:
                    sub = raw.rename(columns=str.title).dropna()
                tail = sub.tail(lookback).copy()
                closes = tail["Close"].tolist()
                sma20 = sub["Close"].rolling(20).mean().tail(lookback).tolist()
                sma50 = sub["Close"].rolling(50).mean().tail(lookback).tolist()
                sma200 = sub["Close"].rolling(200).mean().tail(lookback).tolist()
                idx = tail.index.astype(str).tolist()

                row = out[out["Ticker"] == t]
                if row.empty:
                    continue
                row = row.iloc[0]
                stop_level = row["Stop_Sugerido_Nivel"]
                tp1, tp2, tp3 = row["TP1"], row["TP2"], row["TP3"]
                zlow, zhigh = row.get("Zona_Low", np.nan), row.get("Zona_High", np.nan)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=idx, y=closes, mode="lines", name=f"{t} Close"))
                fig.add_trace(go.Scatter(x=idx, y=sma20, mode="lines", name="SMA20"))
                fig.add_trace(go.Scatter(x=idx, y=sma50, mode="lines", name="SMA50"))
                fig.add_trace(go.Scatter(x=idx, y=sma200, mode="lines", name="SMA200"))

                if not np.isnan(zlow) and not np.isnan(zhigh):
                    fig.add_hrect(y0=zlow, y1=zhigh, line_width=0, fillcolor="rgba(0,200,0,0.12)", annotation_text="Zona Compra", annotation_position="top left")

                if not np.isnan(stop_level):
                    fig.add_hline(y=stop_level, line_dash="dot", annotation_text="Stop sugerido", annotation_position="bottom right")
                if not np.isnan(tp1):
                    fig.add_hline(y=tp1, line_dash="dot", annotation_text="TP1", annotation_position="top right")
                if not np.isnan(tp2):
                    fig.add_hline(y=tp2, line_dash="dash", annotation_text="TP2", annotation_position="top right")
                if not np.isnan(tp3):
                    fig.add_hline(y=tp3, line_dash="solid", annotation_text="TP3", annotation_position="top right")

                fig.update_layout(height=380, title=f"{t} — precio vs SMA, zona y niveles", legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

        # 2) Watchlist
        if wl_file is not None:
            st.subheader("Screening — Watchlist Excel")
            wl_df = pd.read_excel(wl_file)
            if "Ticker" not in wl_df.columns:
                st.error("El Excel de watchlist debe tener columna 'Ticker'.")
            else:
                wl_tickers = sorted(list({str(x).upper().strip() for x in wl_df["Ticker"].dropna().tolist()}))
                if wl_tickers:
                    out_wl = evaluate_universe(
                        wl_tickers, {}, days, buffer_pct, atr_mult, auto_low, auto_high, lookback,
                        tp_params, fallback_price_if_nan=False, strategy_override=strategy_override
                    )
                    st.dataframe(out_wl, use_container_width=True)

                    b2 = io.BytesIO()
                    with pd.ExcelWriter(b2, engine="openpyxl") as writer:
                        out_wl.to_excel(writer, sheet_name="WATCHLIST", index=False)
                    st.download_button(
                        "💾 Descargar Excel Watchlist",
                        data=b2.getvalue(),
                        file_name=f"bonifaz_dashboard_watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
        st.exception(e)
