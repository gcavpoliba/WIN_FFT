#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rsl_core.py - Nucleo numerico per spettri di Fourier di accelerogrammi e
funzione di amplificazione (Risposta Sismica Locale).

Nessuna dipendenza oltre a numpy. Test rapido:  python rsl_core.py --selftest

Convenzioni
-----------
* Internamente le accelerazioni sono in m/s^2.
* Spettro di Fourier (default):  F(f_k) = dt * |FFT(a)_k|   [m/s^2 * s]
  = approssimazione discreta dell'integrale di Fourier continuo, indipendente
  dallo zero-padding. Altre normalizzazioni sono disponibili (NORMS).
* Frequenze: f_k = k / (N*dt),  k = 0 ... N/2.
* Funzione di amplificazione: AF(f) = |F_output(f)| / |F_input(f)|.
"""
import re
import sys
import numpy as np

G = 9.80665
UNITS = {"g": G, "m/s²": 1.0, "cm/s² (Gal)": 0.01, "mm/s²": 0.001}

BASELINES = ["Nessuna", "Rimuovi media", "Detrend lineare"]
PADDINGS = ["Potenza di 2 successiva", "Nessuno", "Potenza di 2 ≥ 2·N"]
NORMS = ["dt·|FFT|  (integrale di Fourier)", "|FFT| / N", "2·|FFT| / N", "|FFT| grezza"]
SMOOTHS = ["Nessuno", "Konno-Ohmachi", "Media mobile"]
SM_NONE, SM_KO, SM_MA = SMOOTHS


# --------------------------------------------------------------------------
# Lettura file
# --------------------------------------------------------------------------
def _tofloat(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def parse_line(line):
    """Ritorna la lista di float della riga, oppure None se non numerica."""
    s = line.strip()
    if not s:
        return None
    if ";" in s:
        toks = [t.replace(",", ".") for t in re.split(r"[;\s]+", s) if t]
    elif re.search(r"\s", s):
        toks = [t.strip(",").replace(",", ".") for t in s.split()]
        toks = [t for t in toks if t]
    else:
        toks = [t for t in s.split(",") if t]
    try:
        vals = [_tofloat(t) for t in toks]
    except ValueError:
        return None
    if not vals or not np.all(np.isfinite(vals)):
        return None
    return vals


_RE_DT = re.compile(r"\bDT\s*[=:]?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.I)
_RE_N = re.compile(r"\bNPTS\s*[=:]?\s*(\d+)", re.I)


def _is_time_column(t):
    if len(t) < 3:
        return False
    d = np.diff(t)
    if not np.all(d > 0):
        return False
    return np.std(d) <= 0.02 * np.mean(d)


def read_table(path):
    """Legge un file testo e ritorna un array (righe numeriche con >=2 colonne)."""
    with open(path, "r", errors="replace") as fh:
        rows = [r for r in (parse_line(l) for l in fh.read().splitlines()) if r]
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        raise ValueError("Nessuna tabella numerica (>=2 colonne) trovata nel file.")
    n = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    return np.array([r for r in rows if len(r) == n], float)


def read_accelerogram(path, dt_override=None, col=0):
    """
    Legge un accelerogramma da file di testo.

    Formati riconosciuti:
      * PEER NGA (.AT2): riga con 'NPTS= ..., DT= ...', poi valori in free-format
      * due (o più) colonne con tempo nella prima colonna (dt ricavato)
      * una colonna di accelerazioni (dt da specificare)
      * valori in sequenza su più colonne (col = -1) (dt da specificare)

    col: 0 = automatico; k>0 = colonna k (1-based) con le accelerazioni;
         -1 = tutti i valori in sequenza.
    Ritorna dict(acc=array (unità del file), dt=float, n=int).
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()

    hdr_dt, hdr_n, start = None, None, 0
    for i, l in enumerate(lines[:20]):
        mdt = _RE_DT.search(l)
        if mdt:
            hdr_dt = float(mdt.group(1))
            mn = _RE_N.search(l)
            hdr_n = int(mn.group(1)) if mn else None
            start = i + 1
            break

    rows = [r for r in (parse_line(l) for l in lines[start:]) if r]
    if not rows:
        raise ValueError("Nessun dato numerico trovato nel file.")

    dt = hdr_dt
    if hdr_dt is not None and col in (0, -1):
        acc = np.array([v for r in rows for v in r], float)
        if hdr_n and len(acc) >= hdr_n:
            acc = acc[:hdr_n]
    elif col == -1:
        acc = np.array([v for r in rows for v in r], float)
    else:
        lens = [len(r) for r in rows]
        ncol = max(set(lens), key=lens.count)
        full = np.array([r for r in rows if len(r) == ncol], float)
        if ncol == 1:
            acc = full[:, 0]
        else:
            if _is_time_column(full[:, 0]):
                dt = float(np.median(np.diff(full[:, 0])))
                idx = 1 if col == 0 else col - 1
            else:
                if col == 0:
                    raise ValueError(
                        "File a più colonne senza colonna dei tempi riconoscibile: "
                        "indicare la colonna (k) oppure -1 per leggere i valori in sequenza.")
                idx = col - 1
            if idx < 0 or idx >= ncol:
                raise ValueError(f"Colonna {col} inesistente (il file ha {ncol} colonne).")
            acc = full[:, idx]

    if dt_override:
        dt = float(dt_override)
    if dt is None or dt <= 0:
        raise ValueError("Passo di campionamento dt non determinabile: inserirlo nel campo dt.")
    if len(acc) < 8:
        raise ValueError("Accelerogramma troppo corto.")
    return {"acc": np.asarray(acc, float), "dt": float(dt), "n": len(acc)}


# --------------------------------------------------------------------------
# Elaborazione e FFT
# --------------------------------------------------------------------------
def next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def padded_length(n, mode):
    if mode == PADDINGS[1]:
        return int(n)
    if mode == PADDINGS[2]:
        return next_pow2(2 * n)
    return next_pow2(n)


def cosine_taper(n, pct):
    """Finestra a coseno (Tukey): pct % di campioni per ciascun lato."""
    m = min(int(round(n * pct / 100.0)), n // 2)
    w = np.ones(n)
    if m > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = ramp
        w[-m:] = ramp[::-1]
    return w


def preprocess(acc, baseline="Nessuna", taper_pct=0.0):
    a = np.array(acc, float)
    n = len(a)
    if baseline == BASELINES[1]:
        a -= a.mean()
    elif baseline == BASELINES[2]:
        x = np.arange(n)
        a -= np.polyval(np.polyfit(x, a, 1), x)
    if taper_pct and taper_pct > 0:
        a *= cosine_taper(n, taper_pct)
    return a


def compute_fft(acc, dt, baseline="Nessuna", taper_pct=0.0, padding=PADDINGS[0]):
    """Ritorna (f, X, N, a_elaborato) con X = rfft non normalizzata."""
    a = preprocess(acc, baseline, taper_pct)
    N = padded_length(len(a), padding)
    X = np.fft.rfft(a, n=N)
    f = np.fft.rfftfreq(N, d=dt)
    return f, X, N, a


def norm_factor(norm, dt, N):
    """Fattore moltiplicativo applicato a |FFT|."""
    if norm == NORMS[0]:
        return dt
    if norm == NORMS[1]:
        return 1.0 / N
    if norm == NORMS[2]:
        return 2.0 / N
    return 1.0


# --------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------
def konno_ohmachi(freq, amp, b=40.0):
    """Smoothing di Konno & Ohmachi (1998), finestra [sin(b log10(f/fc))/(b log10(f/fc))]^4."""
    f = np.asarray(freq, float)
    amp = np.asarray(amp, float)
    out = np.empty_like(amp)
    out[0] = amp[0]
    half = 10 ** (2.5 / b)  # oltre questo rapporto la finestra è trascurabile
    for i in range(1, len(f)):
        fc = f[i]
        i0 = np.searchsorted(f, fc / half)
        i1 = np.searchsorted(f, fc * half, side="right")
        ff = f[i0:i1]
        pos = ff > 0
        x = np.zeros_like(ff)
        x[pos] = b * np.log10(ff[pos] / fc)
        w = np.ones_like(ff)
        nz = np.abs(x) > 1e-9
        w[nz] = (np.sin(x[nz]) / x[nz]) ** 4
        w[~pos] = 0.0
        out[i] = np.sum(w * amp[i0:i1]) / np.sum(w)
    return out


def moving_average(amp, n):
    n = int(n)
    if n <= 1:
        return np.array(amp, float)
    if n % 2 == 0:
        n += 1
    k = np.ones(n)
    num = np.convolve(amp, k, mode="same")
    den = np.convolve(np.ones_like(amp), k, mode="same")
    return num / den


def smooth(freq, amp, mode, param):
    if mode == SM_KO:
        return konno_ohmachi(freq, amp, float(param))
    if mode == SM_MA:
        return moving_average(amp, param)
    return np.array(amp, float)


# --------------------------------------------------------------------------
# Funzione di amplificazione
# --------------------------------------------------------------------------
def peak_in_range(f, y, fmin, fmax):
    m = (f > 0) & (f >= fmin) & (f <= fmax) & np.isfinite(y)
    if not np.any(m):
        return None, None
    idx = np.argmax(np.where(m, y, -np.inf))
    return float(f[idx]), float(y[idx])


def _water(den, wl):
    if wl and wl > 0:
        return np.maximum(den, wl * den.max())
    return den


def amplification(acc_in, dt_in, acc_out, dt_out, baseline="Nessuna", taper_pct=0.0,
                  smooth_mode=SM_KO, smooth_param=40.0, smooth_target="spettri",
                  water_level=1e-4):
    """
    AF(f) = |F_out(f)| / |F_in(f)|, con F = dt*FFT (m/s^2 * s, accelerazioni in m/s^2).

    smooth_target: 'spettri' -> si smussano i due spettri e poi si fa il rapporto;
                   'rapporto' -> si fa il rapporto dei grezzi e poi si smussa.
    water_level:   il denominatore viene limitato inferiormente a
                   water_level * max(denominatore) (0 = disattivato).
    """
    notes = []
    a_in = preprocess(acc_in, baseline, taper_pct)
    a_out = preprocess(acc_out, baseline, taper_pct)

    if abs(dt_in - dt_out) <= 1e-6 * max(dt_in, dt_out):
        dt = dt_in
        N = next_pow2(max(len(a_in), len(a_out)))
        f = np.fft.rfftfreq(N, dt)
        XA = np.fft.rfft(a_in, n=N) * dt
        XB = np.fft.rfft(a_out, n=N) * dt
    else:
        dt = dt_in
        Na, Nb = next_pow2(len(a_in)), next_pow2(len(a_out))
        fa, fb = np.fft.rfftfreq(Na, dt_in), np.fft.rfftfreq(Nb, dt_out)
        XAf = np.fft.rfft(a_in, n=Na) * dt_in
        XBf = np.fft.rfft(a_out, n=Nb) * dt_out
        fmax = min(fa[-1], fb[-1])
        f = fa[fa <= fmax]
        XA = XAf[: len(f)]
        XB = np.interp(f, fb, XBf.real) + 1j * np.interp(f, fb, XBf.imag)
        N = Na
        notes.append(f"dt diversi (input {dt_in:g} s, output {dt_out:g} s): lo spettro di output è "
                     f"interpolato sulla griglia dell'input fino alla f di Nyquist comune "
                     f"({fmax:.4g} Hz); la fase è indicativa.")

    A, B = np.abs(XA), np.abs(XB)
    if A.max() == 0:
        raise ValueError("Lo spettro dell'accelerogramma di input è identicamente nullo.")

    AF_raw = B / _water(A, water_level)
    if smooth_mode == SM_NONE:
        A_s, B_s, AF_s = A, B, AF_raw
    elif smooth_target == "spettri":
        A_s = smooth(f, A, smooth_mode, smooth_param)
        B_s = smooth(f, B, smooth_mode, smooth_param)
        AF_s = B_s / _water(A_s, water_level)
    else:
        A_s, B_s = A, B
        AF_s = smooth(f, AF_raw, smooth_mode, smooth_param)

    return {"f": f, "A": A, "B": B, "A_s": A_s, "B_s": B_s, "AF_raw": AF_raw, "AF_s": AF_s,
            "phase": np.angle(XB * np.conj(XA)), "N": N, "dt": dt, "notes": notes,
            "a_in": a_in, "a_out": a_out}


# --------------------------------------------------------------------------
# Aiuto al confronto con un software esterno (es. SeismoSignal)
# --------------------------------------------------------------------------
def identify_normalization(k, dt, N, n0):
    """
    k = mediana(spettro_riferimento / |FFT| grezza). Cerca la combinazione
    (normalizzazione, fattore di unità) più vicina a k. Ritorna (nome, errore relativo).
    """
    cands = {
        "dt": dt,
        f"1/N (N={N})": 1.0 / N,
        f"2/N (N={N})": 2.0 / N,
        f"1/N0 (N0={n0}, senza padding)": 1.0 / n0,
        f"2/N0 (N0={n0}, senza padding)": 2.0 / n0,
        "1 (FFT grezza)": 1.0,
    }
    ufs = {"": 1.0, " × g": G, " ÷ g": 1 / G, " × 100": 100.0, " ÷ 100": 0.01,
           " × 100·g": 100 * G, " ÷ 100·g": 1 / (100 * G), " × 1000": 1000.0, " ÷ 1000": 0.001}
    best = (None, np.inf)
    for cn, cv in cands.items():
        for un, uv in ufs.items():
            err = abs(k / (cv * uv) - 1.0)
            if err < best[1]:
                best = (cn + un, err)
    return best


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def selftest():
    import os
    import tempfile

    tmp = tempfile.mkdtemp()

    # 1) PEER NGA
    p = os.path.join(tmp, "x.AT2")
    with open(p, "w") as fh:
        fh.write("PEER NGA STRONG MOTION DATABASE RECORD\nLOMA PRIETA\nACCELERATION TIME SERIES IN UNITS OF G\n"
                 "NPTS=     8, DT=   .0100 SEC\n  0.1 0.2 0.3 0.4 0.5\n 0.6 0.7 0.8\n")
    r = read_accelerogram(p)
    assert r["n"] == 8 and abs(r["dt"] - 0.01) < 1e-12 and abs(r["acc"][-1] - 0.8) < 1e-12

    # 2) tempo + accelerazione, con intestazione e virgola decimale
    p = os.path.join(tmp, "y.txt")
    with open(p, "w") as fh:
        fh.write("t(s)  acc\n" + "\n".join(f"{i*0.02:.2f}  {np.sin(i):.5f}".replace(".", ",") for i in range(50)))
    r = read_accelerogram(p)
    assert abs(r["dt"] - 0.02) < 1e-9 and r["n"] == 50

    # 3) sinusoide centrata su un bin: 2|FFT|/N = ampiezza
    dt, N = 0.01, 1000
    t = np.arange(N) * dt
    a = 0.7 * np.cos(2 * np.pi * 5.0 * t)
    f, X, Np, _ = compute_fft(a, dt, padding=PADDINGS[1])
    k = np.argmax(np.abs(X))
    assert abs(f[k] - 5.0) < 1e-9
    assert abs(np.abs(X[k]) * norm_factor(NORMS[2], dt, Np) - 0.7) < 1e-9

    # 4) Parseval con la normalizzazione dt·|FFT|
    rng = np.random.default_rng(1)
    a = rng.standard_normal(1500)
    f, X, Np, ap = compute_fft(a, 0.005, padding=PADDINGS[0])
    F = np.abs(X) * norm_factor(NORMS[0], 0.005, Np)
    df = f[1] - f[0]
    Ef = df * (F[0] ** 2 + 2 * np.sum(F[1:-1] ** 2) + F[-1] ** 2)
    Et = 0.005 * np.sum(ap ** 2)
    assert abs(Ef / Et - 1) < 1e-10

    # 5) smoothing
    ff = np.fft.rfftfreq(4096, 0.01)
    assert np.allclose(konno_ohmachi(ff, np.full(len(ff), 3.0), 40), 3.0)
    spike = np.zeros(len(ff)); spike[300] = 1.0
    assert konno_ohmachi(ff, spike, 40).max() < 1.0

    # 6) funzione di amplificazione nota
    N = 4096; dt = 0.01
    a_in = rng.standard_normal(N)
    f = np.fft.rfftfreq(N, dt)
    H = 1 + 4 * np.exp(-(((f - 3.0) / 0.5) ** 2))
    a_out = np.fft.irfft(np.fft.rfft(a_in) * H, n=N)
    res = amplification(a_in, dt, a_out, dt, "Nessuna", 0.0, SM_NONE, 0, "spettri", 0.0)
    assert np.max(np.abs(res["AF_raw"] - H)) < 1e-8
    res = amplification(a_in, dt, a_out, dt, "Nessuna", 0.0, SM_KO, 40, "spettri", 1e-4)
    f0, A0 = peak_in_range(res["f"], res["AF_s"], 0.5, 20)
    assert abs(f0 - 3.0) < 0.4 and 3.5 < A0 < 5.2, (f0, A0)

    # 7) identificazione normalizzazione
    name, err = identify_normalization(0.01 * G, 0.01, 4096, 3000)
    assert name == "dt × g" and err < 1e-9, (name, err)

    print("Self-test OK (lettura file, FFT, Parseval, smoothing, amplificazione, confronto).")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
