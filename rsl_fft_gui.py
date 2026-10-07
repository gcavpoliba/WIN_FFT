#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rsl_fft_gui.py - GUI per la Risposta Sismica Locale
  Scheda 1: lettura accelerogramma + spettro di Fourier (confrontabile con SeismoSignal)
  Scheda 2: due accelerogrammi (input/base e output/superficie) -> funzione di amplificazione

Requisiti: Python 3.8+, numpy, matplotlib (tkinter incluso in Python)
Uso:       python rsl_fft_gui.py     (rsl_core.py deve stare nella stessa cartella)
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import rsl_core as core


# --------------------------------------------------------------------------
# Utilità GUI
# --------------------------------------------------------------------------
def num(var, label, default=None):
    s = str(var.get()).strip().replace(",", ".")
    if not s:
        if default is None:
            raise ValueError(f"Il campo «{label}» è vuoto.")
        return default
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Valore non valido in «{label}»: {s}")


def add_combo(parent, row, label, var, values, width=26):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
    cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
    cb.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    return cb


def add_entry(parent, row, label, var, width=10):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
    e = ttk.Entry(parent, textvariable=var, width=width)
    e.grid(row=row, column=1, sticky="w", padx=4, pady=2)
    return e


def make_info(parent, height=11):
    fr = ttk.Frame(parent)
    txt = tk.Text(fr, height=height, width=46, wrap="word", font=("Consolas", 9), state="disabled")
    sb = ttk.Scrollbar(fr, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    return fr, txt


def set_text(txt, s):
    txt.configure(state="normal")
    txt.delete("1.0", "end")
    txt.insert("1.0", s)
    txt.configure(state="disabled")


def make_canvas(parent):
    fig = Figure(figsize=(8, 7), dpi=100, constrained_layout=True)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    tb.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return fig, canvas


def ulabel(disp):
    return disp.split(" ")[0]


def fit_y(ax, f, arrays, fmin, fmax, log):
    m = (f >= fmin) & (f <= fmax) & (f > 0)
    vals = np.concatenate([a[m] for a in arrays if a is not None])
    vals = vals[np.isfinite(vals)]
    if log:
        vals = vals[vals > 0]
    if vals.size == 0:
        return
    lo, hi = vals.min(), vals.max()
    if log:
        ax.set_ylim(max(lo, hi * 1e-5) / 1.5, hi * 1.5)
    else:
        pad = 0.06 * ((hi - lo) or 1.0)
        ax.set_ylim(max(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)


def save_table(path_title, header, data):
    path = filedialog.asksaveasfilename(
        title=path_title, defaultextension=".txt",
        filetypes=[("Testo, separatore TAB", "*.txt"), ("CSV, separatore ;", "*.csv")])
    if not path:
        return None
    delim = ";" if path.lower().endswith(".csv") else "\t"
    np.savetxt(path, data, delimiter=delim, header=header, comments="# ", fmt="%.8e")
    return path


# --------------------------------------------------------------------------
# Pannello di caricamento di una registrazione
# --------------------------------------------------------------------------
class RecordPanel(ttk.LabelFrame):
    def __init__(self, master, title):
        super().__init__(master, text=title, padding=4)
        self.title = title
        self.path = None
        self._key = None
        self._raw = None
        self.v_file = tk.StringVar(value="(nessun file caricato)")
        self.v_dt = tk.StringVar()
        self.v_col = tk.StringVar(value="0")
        self.v_unit = tk.StringVar(value="g")
        self.v_scale = tk.StringVar(value="1")

        ttk.Button(self, text="Apri file…", command=self.load).grid(row=0, column=0, sticky="w")
        ttk.Label(self, textvariable=self.v_file, wraplength=250, foreground="#1a4d8f").grid(
            row=0, column=1, columnspan=3, sticky="w", padx=6)

        ttk.Label(self, text="Unità del file").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(self, textvariable=self.v_unit, values=list(core.UNITS), state="readonly",
                     width=12).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(self, text="Fattore scala").grid(row=1, column=2, sticky="e")
        ttk.Entry(self, textvariable=self.v_scale, width=7).grid(row=1, column=3, sticky="w", padx=4)

        ttk.Label(self, text="dt [s] (vuoto=auto)").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(self, textvariable=self.v_dt, width=9).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(self, text="Colonna (0 auto, -1 seq.)").grid(row=2, column=2, sticky="e")
        ttk.Entry(self, textvariable=self.v_col, width=7).grid(row=2, column=3, sticky="w", padx=4)

    def load(self):
        path = filedialog.askopenfilename(
            title=f"{self.title}: scegli il file",
            filetypes=[("Accelerogrammi", "*.txt *.dat *.csv *.at2 *.AT2 *.asc *.acc *.tsv"),
                       ("Tutti i file", "*.*")])
        if not path:
            return
        old = (self.path, self._key, self._raw)
        self.path, self._key, self._raw = path, None, None
        try:
            rec = self.get_record()
        except Exception as e:
            self.path, self._key, self._raw = old
            messagebox.showerror("Errore di lettura", str(e))
            return
        n, dt = len(rec["acc"]), rec["dt"]
        self.v_file.set(f"{rec['name']}\nN = {n},  dt = {dt:g} s,  durata = {n*dt:.2f} s")

    def get_record(self):
        """Ritorna dict(name, dt, acc [m/s²]) usando i valori correnti dei campi."""
        if not self.path:
            raise ValueError(f"Nessun accelerogramma caricato ({self.title}).")
        dt_ov = num(self.v_dt, "dt", 0.0)
        if dt_ov < 0:
            raise ValueError("dt deve essere positivo.")
        col = int(num(self.v_col, "colonna", 0))
        scale = num(self.v_scale, "fattore scala", 1.0)
        key = (self.path, dt_ov, col)
        if key != self._key:
            self._raw = core.read_accelerogram(self.path, dt_ov or None, col)
            self._key = key
        acc = self._raw["acc"] * core.UNITS[self.v_unit.get()] * scale
        return {"name": os.path.basename(self.path), "dt": self._raw["dt"], "acc": acc}


# --------------------------------------------------------------------------
# Scheda 1: spettro di Fourier
# --------------------------------------------------------------------------
class FourierTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.last = None
        self.ref = None  # (f, amp) letto da SeismoSignal

        left = ttk.Frame(self, padding=4)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        self.rec = RecordPanel(left, "1) Accelerogramma")
        self.rec.pack(fill="x", pady=2)

        p = ttk.LabelFrame(left, text="2) Elaborazione / FFT", padding=4)
        p.pack(fill="x", pady=2)
        self.v_base = tk.StringVar(value=core.BASELINES[0])
        self.v_taper = tk.StringVar(value="0")
        self.v_pad = tk.StringVar(value=core.PADDINGS[0])
        self.v_norm = tk.StringVar(value=core.NORMS[0])
        self.v_smooth = tk.StringVar(value=core.SM_NONE)
        self.v_sparam = tk.StringVar(value="40")
        add_combo(p, 0, "Linea di base", self.v_base, core.BASELINES)
        add_entry(p, 1, "Taper coseno [% per lato]", self.v_taper)
        add_combo(p, 2, "Zero-padding", self.v_pad, core.PADDINGS)
        add_combo(p, 3, "Normalizzazione", self.v_norm, core.NORMS)
        cb = add_combo(p, 4, "Smoothing", self.v_smooth, core.SMOOTHS)
        cb.bind("<<ComboboxSelected>>", self._on_smooth)
        add_entry(p, 5, "Param. (b KO / n punti)", self.v_sparam)

        v = ttk.LabelFrame(left, text="3) Visualizzazione", padding=4)
        v.pack(fill="x", pady=2)
        self.v_disp = tk.StringVar(value="g")
        self.v_xlog = tk.BooleanVar(value=True)
        self.v_ylog = tk.BooleanVar(value=True)
        self.v_fmin = tk.StringVar(value="0.1")
        self.v_fmax = tk.StringVar(value="50")
        add_combo(v, 0, "Unità di visualizzazione", self.v_disp, list(core.UNITS)[:3], width=14).bind(
            "<<ComboboxSelected>>", lambda e: self.plot())
        ttk.Checkbutton(v, text="Asse f logaritmico", variable=self.v_xlog, command=self.plot).grid(
            row=1, column=0, sticky="w", padx=4)
        ttk.Checkbutton(v, text="Asse ampiezza log.", variable=self.v_ylog, command=self.plot).grid(
            row=1, column=1, sticky="w", padx=4)
        add_entry(v, 2, "f min [Hz]", self.v_fmin, 8)
        add_entry(v, 3, "f max [Hz]", self.v_fmax, 8)

        b = ttk.Frame(left)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="Calcola FFT", command=self.calculate).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(b, text="Aggiorna grafico", command=self.plot).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(b, text="Esporta spettro…", command=self.export).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(b, text="Rimuovi riferimento", command=self.clear_ref).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(b, text="Carica spettro SeismoSignal (riferimento)…", command=self.load_ref).grid(
            row=2, column=0, columnspan=2, padx=2, pady=2, sticky="ew")
        b.columnconfigure((0, 1), weight=1)

        fr, self.info = make_info(left, 12)
        fr.pack(fill="both", expand=True, pady=2)

        self.fig, self.canvas = make_canvas(right)

    # ---- opzioni -------------------------------------------------------
    def _on_smooth(self, _e=None):
        self.v_sparam.set("40" if self.v_smooth.get() == core.SM_KO else "11")

    def read_options(self):
        taper = num(self.v_taper, "taper", 0.0)
        if not 0 <= taper <= 50:
            raise ValueError("Il taper deve essere compreso tra 0 e 50 %.")
        sp = num(self.v_sparam, "parametro smoothing", 40.0)
        if sp <= 0:
            raise ValueError("Il parametro di smoothing deve essere positivo.")
        return dict(baseline=self.v_base.get(), taper=taper, padding=self.v_pad.get(),
                    norm=self.v_norm.get(), smooth=self.v_smooth.get(), sparam=sp)

    def view_options(self, fnyq):
        fmin = num(self.v_fmin, "f min", 0.1)
        fmax = min(num(self.v_fmax, "f max", fnyq), fnyq)
        if fmin <= 0 and self.v_xlog.get():
            fmin = 1e-3
        if fmin >= fmax:
            raise ValueError("f min deve essere minore di f max.")
        return fmin, fmax

    # ---- calcolo -------------------------------------------------------
    def calculate(self):
        try:
            rec = self.rec.get_record()
            o = self.read_options()
            self.config(cursor="watch")
            self.update_idletasks()
            f, X, N, a_proc = core.compute_fft(rec["acc"], rec["dt"], o["baseline"], o["taper"], o["padding"])
            nf = core.norm_factor(o["norm"], rec["dt"], N)
            amp = np.abs(X) * nf
            amp_s = core.smooth(f, amp, o["smooth"], o["sparam"]) if o["smooth"] != core.SM_NONE else None
        except Exception as e:
            self.config(cursor="")
            messagebox.showerror("Errore", str(e))
            return
        self.config(cursor="")
        self.last = dict(name=rec["name"], dt=rec["dt"], n=len(rec["acc"]), N=N, f=f, X=X,
                         amp=amp, amp_s=amp_s, a_proc=a_proc, opts=o)
        self.plot()

    def plot(self):
        L = self.last
        if L is None:
            return
        try:
            fmin, fmax = self.view_options(L["f"][-1])
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            return
        disp = self.v_disp.get()
        uf = core.UNITS[disp]
        f = L["f"]
        m = f > 0
        amp = L["amp"] / uf
        amp_s = None if L["amp_s"] is None else L["amp_s"] / uf

        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)
        t = np.arange(len(L["a_proc"])) * L["dt"]
        ax1.plot(t, L["a_proc"] / uf, color="k", lw=0.6)
        ax1.set_xlabel("Tempo [s]")
        ax1.set_ylabel(f"Accelerazione [{ulabel(disp)}]")
        ax1.set_title(f"{L['name']} (come usato nella FFT)")
        ax1.grid(True, alpha=0.3)

        smoothed = amp_s is not None
        ax2.plot(f[m], amp[m], lw=0.7 if smoothed else 1.2, color="0.55" if smoothed else "C0",
                 label="FFT")
        if smoothed:
            ax2.plot(f[m], amp_s[m], lw=1.6, color="C0", label=f"FFT smussata ({L['opts']['smooth']})")
        arrays = [amp, amp_s]
        if self.ref is not None:
            fr, ar = self.ref
            ax2.plot(fr[fr > 0], ar[fr > 0], "r--", lw=1.2, label="SeismoSignal (riferimento)")
            arrays.append(np.interp(f, fr, ar))
        ax2.set_xscale("log" if self.v_xlog.get() else "linear")
        ax2.set_yscale("log" if self.v_ylog.get() else "linear")
        ax2.set_xlim(fmin, fmax)
        fit_y(ax2, f, arrays, fmin, fmax, self.v_ylog.get())
        ax2.set_xlabel("Frequenza [Hz]")
        ax2.set_ylabel(f"Ampiezza di Fourier [{ulabel(disp)}·s]" if L["opts"]["norm"] == core.NORMS[0]
                       else f"Ampiezza di Fourier [{ulabel(disp)}] (norm.: {L['opts']['norm'].split()[0]})")
        ax2.set_title("Spettro di ampiezza di Fourier")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()
        self.update_info()

    def update_info(self):
        L = self.last
        disp = self.v_disp.get()
        uf = core.UNITS[disp]
        n, dt, N, f = L["n"], L["dt"], L["N"], L["f"]
        pga = np.max(np.abs(L["a_proc"]))
        amp = L["amp"] / uf
        k = int(np.argmax(np.where(f > 0, amp, -np.inf)))
        lines = [f"File: {L['name']}",
                 f"N campioni = {n}   dt = {dt:g} s   durata = {n*dt:.3f} s",
                 f"Lunghezza FFT N = {N}",
                 f"Δf = {1.0/(N*dt):.5g} Hz   f Nyquist = {0.5/dt:.5g} Hz",
                 f"PGA (dopo elaborazione) = {pga/core.G:.4f} g = {pga:.4f} m/s²",
                 f"Picco spettro = {amp[k]:.5g} a f = {f[k]:.4g} Hz",
                 f"Normalizzazione: {L['opts']['norm']}"]
        if self.ref is not None:
            lines += ["", "--- Confronto con il riferimento ---"] + self.reference_stats(uf)
        set_text(self.info, "\n".join(lines))

    # ---- riferimento SeismoSignal -------------------------------------
    def load_ref(self):
        path = filedialog.askopenfilename(
            title="Spettro di Fourier esportato da SeismoSignal (frequenza, ampiezza)",
            filetypes=[("Testo", "*.txt *.dat *.csv *.tsv *.asc"), ("Tutti i file", "*.*")])
        if not path:
            return
        try:
            T = core.read_table(path)
            order = np.argsort(T[:, 0])
            self.ref = (T[order, 0], T[order, 1])
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            return
        if self.last is None:
            messagebox.showinfo("Riferimento caricato",
                                "Riferimento caricato: calcola ora la FFT per vedere il confronto.\n"
                                "Imposta «Unità di visualizzazione» uguale all'unità dell'export di SeismoSignal.")
        else:
            self.plot()

    def clear_ref(self):
        self.ref = None
        self.plot()

    def reference_stats(self, uf):
        L, (fr, ar) = self.last, self.ref
        f = L["f"]
        raw = np.abs(L["X"]) / uf
        sel = (fr >= f[1]) & (fr <= f[-1]) & (ar > 0)
        if sel.sum() < 5:
            return ["Troppo pochi punti del riferimento nel range di frequenza."]
        r_raw = np.interp(fr[sel], f, raw)
        ok = r_raw > 0
        k = ar[sel][ok] / r_raw[ok]
        kmed = float(np.median(k))
        spread = (np.percentile(k, 95) - np.percentile(k, 5)) / kmed
        name, err = core.identify_normalization(kmed, L["dt"], L["N"], L["n"])
        ours = np.interp(fr[sel], f, L["amp"] / uf)
        rel = np.abs(ours / ar[sel] - 1.0)
        out = [f"Rif./|FFT| grezza: mediana = {kmed:.6g}",
               f"Dispersione 5-95% = {100*spread:.2f} %",
               (f"Normalizzazione compatibile: {name} (scarto {100*err:.2f} %)" if err < 0.02
                else "Nessuna normalizzazione semplice compatibile (< 2 %)."),
               f"Scarto relativo vs. FFT corrente: mediana {100*np.median(rel):.3f} %, "
               f"max {100*rel.max():.2f} %, N punti = {int(sel.sum())}"]
        if spread > 0.02:
            out.append("Dispersione alta: verificare padding, linea di base, taper o unità.")
        return out

    # ---- export --------------------------------------------------------
    def export(self):
        L = self.last
        if L is None:
            messagebox.showinfo("Esporta", "Calcola prima la FFT.")
            return
        uf = core.UNITS[self.v_disp.get()]
        cols = [L["f"], L["amp"] / uf]
        names = "f[Hz]\tampiezza"
        if L["amp_s"] is not None:
            cols.append(L["amp_s"] / uf)
            names += "\tampiezza_smussata"
        cols.append(np.angle(L["X"]))
        names += "\tfase[rad]"
        hdr = (f"File: {L['name']}  dt={L['dt']:g} s  N_FFT={L['N']}\n"
               f"Normalizzazione: {L['opts']['norm']}; unità: {ulabel(self.v_disp.get())}\n" + names)
        p = save_table("Esporta spettro di Fourier", hdr, np.column_stack(cols))
        if p:
            messagebox.showinfo("Esporta", f"Salvato:\n{p}")


# --------------------------------------------------------------------------
# Scheda 2: funzione di amplificazione
# --------------------------------------------------------------------------
class AmplificationTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.last = None

        left = ttk.Frame(self, padding=4)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        self.rec_in = RecordPanel(left, "1) INPUT (base / affioramento)")
        self.rec_in.pack(fill="x", pady=2)
        self.rec_out = RecordPanel(left, "2) OUTPUT (superficie / sito)")
        self.rec_out.pack(fill="x", pady=2)

        p = ttk.LabelFrame(left, text="3) Elaborazione", padding=4)
        p.pack(fill="x", pady=2)
        self.v_base = tk.StringVar(value=core.BASELINES[0])
        self.v_taper = tk.StringVar(value="0")
        self.v_smooth = tk.StringVar(value=core.SM_KO)
        self.v_sparam = tk.StringVar(value="40")
        self.v_target = tk.StringVar(value="Spettri (poi rapporto)")
        self.v_wl = tk.StringVar(value="1e-4")
        add_combo(p, 0, "Linea di base", self.v_base, core.BASELINES)
        add_entry(p, 1, "Taper coseno [% per lato]", self.v_taper)
        cb = add_combo(p, 2, "Smoothing", self.v_smooth, core.SMOOTHS)
        cb.bind("<<ComboboxSelected>>", self._on_smooth)
        add_entry(p, 3, "Param. (b KO / n punti)", self.v_sparam)
        add_combo(p, 4, "Smoothing applicato a", self.v_target, ["Spettri (poi rapporto)", "Rapporto (smoothing finale)"])
        add_entry(p, 5, "Water-level (frazione picco)", self.v_wl)

        v = ttk.LabelFrame(left, text="4) Visualizzazione", padding=4)
        v.pack(fill="x", pady=2)
        self.v_disp = tk.StringVar(value="g")
        self.v_xlog = tk.BooleanVar(value=True)
        self.v_ylog = tk.BooleanVar(value=False)
        self.v_fmin = tk.StringVar(value="0.1")
        self.v_fmax = tk.StringVar(value="30")
        add_combo(v, 0, "Unità accelerazioni", self.v_disp, list(core.UNITS)[:3], width=14).bind(
            "<<ComboboxSelected>>", lambda e: self.plot())
        ttk.Checkbutton(v, text="Asse f logaritmico", variable=self.v_xlog, command=self.plot).grid(
            row=1, column=0, sticky="w", padx=4)
        ttk.Checkbutton(v, text="AF in scala log.", variable=self.v_ylog, command=self.plot).grid(
            row=1, column=1, sticky="w", padx=4)
        add_entry(v, 2, "f min [Hz] (anche ricerca f0)", self.v_fmin, 8)
        add_entry(v, 3, "f max [Hz]", self.v_fmax, 8)

        b = ttk.Frame(left)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="Calcola amplificazione", command=self.calculate).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(b, text="Aggiorna grafico", command=self.plot).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(b, text="Esporta risultati…", command=self.export).grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="ew")
        b.columnconfigure((0, 1), weight=1)

        fr, self.info = make_info(left, 8)
        fr.pack(fill="both", expand=True, pady=2)

        self.fig, self.canvas = make_canvas(right)

    def _on_smooth(self, _e=None):
        self.v_sparam.set("40" if self.v_smooth.get() == core.SM_KO else "11")

    def calculate(self):
        try:
            ri, ro = self.rec_in.get_record(), self.rec_out.get_record()
            taper = num(self.v_taper, "taper", 0.0)
            if not 0 <= taper <= 50:
                raise ValueError("Il taper deve essere compreso tra 0 e 50 %.")
            sp = num(self.v_sparam, "parametro smoothing", 40.0)
            if sp <= 0:
                raise ValueError("Il parametro di smoothing deve essere positivo.")
            wl = num(self.v_wl, "water-level", 0.0)
            if wl < 0:
                raise ValueError("Il water-level non può essere negativo.")
            target = "spettri" if self.v_target.get().startswith("Spettri") else "rapporto"
            self.config(cursor="watch")
            self.update_idletasks()
            res = core.amplification(ri["acc"], ri["dt"], ro["acc"], ro["dt"], self.v_base.get(), taper,
                                     self.v_smooth.get(), sp, target, wl)
        except Exception as e:
            self.config(cursor="")
            messagebox.showerror("Errore", str(e))
            return
        self.config(cursor="")
        res.update(name_in=ri["name"], name_out=ro["name"], acc_in=ri["acc"], acc_out=ro["acc"],
                   dt_in=ri["dt"], dt_out=ro["dt"], smooth=self.v_smooth.get())
        self.last = res
        self.plot()

    def plot(self):
        L = self.last
        if L is None:
            return
        f = L["f"]
        try:
            fmin = num(self.v_fmin, "f min", 0.1)
            fmax = min(num(self.v_fmax, "f max", f[-1]), f[-1])
            if fmin <= 0 and self.v_xlog.get():
                fmin = 1e-3
            if fmin >= fmax:
                raise ValueError("f min deve essere minore di f max.")
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            return
        disp = self.v_disp.get()
        uf = core.UNITS[disp]
        m = f > 0
        xs = "log" if self.v_xlog.get() else "linear"
        sm = L["smooth"] != core.SM_NONE

        self.fig.clear()
        ax1 = self.fig.add_subplot(3, 1, 1)
        ax2 = self.fig.add_subplot(3, 1, 2)
        ax3 = self.fig.add_subplot(3, 1, 3)

        ax1.plot(np.arange(len(L["acc_in"])) * L["dt_in"], L["acc_in"] / uf, lw=0.6, color="C0",
                 label=f"Input: {L['name_in']}")
        ax1.plot(np.arange(len(L["acc_out"])) * L["dt_out"], L["acc_out"] / uf, lw=0.6, color="C3",
                 alpha=0.8, label=f"Output: {L['name_out']}")
        ax1.set_xlabel("Tempo [s]")
        ax1.set_ylabel(f"Acc. [{ulabel(disp)}]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

        for key, ks, col, lab in (("A", "A_s", "C0", "Input"), ("B", "B_s", "C3", "Output")):
            y_raw, y_s = L[key] / uf, L[ks] / uf
            ax2.plot(f[m], y_raw[m], lw=0.6 if sm else 1.2, color=col, alpha=0.35 if sm else 1.0,
                     label=f"{lab}" if not sm else None)
            if sm:
                ax2.plot(f[m], y_s[m], lw=1.6, color=col, label=f"{lab} (smussato)")
        ax2.set_xscale(xs)
        ax2.set_yscale("log")
        ax2.set_xlim(fmin, fmax)
        fit_y(ax2, f, [L["A"] / uf, L["B"] / uf], fmin, fmax, True)
        ax2.set_xlabel("Frequenza [Hz]")
        ax2.set_ylabel(f"Fourier [{ulabel(disp)}·s]")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="best", fontsize=8)

        ylog = self.v_ylog.get()
        ax3.plot(f[m], L["AF_raw"][m], lw=0.6 if sm else 1.4, color="0.55" if sm else "k",
                 label="Rapporto grezzo" if sm else "AF")
        if sm:
            ax3.plot(f[m], L["AF_s"][m], lw=1.8, color="k", label="AF (smussata)")
        ax3.axhline(1.0, color="r", ls=":", lw=1)
        f0, A0 = core.peak_in_range(f, L["AF_s"], fmin, fmax)
        if f0 is not None:
            ax3.plot([f0], [A0], "o", color="C1", ms=7, zorder=5)
            ax3.annotate(f"f0 = {f0:.3g} Hz\nAF = {A0:.3g}", (f0, A0), textcoords="offset points",
                         xytext=(12, -22), fontsize=9)
        ax3.set_xscale(xs)
        ax3.set_yscale("log" if ylog else "linear")
        ax3.set_xlim(fmin, fmax)
        fit_y(ax3, f, [L["AF_raw"], L["AF_s"]] if not sm else [L["AF_s"]], fmin, fmax, ylog)
        ax3.set_xlabel("Frequenza [Hz]")
        ax3.set_ylabel("AF = |Output| / |Input|")
        ax3.set_title("Funzione di amplificazione")
        ax3.grid(True, which="both", alpha=0.3)
        ax3.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

        lines = [f"Input : {L['name_in']}  (dt={L['dt_in']:g} s)",
                 f"Output: {L['name_out']}  (dt={L['dt_out']:g} s)",
                 f"N FFT = {L['N']}   Δf = {f[1]-f[0]:.5g} Hz   f max = {f[-1]:.4g} Hz"]
        if f0 is not None:
            lines.append(f"Picco AF nel range [{fmin:g}, {fmax:g}] Hz:  f0 = {f0:.4g} Hz,  AF = {A0:.4g}")
        pgai, pgao = np.max(np.abs(L["a_in"])), np.max(np.abs(L["a_out"]))
        lines.append(f"PGA input = {pgai/core.G:.4f} g,  PGA output = {pgao/core.G:.4f} g")
        lines.append(f"Rapporto PGA out/in = {pgao/pgai:.3f}")
        lines += L["notes"]
        set_text(self.info, "\n".join(lines))

    def export(self):
        L = self.last
        if L is None:
            messagebox.showinfo("Esporta", "Calcola prima la funzione di amplificazione.")
            return
        uf = core.UNITS[self.v_disp.get()]
        data = np.column_stack([L["f"], L["A"] / uf, L["B"] / uf, L["A_s"] / uf, L["B_s"] / uf,
                                L["AF_raw"], L["AF_s"], L["phase"]])
        hdr = (f"Input: {L['name_in']}  Output: {L['name_out']}  N_FFT={L['N']}  smoothing={L['smooth']}\n"
               f"Ampiezze di Fourier = dt*|FFT| in {ulabel(self.v_disp.get())}*s\n"
               "f[Hz]\t|Input|\t|Output|\t|Input|_sm\t|Output|_sm\tAF_grezza\tAF_smussata\tfase(out-in)[rad]")
        p = save_table("Esporta funzione di amplificazione", hdr, data)
        if p:
            messagebox.showinfo("Esporta", f"Salvato:\n{p}")


# --------------------------------------------------------------------------
# Finestra principale
# --------------------------------------------------------------------------
HELP = """CONFRONTO CON SEISMOSIGNAL

1) Esporta da SeismoSignal lo spettro di Fourier dello stesso accelerogramma
   (frequenza, ampiezza) in un file di testo.
2) Nella scheda «Spettro di Fourier» carica l'accelerogramma, imposta
   «Unità di visualizzazione» uguale all'unità dell'export e premi Calcola FFT.
3) «Carica spettro SeismoSignal (riferimento)…» sovrappone la curva
   e riporta il rapporto riferimento/|FFT| grezza: il programma indica
   quale normalizzazione (dt, 1/N, 2/N …) e quale fattore di unità la
   riproducono, e lo scarto residuo.

Se la dispersione del rapporto è alta, prova a variare: zero-padding
(potenza di 2 / nessuno), linea di base, taper.

NOTA: la normalizzazione dt·|FFT| (integrale di Fourier) non dipende dal
padding; le normalizzazioni in 1/N sì.

FUNZIONE DI AMPLIFICAZIONE
AF(f) = |F_output(f)| / |F_input(f)|. Se l'input è registrato in
profondità (within) e non in affioramento (outcrop), tienine conto
nell'interpretazione (fattore ~2 alle basse frequenze per base rigida).
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RSL – Spettro di Fourier e funzione di amplificazione")
        self.geometry("1380x860")
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        nb.add(FourierTab(nb), text="  Spettro di Fourier  ")
        nb.add(AmplificationTab(nb), text="  Funzione di amplificazione  ")
        menu = tk.Menu(self)
        menu.add_command(label="Guida", command=lambda: messagebox.showinfo("Guida", HELP))
        menu.add_command(label="Esci", command=self.destroy)
        self.config(menu=menu)


def main():
    if sys.platform == "win32":  # interfaccia nitida su schermi ad alta risoluzione
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    app = App()
    if "--smoke" in sys.argv:  # avvio e chiusura immediata (usato dalla CI)
        app.update()
        app.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
