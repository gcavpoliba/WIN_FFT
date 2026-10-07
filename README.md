# RSL – Spettro di Fourier e funzione di amplificazione

Interfaccia grafica (Tkinter + Matplotlib) per la risposta sismica locale:

- **Spettro di Fourier** di un accelerogramma (PEER NGA `.AT2`, due colonne tempo/accelerazione, una colonna + dt),
  con linea di base, taper, zero-padding, normalizzazione e smoothing (Konno-Ohmachi / media mobile).
  Permette di sovrapporre lo spettro esportato da **SeismoSignal** e identifica la normalizzazione compatibile.
- **Funzione di amplificazione** AF(f) = |Output| / |Input| fra due accelerogrammi (base e superficie), con ricerca di f0.

## Uso da sorgente

```bat
pip install -r requirements.txt
python rsl_fft_gui.py
```

Test numerico (senza interfaccia): `python rsl_core.py --selftest`

## Eseguibile Windows

**In locale** (serve Python per Windows): doppio clic su `build_exe.bat` → `dist\RSL_FFT.exe`.

**Su GitHub** (senza installare nulla): il workflow `.github/workflows/build-windows.yml` compila l'exe su un runner Windows.
- *Actions → Build eseguibile Windows → Run workflow*: l'exe è scaricabile dagli *artifacts*.
- Creando un tag `v*` l'exe viene allegato automaticamente alla *Release*.

Note: l'exe è a file singolo, quindi il primo avvio può richiedere qualche secondo. Windows SmartScreen può
mostrare un avviso perché l'exe non è firmato.

## Creare il repository

```bash
git init -b main
git add .
git commit -m "Prima versione"
# con GitHub CLI:
gh repo create rsl-fft --public --source=. --push
# oppure crea il repo vuoto dal sito e poi:
# git remote add origin https://github.com/<utente>/rsl-fft.git && git push -u origin main

# rilascio dell'exe:
git tag v0.1.0
git push origin v0.1.0
```

Non è inclusa una licenza: aggiungi il file `LICENSE` che preferisci prima di rendere pubblico il repo.
