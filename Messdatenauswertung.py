from pathlib import Path
import matplotlib.pyplot as plt
import vallenae as vae
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import openae as oae
import openae.features as oae_feat


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Tkinter-Hauptfenster ausblenden
root = tk.Tk()
root.withdraw()

# Datei-Auswahldialog öffnen
file_path = filedialog.askopenfilename(
    title="Wähle eine .pridb-Datei aus",
    filetypes=[("PriDB Dateien", "*.pridb"), ("Alle Dateien", "*.*")]
)

if not file_path:
    raise FileNotFoundError("Keine Datei ausgewählt!")

pridb_path = Path(file_path)

# Datenbank öffnen
pridb = vae.io.PriDatabase(pridb_path, mode='ro')

# Hits-Daten auslesen
df_hits = pridb.read_hits()

# Überprüfen, ob die Spalten vorhanden sind
required_cols = ["time", "amplitude", "energy"]
for col in required_cols:
    if col not in df_hits:
        raise ValueError(f"Die Datenbank enthält keine '{col}'-Spalte!")







# ==============================================================================
# Hier werden die Einheiten und Größenordnungen der pridb-features in dieselbe Form wie in VisualAE umgerechnet
# ==============================================================================
# Amplitude in dB umrechnen und direkt überschreiben (1 Nachkommastelle)
df_hits["amplitude"] = np.where(df_hits["amplitude"] > 0, 20 * np.log10(df_hits["amplitude"] / 1e-6), 0.0)
df_hits["amplitude"] = np.round(df_hits["amplitude"], 1)

# Rise Time (R) von Sekunden in Mikrosekunden (µs) umrechnen (1 Nachkommastelle)
df_hits["rise_time"] = np.round(df_hits["rise_time"] * 1e6, 1)

# Duration (D) von Sekunden in Mikrosekunden (µs) umrechnen (1 Nachkommastelle)
df_hits["duration"] = np.round(df_hits["duration"] * 1e6, 1)

# RMS von Volt in Mikrovolt (µV) umrechnen (1 Nachkommastelle)
df_hits["rms"] = np.round(df_hits["rms"] * 1e6, 1)

# Threshold (THR) von Volt in dB umrechnen (1 Nachkommastelle)
df_hits["threshold"] = np.where(df_hits["threshold"] > 0, 20 * np.log10(df_hits["threshold"] / 1e-6), 0.0)
df_hits["threshold"] = np.round(df_hits["threshold"], 1)
# ==============================================================================






##Dieser Block liest iterativ die -tradb-Rohdaten aus und führt zwei OpenAE-features drauf aus


# Pfad zur .tradb-Datei definieren (liegt im selben Ordner wie die .pridb)
tradb_path = str(pridb_path).replace(".pridb", ".tradb")

# Hier werden die entsprechenden Listen erstellt, in die später die berechneten OpenAE-features geldaen werden
spectral_centroids = []
peak_frequencies = []
clearance_factors = []
crest_factors = []
impulse_factors = []
kurtoises = []  # NEU











# Prüfen, ob die .tradb-Datei überhaupt existiert
if os.path.exists(tradb_path):
    print("Tradb-Datei gefunden! Starte die Feature-Extraktion...")
    fs = 2.0e6

    with vae.io.TraDatabase(tradb_path, mode='ro') as tra_db:
        for idx, row in df_hits.iterrows():
            trai = int(row["trai"])

            if trai > 0:
                try:
                    y, t = tra_db.read_wave(trai)

                    if len(y) > 0:
                        # --- Bestehende Frequenz-Features ---
                        windowed_y = y * np.hanning(len(y))
                        spectrum = np.fft.rfft(windowed_y)

                        f_axis = np.fft.rfftfreq(len(y), d=1 / fs)
                        idx_max = np.argmax(np.abs(spectrum))
                        f_haupt_khz = f_axis[idx_max] / 1000.0

                        oae_input = oae_feat.Input(
                            float(fs),
                            y.astype(np.float32),
                            spectrum.astype(np.complex64)
                        )
                        f_centroid_hz = oae_feat.spectral_centroid(oae_input)

                        spectral_centroids.append(np.round(f_centroid_hz / 1000.0, 1))
                        peak_frequencies.append(np.round(f_haupt_khz, 1))

                        # --- ALGO 1: CLEARANCE FACTOR ---
                        mean_sqrt = np.mean(np.sqrt(np.abs(y)))
                        if mean_sqrt > 0:
                            clf = np.max(np.abs(y)) / (mean_sqrt ** 2)
                            clearance_factors.append(np.round(clf, 2))
                        else:
                            clearance_factors.append(0.0)

                        # --- ALGO 2: CREST FACTOR ---
                        rms = np.sqrt(np.mean(y ** 2))
                        if rms > 0:
                            crf = np.max(np.abs(y)) / rms
                            crest_factors.append(np.round(crf, 2))
                        else:
                            crest_factors.append(0.0)

                        # --- ALGO 3: IMPULSE FACTOR ---
                        mean_abs = np.mean(np.abs(y))
                        if mean_abs > 0:
                            imf = np.max(np.abs(y)) / mean_abs
                            impulse_factors.append(np.round(imf, 2))
                        else:
                            impulse_factors.append(0.0)

                        # --- NEU: ALGO 4 - KURTOSIS (Exakt nach Doku) ---
                        y_centered = y - np.mean(y)
                        m2 = np.mean(y_centered ** 2)
                        m4 = np.mean(y_centered ** 4)

                        if m2 > 0:
                            kurt = m4 / (m2 ** 2)
                            kurtoises.append(np.round(kurt, 2))
                        else:
                            kurtoises.append(0.0)

                    else:
                        spectral_centroids.append(None)
                        peak_frequencies.append(None)
                        clearance_factors.append(None)
                        crest_factors.append(None)
                        impulse_factors.append(None)
                        kurtoises.append(None)

                except Exception as e:
                    print(f"Fehler bei Hit ID {idx} (TRAI {trai}): {e}")
                    spectral_centroids.append(None)
                    peak_frequencies.append(None)
                    clearance_factors.append(None)
                    crest_factors.append(None)
                    impulse_factors.append(None)
                    kurtoises.append(None)
            else:
                spectral_centroids.append(None)
                peak_frequencies.append(None)
                clearance_factors.append(None)
                crest_factors.append(None)
                impulse_factors.append(None)
                kurtoises.append(None)
else:
    print(f"Warnung: Die Datei {tradb_path} wurde nicht gefunden!")
    placeholder = [None] * len(df_hits)
    spectral_centroids = placeholder.copy()
    peak_frequencies = placeholder.copy()
    clearance_factors = placeholder.copy()
    crest_factors = placeholder.copy()
    impulse_factors = placeholder.copy()
    kurtoises = placeholder.copy()





# Die Ergebnisse an die Tabelle hängen
df_hits["spectral_centroid_khz"] = spectral_centroids
df_hits["peak_frequency_khz"] = peak_frequencies
df_hits["clearance_factor"] = clearance_factors
df_hits["crest_factor"] = crest_factors
df_hits["impulse_factor"] = impulse_factors
df_hits["kurtosis"] = kurtoises  # NEU




















# Zuerst nach Kanal trennen
df_hits_1 = df_hits[df_hits["channel"] == 1]
df_hits_2 = df_hits[df_hits["channel"] == 2]

df_channel_1 = df_hits_1[
    (df_hits_1["counts"] > 0) &
    (df_hits_1["energy"] > 0)
]

df_channel_2 = df_hits_2[
    (df_hits_2["counts"] > 0) &
    (df_hits_2["energy"] > 0)
]
# Dann Filter pro Kanal anwenden
#df_channel_1 = df_channel_1[df_channel_1["counts"] > 0]    # Kanal 1: counts > 40
#df_channel_2 = df_channel_2[df_channel_2["counts"] > 0]    # Kanal 2: energy > 50

# Zeit, Amplitude und Energie extrahieren
time_1 = df_channel_1["time"]
amplitude_1 = df_channel_1["amplitude"]
amplitude_1_db = 20 * np.log10(amplitude_1/1e-6)
energy_1 = df_channel_1["energy"]

time_2 = df_channel_2["time"]
amplitude_2 = df_channel_2["amplitude"]
amplitude_2_db = 20 * np.log10(amplitude_2/1e-6)
energy_2 = df_channel_2["energy"]

# ---- Plot 1: Amplitude ----
plt.figure(figsize=(10, 6))
plt.scatter(time_1, amplitude_1_db, marker='o', s=20, label="Seil", color="blue", alpha=0.7)
plt.scatter(time_2, amplitude_2_db, marker='o', s=20, label="Scheibe", color="red", alpha=0.7)

plt.xlabel("Zeit [s]")
plt.ylabel("Peak Amplitude [dB]")
plt.title("Amplitude über die Zeit für beide Kanäle")
# Gitter
plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

# Legende mit Kasten
plt.legend(frameon=True, edgecolor='black', facecolor='white', loc='best')

plt.tight_layout()
plt.show()


# ---- Plot 2: Energie ----
plt.figure(figsize=(10, 6))
plt.scatter(time_1, energy_1, marker='o', s=20, label="Seil", color="blue", alpha=0.7)
plt.scatter(time_2, energy_2, marker='o', s=20, label="Scheibe", color="red", alpha=0.7)

plt.xlabel("Zeit [s]")
plt.ylabel("Energie [a.u.]")
plt.title("Muster 02 ohne Filter")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---- Plot 3: Energie ----
plt.figure(figsize=(10, 6))
plt.scatter(time_1, energy_1, marker='o', s=20, label="Seil", color="blue", alpha=0.7)
#plt.scatter(time_2, energy_2, marker='o', s=20, label="Scheibe", color="red", alpha=0.7)

plt.xlabel("Zeit [s]")
plt.ylabel("Energie [a.u.]")  # Einheit je nach Vallen-Konfiguration
plt.title("Energie über die Zeit für beide Kanäle")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()












# ==============================================================================
# HIER NUR DEN EXPORT EINFÜGEN (Ganz unten im Skript)
# ==============================================================================
# ID aus dem Index als echte Spalte holen
df_export = df_hits.copy()
df_export.index.name = "Id"
df_export = df_export.reset_index()

# Die exakte Reihenfolge aller Eigenschaften inkl. param_id (DSET)
spalten_reihenfolge = [
    "Id",
    "param_id",
    "time",
    "channel",
    "amplitude",
    "duration",
    "energy",
    "rms",
    "status",
    "threshold",
    "rise_time",
    "counts",
    "trai",
    "spectral_centroid_khz",
    "peak_frequency_khz" ,
    "clearance_factor" ,
    "crest_factor" ,
    "impulse_factor" ,
    "kurtosis"
]

# Nur diese Spalten exportieren
df_export = df_export[spalten_reihenfolge]

# Als CSV speichern
df_export.to_csv("beweis_vollstaendige_tabelle.csv", index=False, sep=";")
print("Erfolgreich! CSV-Datei wurde mit allen korrekten Einheiten am Ende des Skripts gespeichert.")
# ==============================================================================