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
import math

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

fmin_param = 100000.0  # Untere Grenze in Hz (z.B. 100 kHz)
fmax_param = 400000.0  # Obere Grenze in Hz (z.B. 400 kHz)
rolloff_param = 0.95  # 95% Energie-Grenze laut Doku


spectral_centroids = []
peak_frequencies = []
spectral_peak_frequencies = []  # NEU für den exakten OpenAE-Namen (in Hz
clearance_factors = []
crest_factors = []
impulse_factors = []
kurtoises = []
partial_powers = []
shape_factors = []
skewnesses = []
spectral_entropies = []
spectral_flatnesses = []
spectral_kurtoises = []
spectral_rolloffs = []
spectral_skewnesses = []  # NEU (Letztes Feature!)



# Prüfen, ob die .tradb-Datei überhaupt existiert
if os.path.exists(tradb_path):
    print("Tradb-Datei gefunden! Starte die finale Feature-Extraktion ...")
    fs = 2.0e6

    with vae.io.TraDatabase(tradb_path, mode='ro') as tra_db:
        for idx, row in df_hits.iterrows():
            trai = int(row["trai"])

            if trai > 0:
                try:
                    y, t = tra_db.read_wave(trai)

                    if len(y) > 0:
                        # --- Frequenz-Features & Spektrum ---
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
                        spectral_peak_frequencies.append(np.round(f_haupt_khz * 1000.0, 1))

                        # --- ALGO 1: CLEARANCE FACTOR ---
                        mean_sqrt = np.mean(np.sqrt(np.abs(y)))
                        if mean_sqrt > 0:
                            clearance_factors.append(np.round(np.max(np.abs(y)) / (mean_sqrt ** 2), 2))
                        else:
                            clearance_factors.append(0.0)

                        # --- ALGO 2: CREST FACTOR ---
                        rms = np.sqrt(np.mean(y ** 2))
                        if rms > 0:
                            crest_factors.append(np.round(np.max(np.abs(y)) / rms, 2))
                        else:
                            crest_factors.append(0.0)

                        # --- ALGO 3: IMPULSE FACTOR ---
                        mean_abs = np.mean(np.abs(y))
                        if mean_abs > 0:
                            impulse_factors.append(np.round(np.max(np.abs(y)) / mean_abs, 2))
                        else:
                            impulse_factors.append(0.0)

                        # --- ALGO 4 & 7: STATISTISCHE MOMENTE (ZEITBEREICH) ---
                        y_centered = y - np.mean(y)
                        m2 = np.mean(y_centered ** 2)
                        m3 = np.mean(y_centered ** 3)
                        m4 = np.mean(y_centered ** 4)

                        if m2 > 0:
                            kurtoises.append(np.round(m4 / (m2 ** 2), 2))
                            skewnesses.append(np.round(m3 / (m2 ** 1.5), 2))
                        else:
                            kurtoises.append(0.0)
                            skewnesses.append(0.0)

                        # --- ALGO 5: PARTIAL POWER ---
                        ps = np.abs(spectrum) ** 2
                        n = len(ps)
                        n_lower = max(0, min(math.floor(2 * n * fmin_param / fs), n - 1))
                        n_upper = max(0, min(math.floor(2 * n * fmax_param / fs), n))
                        ps_sum = np.sum(ps)
                        if ps_sum > 0:
                            partial_powers.append(np.round(np.sum(ps[n_lower:n_upper]) / ps_sum, 4))
                        else:
                            partial_powers.append(0.0)

                        # --- ALGO 6: SHAPE FACTOR ---
                        if mean_abs > 0:
                            shape_factors.append(np.round(rms / mean_abs, 2))
                        else:
                            shape_factors.append(0.0)

                        # --- ALGO 8: SPECTRAL ENTROPY ---
                        if ps_sum > 0 and n > 1:
                            p_dist = ps / ps_sum
                            p_dist = p_dist[p_dist > 0]
                            if len(p_dist) > 0:
                                ent = -np.sum(p_dist * np.log2(p_dist)) / np.log2(n)
                                spectral_entropies.append(np.round(ent, 4))
                            else:
                                spectral_entropies.append(0.0)
                        else:
                            spectral_entropies.append(0.0)

                        # --- ALGO 9: SPECTRAL FLATNESS ---
                        arithmetic_mean = np.mean(ps)
                        if arithmetic_mean > 0:
                            ps_nonzero = ps[ps > 0]
                            if len(ps_nonzero) > 0:
                                geom_mean = np.exp(np.mean(np.log(ps_nonzero)))
                                spectral_flatnesses.append(np.round(geom_mean / arithmetic_mean, 4))
                            else:
                                spectral_flatnesses.append(0.0)
                        else:
                            spectral_flatnesses.append(0.0)

                        # --- ALGO 10 & 13: SPECTRAL KURTOSIS & SKEWNESS (Exakt nach Doku) ---
                        if ps_sum > 0:
                            p_spec = ps / ps_sum
                            spec_mean = np.sum(f_axis * p_spec)
                            spec_m2 = np.sum(((f_axis - spec_mean) ** 2) * p_spec)
                            spec_m3 = np.sum(((f_axis - spec_mean) ** 3) * p_spec)  # Für Spektrale Schiefe
                            spec_m4 = np.sum(((f_axis - spec_mean) ** 4) * p_spec)  # Für Spektrale Kurtosis

                            if spec_m2 > 0:
                                spectral_kurtoises.append(np.round(spec_m4 / (spec_m2 ** 2), 2))
                                # NEU: Spektrale Schiefe Formel umgesetzt
                                spec_skew = (spec_m3 / ps_sum) / np.sqrt(spec_m2 / ps_sum) ** 3
                                spectral_skewnesses.append(np.round(spec_skew, 2))
                            else:
                                spectral_kurtoises.append(0.0)
                                spectral_skewnesses.append(0.0)
                        else:
                            spectral_kurtoises.append(0.0)
                            spectral_skewnesses.append(0.0)

                        # --- ALGO 12: SPECTRAL ROLLOFF ---
                        if ps_sum > 0:
                            ps_cumsum = np.cumsum(ps)
                            ps_sum_rolloff = rolloff_param * ps_sum
                            idx_rolloff = np.where(ps_cumsum >= ps_sum_rolloff)[0][0]
                            f_rolloff_hz = 0.5 * fs / (n - 1) * idx_rolloff
                            spectral_rolloffs.append(np.round(f_rolloff_hz, 1))
                        else:
                            spectral_rolloffs.append(0.0)

                    else:
                        for lst in [spectral_centroids, peak_frequencies, spectral_peak_frequencies, clearance_factors,
                                    crest_factors, impulse_factors, kurtoises, partial_powers, shape_factors,
                                    skewnesses, spectral_entropies, spectral_flatnesses, spectral_kurtoises,
                                    spectral_rolloffs, spectral_skewnesses]:
                            lst.append(None)

                except Exception as e:
                    print(f"Fehler bei Hit ID {idx} (TRAI {trai}): {e}")
                    for lst in [spectral_centroids, peak_frequencies, spectral_peak_frequencies, clearance_factors,
                                crest_factors, impulse_factors, kurtoises, partial_powers, shape_factors,
                                skewnesses, spectral_entropies, spectral_flatnesses, spectral_kurtoises,
                                spectral_rolloffs, spectral_skewnesses]:
                        lst.append(None)
            else:
                for lst in [spectral_centroids, peak_frequencies, spectral_peak_frequencies, clearance_factors,
                            crest_factors, impulse_factors, kurtoises, partial_powers, shape_factors,
                            skewnesses, spectral_entropies, spectral_flatnesses, spectral_kurtoises, spectral_rolloffs,
                            spectral_skewnesses]:
                    lst.append(None)
else:
    print(f"Warnung: Die Datei {tradb_path} wurde nicht gefunden!")
    placeholder = [None] * len(df_hits)
    spectral_centroids = placeholder.copy()
    peak_frequencies = placeholder.copy()
    spectral_peak_frequencies = placeholder.copy()
    clearance_factors = placeholder.copy()
    crest_factors = placeholder.copy()
    impulse_factors = placeholder.copy()
    kurtoises = placeholder.copy()
    partial_powers = placeholder.copy()
    shape_factors = placeholder.copy()
    skewnesses = placeholder.copy()
    spectral_entropies = placeholder.copy()
    spectral_flatnesses = placeholder.copy()
    spectral_kurtoises = placeholder.copy()
    spectral_rolloffs = placeholder.copy()
    spectral_skewnesses = placeholder.copy()





df_hits["spectral_centroid_khz"] = spectral_centroids
df_hits["peak_frequency_khz"] = peak_frequencies
df_hits["spectral_peak_frequency_hz"] = spectral_peak_frequencies
df_hits["clearance_factor"] = clearance_factors
df_hits["crest_factor"] = crest_factors
df_hits["impulse_factor"] = impulse_factors
df_hits["kurtosis"] = kurtoises
df_hits["partial_power_100_400khz"] = partial_powers
df_hits["shape_factor"] = shape_factors
df_hits["skewness"] = skewnesses
df_hits["spectral_entropy"] = spectral_entropies
df_hits["spectral_flatness"] = spectral_flatnesses
df_hits["spectral_kurtosis"] = spectral_kurtoises
df_hits["spectral_rolloff_hz"] = spectral_rolloffs
df_hits["spectral_skewness"] = spectral_skewnesses  # NEU (Letztes Puzzleteil!)


















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
    "kurtosis" ,
    "partial_power_100_400khz" ,
    "shape_factor" ,
    "skewness" ,
    "spectral_entropy" ,
    "spectral_flatness" ,
    "spectral_kurtosis" ,
    "spectral_peak_frequency_hz" ,
    "spectral_rolloff_hz" ,
    "spectral_skewness"
]

# Nur diese Spalten exportieren
df_export = df_export[spalten_reihenfolge]

# Als CSV speichern
df_export.to_csv("beweis_vollstaendige_tabelle.csv", index=False, sep=";")
print("Erfolgreich! CSV-Datei wurde mit allen korrekten Einheiten am Ende des Skripts gespeichert.")
# ==============================================================================