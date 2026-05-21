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
spectral_variances = []
spectral_skewnesses = []
spectral_kurtoises = []
weighted_peak_freqs = []














# Prüfen, ob die .tradb-Datei überhaupt existiert
if os.path.exists(tradb_path):
    print("Tradb-Datei gefunden! Starte die Feature-Extraktion für jeden Hit...")

    # Samplerate fest definieren (wie in deinem funktionierenden Skript)
    fs = 2.0e6

    # Öffnen der transienten Datenbank
    with vae.io.TraDatabase(tradb_path, mode='ro') as tra_db:

        # Wir fragen iterytiv den trai-index df_hits ab um danach die entsprechende Wellenform zu adressieren
        for idx, row in df_hits.iterrows():
            trai = int(row["trai"])
            # If-Bedingung prüft, ob überhaupt ein trai-index vorhanden ist, sprich ob eine Wellenform für den nächsten Iterationszyklus existiert
            if trai > 0:
                try:
                    # === FIX: y und t getrennt abfangen ===
                    y, t = tra_db.read_wave(trai)

                    if len(y) > 0:
                        # 1. Spektrale Features vorbereiten (Hanning-Fensterung + FFT)
                        windowed_y = y * np.hanning(len(y))
                        spectrum = np.fft.rfft(windowed_y)

                        # Peak Frequency bestimmen (für unsere Tabelle)
                        f_axis = np.fft.rfftfreq(len(y), d=1 / fs)
                        idx_max = np.argmax(np.abs(spectrum))
                        f_haupt_khz = f_axis[idx_max] / 1000.0

                        # 2. OpenAE Input-Objekt exakt wie im alten Skript füttern
                        oae_input = oae_feat.Input(
                            float(fs),
                            y.astype(np.float32),
                            spectrum.astype(np.complex64)
                        )

                        # 3. Features über OpenAE berechnen und in kHz umrechnen
                        f_centroid_hz = oae_feat.spectral_centroid(oae_input)
                        f_schwerpunkt_khz = f_centroid_hz / 1000.0

                        # Werte gerundet in die Listen schreiben
                        spectral_centroids.append(np.round(f_schwerpunkt_khz, 1))
                        peak_frequencies.append(np.round(f_haupt_khz, 1))
                    else:
                        spectral_centroids.append(None)
                        peak_frequencies.append(None)

                except Exception as e:
                    print(f"Fehler bei Hit ID {idx} (TRAI {trai}): {e}")
                    spectral_centroids.append(None)
                    peak_frequencies.append(None)
            else:
                spectral_centroids.append(None)
                peak_frequencies.append(None)
else:
    print(f"Warnung: Die Datei {tradb_path} wurde nicht gefunden!")
    spectral_centroids = [None] * len(df_hits)
    peak_frequencies = [None] * len(df_hits)

# Die Ergebnisse an die Tabelle hängen
df_hits["spectral_centroid_khz"] = spectral_centroids
df_hits["peak_frequency_khz"] = peak_frequencies






















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
    "spectral_centroid_khz",  # OpenAE-feature
    "peak_frequency_khz"       # OpenAE-feature
]

# Nur diese Spalten exportieren
df_export = df_export[spalten_reihenfolge]

# Als CSV speichern
df_export.to_csv("beweis_vollstaendige_tabelle.csv", index=False, sep=";")
print("Erfolgreich! CSV-Datei wurde mit allen korrekten Einheiten am Ende des Skripts gespeichert.")
# ==============================================================================