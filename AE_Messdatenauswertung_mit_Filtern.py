from pathlib import Path
import matplotlib.pyplot as plt
import vallenae as vae
import numpy as np
import tkinter as tk
import openae.features
from tkinter import filedialog
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
# --- Korrektur: ALLES einlesen, was Counts oder Wellenformen hat ---
df_hits = pridb.read_hits()

# Falls df_hits nicht alle Zeilen enthält, die du in VisualAE siehst,
# liegt es oft an der Filterung. Wir stellen sicher, dass wir den vollen Index haben.
print(f"Anzahl Hits im Dataframe: {len(df_hits)}")




# --- SCHRITT 1: Öffnen der Transientendatenbank (.tradb) ---

# Wir nehmen an, die .tradb liegt im gleichen Verzeichnis wie die .pridb
tradb_path = pridb_path.with_suffix('.tradb')

if not tradb_path.exists():
    print(f"Warnung: Keine zugehörige .tradb unter {tradb_path} gefunden!")
else:
    # Verbindung zur TRADB herstellen
    tra_db = vae.io.TraDatabase(tradb_path)
    print(f"Erfolgreich mit TRADB verbunden: {tradb_path.name}")


    # --- SCHRITT 4: Automatisierte Berechnung für alle Hits ---
    # (Dieser Block ersetzt deine Zeilen 60 bis 131)

    def calculate_ae_features(hit_id, tra_database, samplerate):
        """Extrahiert Centroid und Variance für einen einzelnen Hit"""
        try:
            y, _ = tra_database.read_wave(hit_id)
            if len(y) == 0: return None, None

            # Fensterung und FFT
            windowed_y = y * np.hanning(len(y))
            spectrum = np.fft.rfft(windowed_y)

            # OpenAE Input Objekt (fs, y, spectrum)
            oae_input = oae_feat.Input(
                float(samplerate),
                y.astype(np.float32),
                spectrum.astype(np.complex64)
            )

            # Features berechnen
            sc = oae_feat.spectral_centroid(oae_input)
            sv = oae_feat.spectral_variance(oae_input)
            return sc, sv
        except:
            return None, None


    # --- KORRIGIERTER SCHLEIFEN-BLOCK ---

    print(f"Berechne spektrale Merkmale für {len(df_hits)} Hits...")

    # Samplerate fest definieren (basierend auf deinen Messdaten 2.00 MHz)
    # --- DIAGNOSE & BERECHNUNG: DIESEN BLOCK EINSETZEN ---

    # 1. Samplerate festlegen (Wichtig: muss vor der Schleife stehen)
    # --- FINALE KORREKTUR: Direkter Tabellen-Zugriff ---

    fs = 2.0e6

    # --- VERBESSERTE BERECHNUNG: Nutzt die TRAI-Referenz ---

    print(f"Starte Analyse von {len(df_hits)} Hits mit fs = {fs / 1e6:.2f} MHz...")

    # --- Vor der Schleife: Spalten initialisieren ---
    df_hits["spectral_centroid"] = np.nan
    df_hits["spectral_variance"] = np.nan
    df_hits["energy_openae"] = np.nan  # NEU
    df_hits["clearance_factor"] = np.nan  # NEU

    fs = 2.0e6  # Deine Samplerate

    print(f"Starte Analyse von {len(df_hits)} Hits...")

    for idx, row in df_hits.iterrows():
        try:
            trai = int(row["trai"])
            if trai > 0:
                y, t = tra_db.read_wave(trai)

                if len(y) > 0:
                    # 1. ENERGIE (OpenAE: E = 1/fs * sum(y^2))
                    energy = (1 / fs) * np.sum(np.square(y))

                    # 2. CLEARANCE FACTOR (OpenAE)
                    peak = np.max(np.abs(y))
                    mean_sqrt = np.mean(np.sqrt(np.abs(y)))
                    clf = peak / (mean_sqrt ** 2) if mean_sqrt > 0 else 0

                    # 3. SPEKTRALE FEATURES (Dein bisheriger Code)
                    windowed_y = y * np.hanning(len(y))
                    spectrum = np.fft.rfft(windowed_y)
                    oae_input = oae_feat.Input(float(fs), y.astype(np.float32), spectrum.astype(np.complex64))

                    # Ergebnisse in den DataFrame schreiben
                    df_hits.at[idx, "spectral_centroid"] = oae_feat.spectral_centroid(oae_input)
                    df_hits.at[idx, "spectral_variance"] = oae_feat.spectral_variance(oae_input)
                    df_hits.at[idx, "energy_openae"] = energy
                    df_hits.at[idx, "clearance_factor"] = clf

        except Exception as e:
            continue

    print("Feature Extraktion (inkl. Energie & CLF) abgeschlossen!")

    # Check für dich in der Konsole
    valid_count = df_hits["spectral_centroid"].notnull().sum()
    print(f"Berechnung abgeschlossen! Erfolgreich: {valid_count} von {len(df_hits)} Hits.")

    # ... (Ende deiner for-Schleife)

    # Hier steht bei dir wahrscheinlich:
    print(f"Berechnung abgeschlossen! Erfolgreich: {valid_count} von {len(df_hits)} Hits.")

    # --- GENAU HIER FÜGST DU DEN NEUEN TEIL EIN ---

    # 1. Berechnung (Wichtig: Sortieren nach Zeit für eine saubere Kurve)
    df_hits = df_hits.sort_values("time")
    df_hits["cum_energy"] = df_hits["energy_openae"].cumsum()

    # 2. Der Plot für die Energie-Akkumulation
    plt.figure(figsize=(10, 6))
    plt.plot(df_hits["time"], df_hits["cum_energy"], color="firebrick", linewidth=2.5, label="Kumulative Energie")
    plt.fill_between(df_hits["time"], df_hits["cum_energy"], color="red", alpha=0.1)
    plt.xlabel("Zeit [s]")
    plt.ylabel("Energie Akkumulation [V²s]")
    plt.title("Zerstörender Versuch: Energetischer Schadensverlauf über die Zeit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # --- DANACH KOMMT DEIN BESTEHENDER CODE FÜR DAS MASTER-DIAGRAMM ---
    # (df_plot = df_hits.dropna(subset=['spectral_centroid']).copy() ... usw.)




















    # Spezieller Check für Hit 391
    if 391 in df_hits.index:
        val = df_hits.loc[391, "spectral_centroid"]
        print(
            f"Status Hit 391: {'Berechnet (' + str(round(val / 1000, 2)) + ' kHz)' if not np.isnan(val) else 'Fehlt immer noch'}")


import pandas as pd

# --- ALLE ZEILEN IN DER KONSOLE ANZEIGEN ---

print("\n--- VOLLSTÄNDIGE LISTE ALLER 183 HITS ---")

# Mit diesem Befehl sagen wir Python: "Zeige ALLES an, egal wie lang es ist"
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
    # Wir zeigen die wichtigsten Spalten für alle 183 Zeilen
    print(df_hits[["channel", "time", "amplitude", "spectral_centroid", "spectral_variance"]])

print("\n--- ENDE DER LISTE ---")








#Von Franziska implementiert

# Überprüfen, ob die Spalten vorhanden sind
required_cols = ["time", "amplitude", "energy"]
for col in required_cols:
    if col not in df_hits:
        raise ValueError(f"Die Datenbank enthält keine '{col}'-Spalte!")

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


print(df_hits.columns)


print(df_hits[["spectral_centroid", "spectral_variance"]].head(15))




print(df_hits[["spectral_centroid", "spectral_variance"]].describe())


















#Zum Plotten von einem ausgewähltem Hit

import matplotlib.pyplot as plt
import numpy as np


# --- 1. DEFINITION DER FUNKTION ---
def plot_single_hit_analysis(hit_id):
    try:
        # 1. Daten laden
        trai = int(df_hits.loc[hit_id, "trai"])
        y, t = tra_db.read_wave(trai)

        # Umrechnung Wellenform in mV
        y_mv = y * 1000
        n = len(y)

        # 2. FFT berechnen & Normieren
        windowed_y = y * np.hanning(n)
        # Normierung durch n/2 für physikalisch korrekte Amplituden
        spectrum_linear = np.abs(np.fft.rfft(windowed_y)) / (n / 2)

        # Umrechnung in dB bezogen auf 1µV (Schutz gegen log(0))
        spectrum_db = 20 * np.log10(spectrum_linear / 1e-6 + 1e-9)

        # Frequenz-Achse in kHz
        freqs = np.fft.rfftfreq(n, 1 / fs) / 1000

        # 3. Plots ERSTELLEN (Wichtig: Hier werden ax1 und ax2 definiert!)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f"Analyse Hit ID: {hit_id} (Kanal {df_hits.loc[hit_id, 'channel']})", fontsize=14)

        # Oben: Zeitbereich (mV)
        ax1.plot(t * 1e6, y_mv, color='lime', linewidth=0.8)
        ax1.set_xlabel("Zeit [µs]")
        ax1.set_ylabel("Amplitude [mV]")
        ax1.set_title("Wellenform (Time Domain)")
        ax1.grid(True, alpha=0.3)

        # Unten: Frequenzbereich (dB)
        ax2.plot(freqs, spectrum_db, color='cyan', linewidth=0.8)
        ax2.set_xlabel("Frequenz [kHz]")
        ax2.set_ylabel("Magnitude [dB_µV]")
        ax2.set_title(f"Spektrum (FFT) - Schwerpunkt: {df_hits.loc[hit_id, 'spectral_centroid'] / 1000:.2f} kHz")

        # Skalierung anpassen
        ax2.set_xlim([0, 500])
        ax2.set_ylim([-20, 70])  # Jetzt sollte der Peak sauber zwischen 40 und 60 liegen
        ax2.grid(True, alpha=0.3)

        # Infobox
        info_text = (f"Amplitude: {df_hits.loc[hit_id, 'amplitude']} dB\n"
                     f"Energie: {df_hits.loc[hit_id, 'energy']:.2e}\n"
                     f"Counts: {df_hits.loc[hit_id, 'counts']}")
        plt.figtext(0.15, 0.85, info_text, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        # Falls oben was schiefgeht, sehen wir hier genau was
        print(f"Fehler beim Plotten von Hit {hit_id}: {e}")


# Aufruf
plot_single_hit_analysis(103)






#Implementierung des Correlation plot



import matplotlib.pyplot as plt
import numpy as np

# 1. Daten vorbereiten
df_plot = df_hits.dropna(subset=['spectral_centroid']).copy()

# 2. UMRECHNUNG: Volt in dB_AE (Anpassung an VisualAE)
# Wir nutzen die physikalische Definition: dB = 20 * log10(U / 1µV)
# Da die tradb-Werte meistens schon vorverstärkt sind,
# passen wir den Referenzpunkt so an, dass wir im Bereich 40-100 dB landen.
df_plot['amplitude_db'] = 20 * np.log10(df_plot['amplitude'] / 1e-6) - 40

# 3. Plot erstellen
plt.figure(figsize=(12, 7))

scatter = plt.scatter(
    df_plot['amplitude_db'],             # Jetzt die dB-Werte auf der X-Achse
    df_plot['spectral_centroid'] / 1000, # Frequenz in kHz
    c=df_plot['time'],
    cmap='plasma',
    alpha=0.7,
    edgecolors='w',
    linewidth=0.5
)

# 4. Design & Beschriftung
plt.xlabel('Amplitude [dB_AE]', fontsize=12)
plt.ylabel('Spectral Centroid [kHz]', fontsize=12)
plt.title('Master-Diagramm: Frequenz vs. Amplitude (dB)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Farbbalken für die Zeit
cbar = plt.colorbar(scatter)
cbar.set_label('Versuchszeit [s]', fontsize=12)

# Hilfslinie für die Frequenz-Trennung
plt.axhline(y=150, color='r', linestyle=':', alpha=0.4, label='Mögliche Riss-Grenze')

plt.legend()
plt.tight_layout()
plt.show()








# Realitäts-Check: Was sind unsere höchsten Volt-Werte?
top_hits = df_hits.sort_values(by='amplitude', ascending=False).head(10)
print("--- TOP 10 HITS IN PYTHON (VOLT) ---")
print(top_hits[['time', 'amplitude']])








































