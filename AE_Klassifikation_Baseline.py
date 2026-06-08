import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture



# 1. CSV-Datei aus der Feature-Extraktion einlesen
# Wichtig: sep=";" nutzen, da wir es gestern so abgespeichert haben!
dateiname = "Vollstaendige_tabelle.csv"
df = pd.read_csv(dateiname, sep=";")

print(f"Erfolgreich geladen! Die Tabelle hat {df.shape[0]} Zeilen und {df.shape[1]} Spalten.\n")

# 2. Schneller Klassen-Check: Wie viele Nullen und Einsen haben wir wirklich?
klassen_verteilung = df["ml_label"].value_counts()
prozent_verteilung = df["ml_label"].value_counts(normalize=True) * 100

print("=== KLASSENVERTEILUNG DER AE-HITS ===")
for klasse in klassen_verteilung.index:
    typ = "Echter Riss (1)" if klasse == 1 else "Harmloses Rauschen (0)"
    anzahl = klassen_verteilung[klasse]
    prozent = prozent_verteilung[klasse]
    print(f"{typ}: {anzahl} Hits ({prozent:.2f}%)")






## Schritt 1 und 2 des K-means ALgorithmus
    # 1. Die exakte Liste deiner 19 OpenAE-Features
    # Hinweis: Falls Python gleich meckert, dass eine Spalte fehlt, passen wir die Namen kurz an!
    feature_spalten = [
        "clearance_factor","crest_factor","energy","impulse_factor","kurtosis","partial_power_100_400khz","amplitude",
        "rms","shape_factor","skewness","spectral_centroid_khz","spectral_entropy","spectral_flatness","spectral_kurtosis",
        "spectral_peak_frequency_hz","spectral_rolloff_hz","spectral_skewness","spectral_variance_khz2",
        "zero_crossing_rate_hz",
    ]

    # 2. Die Feature-Matrix X aus der vollständigen Tabelle herausschneiden
    X = df[feature_spalten]

    # 3. Den StandardScaler initialisieren und die Daten transformieren
    # Hier passiert die Z-Transformation: Mittelwert = 0, Standardabweichung = 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== SCHRITT 1: ERFOLGREICH ABGESCHLOSSEN ===")
    print(f"Feature-Matrix X isoliert: {X.shape[0]} Hits und {X.shape[1]} Features.")
    print(f"Standardisierte Datenform (X_scaled): {X_scaled.shape}")
    print("Bereit für den K-Means-Algorithmus!\n")
    






## Implementierung von Schritt 3 und 4

# 1. K-Means Modell definieren
# n_clusters=2: Wir suchen nach 2 physikalischen Gruppen (Riss vs. Rauschen)
# random_state=42: Friert den Zufall ein, damit deine Ergebnisse reproduzierbar bleiben
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# 2. Das Clustering ausführen (Schwerpunkte berechnen und Hits zuordnen)
# Wichtig: Wir füttern die standardisierten Daten (X_scaled)!
cluster_labels = kmeans.fit_predict(X_scaled)

# 3. Die neuen Cluster-Ergebnisse direkt in deine originale Tabelle abspeichern
df["kmeans_cluster"] = cluster_labels

# 4. Mathematische Qualitätsprüfung: Der Silhouette Score
score = silhouette_score(X_scaled, cluster_labels)

print("=== SCHRITT 3: K-MEANS CLUSTERING === ")
print(f"Silhouette Score: {score:.4f}")
print("\nVerteilung der Hits auf die zwei Cluster:")
print(df["kmeans_cluster"].value_counts())


## ==============================================================================
## EIGENE Implementierung des DBSCAN- Algorithmus #
## ==============================================================================

# 1. DBSCAN-Modell definieren
# Radius eps auf 1.5 gesetzt
# min_samples auf 5 gesetzt
dbscan = DBSCAN(eps=1.6, min_samples=5)

# 2. Clustering mit DBSCAN ausführen (Dichteverkettung und Clusterwolken bestimmen)
# Wichtig: .fit_predict nutzen, um direkt die echten Labels (0, 1, -1) zu bekommen!
cluster_labels_dbscan = dbscan.fit_predict(X_scaled)

# 3. Die neuen Cluster-Ergebnisse in der originalen Tabelle abspeichern
df["dbscan_cluster"] = cluster_labels_dbscan

# 4. Kurzer Check im Terminal
print("\n=== SCHRITT 3.1: DBSCAN CLUSTERING ===")
print("Verteilung der Hits bei DBSCAN (-1 bedeutet Ausreißer/Noise):")
print(df["dbscan_cluster"].value_counts())




## ==============================================================================
##  Implementierung des Gaussian mixture models
## ==============================================================================

# 1. GMM-Modell definieren
# Wir suchen n_components=2 Gruppen (Riss vs. Rauschen)
# GMM nutzt statistische Glockenkurven (Gauß-Verteilungen) im 19D-Raum
gmm = GaussianMixture(n_components=2, random_state=42)

# 2. Clustering mit GMM ausführen
# Der EM-Algorithmus (Expectation-Maximization) optimiert iterativ die Kurvenlage
cluster_labels_gmm = gmm.fit_predict(X_scaled)

# 3. Die neuen GMM-Cluster-Ergebnisse in der originalen Tabelle abspeichern
df["gmm_cluster"] = cluster_labels_gmm

# 4. Kurzer Check im Terminal
print("\n=== SCHRITT 3.2: GMM CLUSTERING ===")
print("Verteilung der Hits bei GMM:")
print(df["gmm_cluster"].value_counts())














# # Da K-Means die Cluster-Nummern (0 und 1) zufällig vergibt, schauen wir kurz,
# # welches Cluster zu welcher deiner Klassen passt, indem wir die Kreuztabelle drucken.
# kreuztabelle = pd.crosstab(
#     df["ml_label"],
#     df["kmeans_cluster"],
#     margins=True  # Zeigt uns die Gesamtsummen an
# )
#
# print("=== SCHRITT 4: DER WISSENSCHAFTLICHE VERGLEICH ===")
# print("Kreuztabelle (Zeilen: Deine Regeln | Spalten: K-Means):")
# print(kreuztabelle)
# print("\n" + "="*50)
#
# # Jetzt filtern wir gezielt die 16 'Streitfälle' heraus, um sie zu analysieren!
# # Wir suchen die Hits, bei denen deine Regel 'Rauschen (0)' sagt, aber K-Means 'Cluster 1'
# streitfaelle = df[(df["ml_label"] == 0) & (df["kmeans_cluster"] == 1)]
#
# print(f"Hier sind die ersten 5 von den 16 Streitfällen mit ihren echten Werten:")
# spalten_fuer_blick = ["time", "counts", "energy", "amplitude", "spectral_entropy", "kmeans_cluster"]
# print(streitfaelle[spalten_fuer_blick].head())
#
# # print(df.head())
# #
# #



## ================================================================================================================
## Implementierung der Kontingenzmatrizen/Korrelationstabellen zwischen den drei Algorithmen
## ================================================================================================================

# 1. Kontingenztabelle zwischen K-Means und GMM
corelation_table1 = pd.crosstab(df["kmeans_cluster"], df["gmm_cluster"])
print("\n=== KREUZTABELLE: K-MEANS vs GMM ===")
print(corelation_table1)


#2. Kontingenztabelle zwischen GMM und DBSCAN
corelation_table2 = pd.crosstab(df["gmm_cluster"], df["dbscan_cluster"])
print("\n=== KREUZTABELLE: DBSCAN vs GMM ===")
print(corelation_table2)


#3. Kontingenztabelle zwischen K-MEANS und DBSCAN
corelation_table3 = pd.crosstab(df["kmeans_cluster"], df["dbscan_cluster"])
print("\n=== KREUZTABELLE: K-MEANS vs DBSCAN ===")
print(corelation_table3)