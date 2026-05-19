from pathlib import Path
import vallenae as vae
import pandas as pd
import numpy as np

#tradb_path = Path(r"C:\Users\ac128559\Nextcloud\Akustik Drahtbruch\01_Vorversuche\Akustik_Vorversuche\Muster04\PPTest_Kanal1u2.tradb")


# Achte auf den Ordnernamen UND den exakten Dateinamen
tradb_path = Path(r"C:\Users\yassi\Desktop\Messdaten\260305_Muster10_I_04.tradb")


def get_channel_1_entries():
    with vae.io.TraDatabase(tradb_path) as tradb:
        data_table = tradb.read()
    channel_1_entries = data_table[data_table.iloc[:, 1] == 1]
    return channel_1_entries

def build_waveform_table():
    channel_1_entries = get_channel_1_entries()

    waveform_list = []

    with vae.io.TraDatabase(tradb_path) as tradb:
        for trai in channel_1_entries.index:
            y, t = tradb.read_wave(trai)

            waveform_list.append({
                "TRAI": trai,
                "y": y,
                "t": t
            })

    df_waveforms = pd.DataFrame(waveform_list)
    return df_waveforms


if __name__ == "__main__":
    df = build_waveform_table()
    print(df.head())

