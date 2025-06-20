import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_valve_dataset():
    # Load data mentah
    df_ot = pd.read_csv('../valveplatefailure_raw/dane_OT.csv')
    df_ut1 = pd.read_csv('../valveplatefailure_raw/dane_UT1.csv')
    df_ut2 = pd.read_csv('../valveplatefailure_raw/dane_UT2.csv')
    df_ut3 = pd.read_csv('../valveplatefailure_raw/dane_UT3.csv')

    # Labeling
    df_ot['label'] = 0
    df_ut1['label'] = 1
    df_ut2['label'] = 2
    df_ut3['label'] = 3

    # Gabungkan
    df = pd.concat([df_ot, df_ut1, df_ut2, df_ut3], ignore_index=True)

    # Imputasi missing 'Flow - output' pakai median
    median_flow = df['Flow - output'].median()
    df['Flow - output'].fillna(median_flow)

    # Hapus duplikat jika ada
    df = df.drop_duplicates().reset_index(drop=True)

    # Standarisasi fitur numerik (kecuali 'stan', 'label')
    features_to_scale = df.select_dtypes(include='number').drop(columns=['stan', 'label'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_to_scale)
    df_scaled = pd.DataFrame(scaled, columns=features_to_scale.columns)

    # Gabungkan kembali
    df_clean = pd.concat([df_scaled, df[['stan', 'label']]], axis=1)

    # Simpan hasil
    df_clean.to_csv('../preprocessing/valve_plate_clean_automate.csv', index=False)
    print("✅ Dataset telah diproses dan disimpan ke  folder preprocessing dengan nama valve_plate_clean_automate.csv")

if __name__ == "__main__":
    preprocess_valve_dataset()
