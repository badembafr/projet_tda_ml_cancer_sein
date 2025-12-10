import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import DATA_RAW_PATH, RANDOM_STATE, TEST_SIZE


def load_and_prepare_data(filepath=DATA_RAW_PATH):
    # chargement et preparation des donnees
    df = pd.read_csv(filepath) 
    
    print("\nChargement des données") 
    print(f"Dimensions: {df.shape}")
    print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
    print(f"Distribution:\n{df['diagnosis'].value_counts()}")
    
    # features et cible
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # separation train/test
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y 
    )
    
    # normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain: {X_train.shape[0]} échantillons")
    print(f"Test: {X_test.shape[0]} échantillons")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
