from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)
from config import RANDOM_STATE, N_ESTIMATORS_RF, KNN_NEIGHBORS
import joblib


def train_random_forest(X_train, y_train):
    # random forest avec 100 arbres
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    # svm avec kernel RBF
    model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    # KNN avec 5 voisins
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    # evaluation modele
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def save_model(model, filepath):
    # sauvegarde du modele
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé: {filepath}")


def load_model(filepath):
    # chargement du modele
    return joblib.load(filepath)
