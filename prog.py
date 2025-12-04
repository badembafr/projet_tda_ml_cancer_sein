import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(filepath):
    # préparation des données 
    df = pd.read_csv(filepath) 
    
    print("\nChargement des données") 
    print(f"Dimensions: {df.shape}")
    print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
    print(f"Distribution:\n{df['diagnosis'].value_counts()}")
    
    # features et cible
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})  # M = Maligne B Bénigne
    
    # separation train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=42, stratify=y 
    )
    
    # normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain: {X_train.shape[0]} échantillons")
    print(f"Test: {X_test.shape[0]} échantillons")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_random_forest(X_train, y_train):
    # random Forest avec 100 arbres
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    # svm avec kernel RBF
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    # KNN avec 5 voisins
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    # evaluation modèle
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


def plot_confusion_matrix(y_test, y_pred, model_name, ax):
    # matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Bénigne (B)', 'Maligne (M)'],
                yticklabels=['Bénigne (B)', 'Maligne (M)'])
    ax.set_title(f'Matrice de Confusion : {model_name}', fontweight='bold')
    ax.set_ylabel('Vrai Label')
    ax.set_xlabel('Prédiction')


def plot_roc_curves(models_results, y_test):
    # courbes ROC
    plt.figure(figsize=(10, 8))
    
    # ligne de hasard 
    plt.plot([0, 1], [0, 1], 'k--', label='Hasard (AUC=0.5)', linewidth=2) 
    
    colors = ["blue", 'green', 'orange'] 
    
    for result, color in zip(models_results, colors):
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr) 
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{result['name']} (AUC={roc_auc:.3f})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs', fontsize=12, fontweight='bold')
    plt.ylabel('Taux de Vrais Positifs (Rappel)', fontsize=12, fontweight='bold')
    plt.title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10) 
    plt.grid(alpha=0.3) 
    
    return plt.gcf()


def plot_metrics_comparison(results_df):
    # comparaison des métriques 
    fig, ax = plt.subplots(figsize=(12, 6))  
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(results_df))   
    width = 0.2
    
    colors = ['blue', 'green', 'orange', 'purple']  
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width, results_df[metric], width,  
               label=metric.replace('_', ' ').title(), color=color) 
    
    ax.set_xlabel('Modèles', fontweight='bold') 
    ax.set_ylabel('Score', fontweight='bold')  
    ax.set_title('Comparaison des Métriques', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)  
    ax.set_xticklabels(results_df.index) 
    ax.set_ylim([0.9, 1.0])  
    ax.legend() 
    ax.grid(axis='y', alpha=0.3)
    
    return fig 


def main():
    # charger dataset 
    X_train, X_test, y_train, y_test = load_and_prepare_data('breast_cancer_data.csv')
    
    print("\nEntraînement des modèles")
    
    # entrainement des models
    models = {
        'Random Forest': train_random_forest(X_train, y_train), 
        'SVM': train_svm(X_train, y_train),
        'KNN': train_knn(X_train, y_train) 
    }
    
    # evaluation des modeles
    results = []
    for name, model in models.items():
        print(f"\nÉvaluation: {name}") 
        metrics = evaluate_model(model, X_test, y_test, name) 
        results.append(metrics)
        print(f"Accuracy: {metrics['accuracy']:.4f}") 
    
    # dataframe des résultats
    results_df = pd.DataFrame([
        {
            'Model': r['name'],
            'accuracy': r['accuracy'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1_score': r['f1_score'],
            'roc_auc': r['roc_auc']
        }
        for r in results
    ]).set_index('Model') 
    
    print("\nRésultats détaillés")
    print(results_df.round(4))
    
    # affichage meilleur modele 
    best_model_name = results_df['accuracy'].idxmax()
    print(f"\nMEILLEUR MODÈLE: {best_model_name}") 
    print(f"Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
    
    # rapport du meilleur modèle
    best_result = [r for r in results if r['name'] == best_model_name][0]
    print(f"\nRapport- {best_model_name}") 
    print(classification_report(y_test, best_result['y_pred'],
                              target_names=['Bénigne (B)', 'Maligne (M)']))
    
    # visualisations
    fig = plt.figure(figsize=(15, 10)) 
    
    # matrices de confusion
    for idx, result in enumerate(results, 1): 
        ax = plt.subplot(2, 3, idx)
        plot_confusion_matrix(y_test, result['y_pred'], result['name'], ax)
    
    # comparaison des métriques
    ax = plt.subplot(2, 3, 4)
    metrics_data = results_df[['accuracy', 'precision', 'recall', 'f1_score']] 
    metrics_data.plot(kind='bar', ax=ax, rot=0)
    ax.set_title('Comparaison des Métriques', fontweight='bold') 
    ax.set_ylabel('Score')
    ax.set_ylim([0.9, 1.0]) 
    ax.legend(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.grid(axis='y', alpha=0.3) 
    
    plt.tight_layout()
    plt.savefig('comparaisons_models.png', dpi=300, bbox_inches='tight') 
    print("\nGraphique sauvegardé: comparaisons_models.png") 
    
    # courbes ROC
    roc_fig = plot_roc_curves(results, y_test) 
    roc_fig.savefig('roc_curves.png', dpi=300, bbox_inches='tight') 
    print("Courbes ROC sauvegardées: roc_curves.png") 
    
    plt.close('all')

if __name__ == "__main__":
    main()