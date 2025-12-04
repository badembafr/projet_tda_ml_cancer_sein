import sys
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

sys.path.append('src')
from dataset import load_and_prepare_data
from modeling import train_random_forest, train_svm, train_knn, evaluate_model, save_model
from plots import plot_confusion_matrix, plot_roc_curves
from config import MODELS_PATH, FIGURES_PATH


def main():
    # charger dataset 
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
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
    
    # dataframe des resultats
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
    
    # sauvegarde du meilleur modele
    best_model = models[best_model_name]
    save_model(best_model, f"{MODELS_PATH}{best_model_name.replace(' ', '_').lower()}_best.pkl")
    
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
    
    # comparaison des metriques
    ax = plt.subplot(2, 3, 4)
    metrics_data = results_df[['accuracy', 'precision', 'recall', 'f1_score']] 
    metrics_data.plot(kind='bar', ax=ax, rot=0)
    ax.set_title('Comparaison des Métriques', fontweight='bold') 
    ax.set_ylabel('Score')
    ax.set_ylim([0.9, 1.0]) 
    ax.legend(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.grid(axis='y', alpha=0.3) 
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_PATH}comparaisons_models.png', dpi=300, bbox_inches='tight') 
    print(f"\nGraphique sauvegardé: {FIGURES_PATH}comparaisons_models.png") 
    
    # courbes ROC
    roc_fig = plot_roc_curves(results, y_test) 
    roc_fig.savefig(f'{FIGURES_PATH}roc_curves.png', dpi=300, bbox_inches='tight') 
    print(f"Courbes ROC sauvegardées: {FIGURES_PATH}roc_curves.png") 
    
    plt.close('all')


if __name__ == "__main__":
    main()
