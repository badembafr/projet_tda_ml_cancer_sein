import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


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
    # comparaison des metriques
    import numpy as np
    
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
