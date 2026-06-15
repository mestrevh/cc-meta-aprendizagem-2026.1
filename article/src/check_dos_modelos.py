import os
import warnings
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import wilcoxon, friedmanchisquare

warnings.filterwarnings("ignore")



meta_models = {
    "all": {
        "dataset": "data/metafeatures_dataset_with_best.csv",
        "svm": make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=42)),
    },
    "me1": {
        "dataset": "data/metafeatures_selected_with_best_subset_selection_filter_milp.csv",
        "train": DecisionTreeClassifier(random_state=42),
        },
    "me2": {
        "dataset": "data/metafeatures_selected_with_knapsack_problem_ilp.csv",
        "train": DecisionTreeClassifier(random_state=42),
    },
    "me3": {
        "dataset": "data/metafeatures_selected_with_correlation_penalized_by_milp.csv",
        "train": DecisionTreeClassifier(random_state=42),
    },
    "ma1": {
        "dataset": "data/metafeatures_selected_with_ag_1_classico_binario.csv",
        "train": ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    },
    "ma2":
    {
        "dataset": "data/metafeatures_selected_with_ag_2_guiado_por_mi_busca_local.csv",
        "train": ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    },
    "ma3": {
        "dataset": "data/metafeatures_selected_with_ag_3_compacto_esparso.csv",
        "train": ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    },
    "ah1": {
        "dataset": "data/metafeatures_selected_with_hybrid_ag_1_classico_binario.csv",
        "train": ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    },
    "ah2": {
        "dataset": "data/metafeatures_selected_with_hybrid_ag_2_guiado_por_mi_busca_local.csv",
        "train": ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    },
    "ah3": {
        "dataset": "data/metafeatures_selected_with_hybrid_ag_3_compacto_esparso.csv",
        "train": ExtraTreesClassifier(
            n_estimators=180,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    },
}
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import pandas as pd

def execute_meta_model(data, model, key, global_dataset):
    """
    Se for me[numero] e all faça com leaveoneout
    se não faça com StratifiedKFold com k igual 5
    return: acuracia e a predição com todos os valores igual no metodo exato
    """
    summary_of_predictions = {
        'Dataset':[],
        'Best clf (true)':[],
        'Perf of best clf (true)':[],
        'Best clf (pred)':[],
        'Perf of best clf (pred)':[]
    }
    
    X = data.drop(columns=['Dataset', 'Best'])
    y = data['Best']
    X = X.fillna(0)
    
    y_pred_list = []
    y_true_list = []
    
    if key.startswith('me') or key == 'all':
        cv = LeaveOneOut()
    else:
        class_counts = y.value_counts()
        min_class = class_counts.min()
        k = min(5, min_class)
        if k < 2:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf = clone(model)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        for i, pred in enumerate(preds):
            idx = test_index[i]
            dataset_name = data.iloc[idx]['Dataset']
            true_best = y_test.iloc[i]
            
            perf_true = global_dataset.loc[global_dataset['Dataset'] == dataset_name, true_best].values[0]
            perf_pred = global_dataset.loc[global_dataset['Dataset'] == dataset_name, pred].values[0]
            
            summary_of_predictions['Dataset'].append(dataset_name)
            summary_of_predictions['Best clf (true)'].append(true_best)
            summary_of_predictions['Perf of best clf (true)'].append(perf_true)
            summary_of_predictions['Best clf (pred)'].append(pred)
            summary_of_predictions['Perf of best clf (pred)'].append(perf_pred)
            
            y_pred_list.append(pred)
            y_true_list.append(true_best)
            
    summary_df = pd.DataFrame(summary_of_predictions)
    acc = accuracy_score(y_true_list, y_pred_list)
    
    return acc, summary_df

def acc_base_cal(dataset, predics):
    # leia metafeatures_dataset_with_best.csv
    """return: acuracia do nivel base"""
    return predics['Perf of best clf (pred)'].mean()

dataset = pd.read_csv("data/metafeatures_dataset_with_best.csv")

for key, meta_model in meta_models.items():
    data = pd.read_csv(meta_model["dataset"], index_col=False)
    acc_meta, table = execute_meta_model(data=data, model=meta_model["train"], key=key, global_dataset=dataset)
    meta_models[key]["acc_meta"] = acc_meta
    meta_models[key]["acc_base"] = acc_base_cal(dataset=dataset, predics=table)
    print(f"[{key.upper()}] Acc Meta: {acc_meta:.4f} | Acc Base: {meta_models[key]['acc_base']:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

labels = list(meta_models.keys())
acc_meta_values = [meta_models[k]['acc_meta'] for k in labels]

plt.figure(figsize=(12, 6))
sns.barplot(x=labels, y=acc_meta_values, palette="viridis")
plt.title("Comparação da Acurácia no Nível Meta por Abordagem")
plt.xlabel("Abordagens")
plt.ylabel("Acurácia (Meta)")
plt.ylim(0, 1.0)
for i, v in enumerate(acc_meta_values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
plt.show()

acc_base_values = [meta_models[k]['acc_base'] for k in labels]

plt.figure(figsize=(12, 6))
sns.barplot(x=labels, y=acc_base_values, palette="mako")
plt.title("Comparação da Acurácia no Nível Base por Abordagem")
plt.xlabel("Abordagens")
plt.ylabel("Acurácia Média (Base)")
plt.ylim(0.5, 1.0) # Ajustado para melhor visualização (geralmente não cai abaixo de 0.5)
for i, v in enumerate(acc_base_values):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=10)
plt.show()
