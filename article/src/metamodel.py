import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import warnings

class  MetaModel:
    def __init__(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        
    def train_and_evaluate_best_metamodel(self, meta_dataset, meta_dataset_train, model):
        
        # Create a dictionary to store the reuslts:
        summary_of_predictions = {
                'Dataset':[],
                'Best clf (true)':[],
                'Perf of best clf (true)':[],
                'Best clf (pred)':[],
                'Perf of best clf (pred)':[]
            }
        
        loo = LeaveOneOut()
        
        y_pred = []

        for train_index, test_index in loo.split(meta_dataset_train):
            # Split the data into training and test sets
            X = meta_dataset_train.drop(columns=['Dataset', 'Best']) # Drop everything except meta-features
            y = meta_dataset_train['Best']
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
        
            # Train a simple classifier (e.g., Decision Tree) on the training set
            clf = clone(model)
            clf.fit(X_train, y_train)
            
            # Predict the best classifier for the test dataset
            y_pred.append(clf.predict(X_test)[0])
            
            # Store results in the summary dictionary
            summary_of_predictions['Dataset'].append(meta_dataset_train['Dataset'].iloc[test_index].values[0])
            summary_of_predictions['Best clf (true)'].append(y_test.values[0])
            summary_of_predictions['Perf of best clf (true)'].append(meta_dataset.loc[test_index, y_test.values[0]].values[0])
            summary_of_predictions['Best clf (pred)'].append(y_pred[-1])
            summary_of_predictions['Perf of best clf (pred)'].append(meta_dataset.loc[test_index, y_pred[-1]].values[0])


        # Create a DataFrame from the summary of predictions
        summary_df = pd.DataFrame(summary_of_predictions)

        return summary_df
    
    def meta_feature_scores(self, meta_dataset, model):
        
        X = meta_dataset.drop(columns=['Dataset', 'Best'])
        y = meta_dataset['Best']
        
        X = X.fillna(0)
        y = y.fillna(0)
        
        model_train = clone(model)
        model_train.fit(X, y)
        
        if hasattr(model_train, 'feature_importances_'):
            feature_scores = model_train.feature_importances_
            
        elif hasattr(model_train, 'steps') and hasattr(model_train.steps[-1][1], 'feature_importances_'):
            feature_scores = model_train.steps[-1][1].feature_importances_
        
        elif "gaussiannb" in str(model).lower():
                from sklearn.feature_selection import f_classif
                f_values, _ = f_classif(X, y)
                
                feature_scores = np.nan_to_num(f_values)
        else:
            resultado = permutation_importance(
                model_train, X, y, n_repeats=5, random_state=42, scoring='accuracy'
            )
            feature_scores = resultado.importances_mean
            
            feature_scores = np.maximum(feature_scores, 0)

        return feature_scores

# TESTE
# if __name__ == "__main__":
    
#     from sklearn.metrics import accuracy_score, f1_score
#     from sklearn.tree import DecisionTreeClassifier
    
#     dt = DecisionTreeClassifier()
#     meta_dataset = pd.read_csv('data/metafeatures_dataset_with_best.csv', index_col=0)
    
#     metamodel = MetaModel()
#     df_predicts = metamodel.train_and_evaluate_best_metamodel(meta_dataset, model=dt)
    
#     y_pred = list(df_predicts['Best clf (pred)'])
#     y_true = meta_dataset['Best'].values
    
#     metamodel_accuracy = accuracy_score(y_true, y_pred)
#     metamodel_f1 = f1_score(y_true, y_pred, average='weighted')
    
#     print("acuracia: ", metamodel_accuracy)
#     print("f1: ", metamodel_f1)

