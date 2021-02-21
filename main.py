from load_datasets import *
from common import Experiment

import sklearn.tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

import numpy as np

if __name__=='__main__':
    np.random.seed(42)

    experiments = [
                {
                    'estimator': DecisionTreeClassifier(),
                    'interesting_parameters': {
                        'max_depth': np.arange(3,10),
                        'min_samples_leaf': list(np.linspace(.001, .01, 5)),
                        'min_samples_split': list(np.linspace(.001, .5, 5)),
                        }
                },
                {
                    'estimator': AdaBoostClassifier(DecisionTreeClassifier()),
                    'interesting_parameters': {
                        'base_estimator__max_depth': np.arange(1,4),
                        'base_estimator__min_samples_leaf': np.linspace(.002, .01, 5),
                        'n_estimators': np.arange(20, 80, 10),
                        'learning_rate': np.linspace(.01, 1, 5),
                    }
                },
                {
                    'estimator': KNeighborsClassifier(n_jobs=-1),
                    'interesting_parameters': {
                        'n_neighbors': np.arange(3, 8),
                        'weights': ['uniform', 'distance'],
                    }
                },
                {
                    'estimator': SVC(),
                    'interesting_parameters': {
                        'C': [.1, 1, 10, 100],
                        'kernel': ['linear', 'rbf']
                    }
                },
                {
                    'estimator': MLPClassifier(learning_rate_init=.001, max_iter=10000),
                    'interesting_parameters': {
                        'hidden_layer_sizes': [(10, ), (10, 10), (10, 10, 10), (100, ), (100, 100)]
                    }
                }
            ]

    data = {
           "Cancer": load_cancer(),
           "Wine": load_wine(),
            }

    results = []
    for d_name in data:
        df = data[d_name]
        for ex_params in experiments:
            for scale in [0, 1]:
                for grid_search in [0, 1]:
                    ex = Experiment(scale, grid_search, **ex_params)
                    ex.load(d_name, df)
                    accuracy, f1, precision, recall = ex.do()
                    results.append({
                        'Dataset': d_name,
                        'Classifier': ex.estimator_name,
                        'Scale': scale,
                        'GridSearch': grid_search,
                        'Accuracy': accuracy,
                        'F1': f1,
                        'Precision': precision,
                        'Recall': recall
                    })

    rdf = pd.DataFrame(results)
    print(rdf)
    rdf.to_csv('results.csv')