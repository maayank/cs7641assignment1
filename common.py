import pandas as pd
import numpy as np
from load_datasets import *
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class StubLearner:
    '''
    Abstract class representing a learner
    '''
    def fit(self, X, y):
        pass

    def predict(self, X):
        return [i % 2 for i in range(len(df))]

class Experiment:
    '''
    Responsible for preprocessing (e.g. dividing data to sets) and call learners
    '''
    def __init__(self, df: pd.DataFrame, learner):
        np.random.seed(42)
        self.training_df, self.test_df = train_test_split(df, train_size=.8, shuffle=True)
        self.learner = learner

    @staticmethod
    def _split(df):
        return df.iloc[:, :-1], df.iloc[:, -1]

    def _eval(self, df):
        test_features, test_y = self._split(df)
        start = perf_counter()
        pred_y = self.learner.predict(test_features)
        took = perf_counter() - start
 #       print(f'Predicting took {took} seconds.')

        return tuple(func(test_y, pred_y) for func in [accuracy_score, f1_score, precision_score, recall_score])


    def do(self):
        start = perf_counter()
        self.learner.fit(*self._split(self.training_df))
        took = perf_counter() - start
#        print(f'Fitting took {took} seconds.')


#        print('Predicting on training_df')
#        print(self._eval(self.training_df))
#        print('----------------')

#        print('Predicting on test_df')
        res = self._eval(self.test_df)
#        print(res)
        return res

if __name__=='__main__':
    ex = Experiment(load_cancer(), StubLearner())
    print(ex.training_df)
    print(ex.test_df)
    print(ex.do())
