import pandas as pd
import numpy as np
from load_datasets import *
from time import perf_counter
from sklearn.model_selection import train_test_split

class Learner:
    '''
    Abstract class representing a learner
    '''
    def fit(self, df):
        pass

    def predict(self, df):
        return [0] * len(df)

class Experiment:
    '''
    Responsible for preprocessing (e.g. dividing data to sets) and call learners
    '''
    def __init__(self, df: pd.DataFrame, learner: Learner):
        np.random.seed(42)
        self.df = df.sample(frac=1).reset_index(drop=True)
        cutoff_idx = int(.8 * len(self.df))
        self.training_df = self.df.iloc[:cutoff_idx]
        self.test_df = self.df.iloc[cutoff_idx:]
        self.learner = learner

    def do(self):
        start = perf_counter()
        self.learner.fit(self.training_df)
        took = perf_counter() - start
        print(f'Fitting took {took} seconds.')

        test_features = self.test_df.iloc[:, :-1]
        test_y = self.test_df.iloc[:, -1]
        start = perf_counter()
        pred_y = self.learner.predict(test_features)
        took = perf_counter() - start
        print(f'Predicting took {took} seconds.')


if __name__=='__main__':
    ex = Experiment(load_cancer(), Learner())
    print(ex.training_df)
    print(ex.test_df)

