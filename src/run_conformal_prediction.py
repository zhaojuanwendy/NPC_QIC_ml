import pandas as pd
from pathlib import Path

from collections import Counter

import numpy as np

from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.model_selection import train_test_split

from nonconformist.base import ClassifierAdapter
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc


DATA_PATH = '../data'
RESUlT_PATH = '../result'

"""
models, params
"""
models = [
    # (LogisticRegression(C=2, random_state=1234), 'lr'),
    (ensemble.GradientBoostingClassifier(random_state=1234), 'gbt')
]

def get_X_y(df):
    X = df.drop(['site_deid', 'subject_deid', 'followup_time', 'interstage_mortality'], axis=1).values
    y = df['interstage_mortality'].values
    return X, y


def impute_transformer(imputation_method='median'):
    transformer = FeatureUnion(
        transformer_list=[
            ('features', SimpleImputer(strategy=imputation_method))])
    return transformer


def make_pipeline_estimators(model: object, model_name: object, scaler: object = None, imputer: object = None,
                             feature_reducer: object = None) -> object:
    estimators = []
    if scaler is not None:
        estimators.append(('scaler', scaler))

    if imputer is not None:
        estimators.append(('imputer', imputer))

    if feature_reducer is not None:
        estimators.append(('feature_reducer', feature_reducer))

    estimators.append((model_name, model))

    return estimators


def model_pipeline_cv(X, y, model, model_name, scaler=None, imputer=None,
                      feature_reducer=None):
    print("\n************************* model %s **************************\n" % model_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1000)


    print("X_train", X_train.shape)
    print("X_val", X_val.shape)
    print("X_test", X_test.shape)

    print('y_train num',y_train.shape)
    print('y_val num',y_val.shape)
    print('y_test num', y_test.shape)

    # estimators = make_pipeline_estimators(model, model_name, None, imputer, feature_reducer)
    estimators = make_pipeline_estimators(model, 'gbt', None,  SimpleImputer(missing_values=np.nan, strategy='median'), None)
    clf = Pipeline(estimators)
    model_adapter = ClassifierAdapter(clf)
    nc = ClassifierNc(model_adapter, MarginErrFunc())
    icp = IcpClassifier(nc, condition=lambda x: x[1])
    icp.fit(X_train, y_train)
    icp.calibrate(X_val, y_val)
    icp.predict_conf(X_test)
    print(pd.DataFrame(icp.predict_conf(X_test),
                       columns=['Label', 'Confidence', 'Credibility']))
    prediction = icp.predict(X_test, significance=0.5)
    print(pd.DataFrame(prediction, columns=['Class0', 'Class1']))


def run_benchmark():
    df = pd.read_csv(
        Path(DATA_PATH) / 'processed' / 'merged_data_removed_high_missing_features.csv')
    X, y = get_X_y(df)
    print(Counter(y))

    for model_class, model_name in models:
        model_pipeline_cv(X, y, model_class, model_name, scaler=StandardScaler(), imputer=impute_transformer(),
                           feature_reducer=None)


if __name__ == '__main__':

    run_benchmark()
