import pandas as pd
from pathlib import Path
import os
import errno
import joblib
import scipy.stats as st
from collections import Counter
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import namedtuple
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from nonconformist.base import ClassifierAdapter
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc

import shap

DATA_PATH = '../data'
RESUlT_PATH = '../result'

"""
models, params
"""
models = [
    (LogisticRegression(C=2, random_state=1234), 'lr'),
    (ensemble.RandomForestClassifier(n_estimators=100, random_state=1234), 'rf'),
    (XGBClassifier(learning_rate=0.005,
                   objective='binary:logistic',
                   # tree_method='gpu_hist', nthread=-1
                   min_child_weight=5,
                   gamma=2,
                   subsample=0.6,
                   colsample_bytree=0.4,
                   max_depth=6,
                   random_state=1234, eval_metric="auc"), 'xgb' ),

    (LGBMClassifier(max_bin=512,
                    learning_rate=0.005,
                    min_child_weight=5,
                    boosting_type="gbdt",
                    max_depth=6,
                    min_data_in_leaf=100,
                    num_leaves=40,
                    bagging_fraction=0.7,
                    subsample=0.6,
                    colsample_bytree=0.4,
                    lambda_l1=1,
                    lambda_l2=6,
                    n_estimators=5000,
                    min_data_per_group=10,
                    objective='binary',
                    # boost_from_average=True,
                    n_jobs=-1, random_state=1234), 'lightGBM')

    # (ensemble.GradientBoostingClassifier(random_state=1234), 'gbt')
]


def cal_CI(data, confidence=0.95):
    return st.t.interval(alpha=confidence, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))


def get_X_y(df, return_sample_unique_IDs=False):
    if return_sample_unique_IDs is True:
        X = df.drop(['site_deid', 'subject_deid', 'followup_time', 'interstage_mortality'], axis=1).values
        y = df['interstage_mortality'].values
        uqniue_ids = df['subject_deid'].ravel()
        return X, y, uqniue_ids
    else:
        X = df.drop(['site_deid', 'subject_deid', 'followup_time', 'interstage_mortality'], axis=1).values
        y = df['interstage_mortality'].values
    return X, y


def impute_transformer(imputation_method='median'):
    transformer = FeatureUnion(
        transformer_list=[
            ('features', SimpleImputer(strategy=imputation_method))])
    return transformer


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise OSError


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


def find_best_threthold(y_labels, y_scores):
    """
    # find the best decision threshold, using J max (Youden Index)
    :param y_labels:
    :param y_scores:
    :return:
    """
    fpr, tpr, thresholds = roc_curve(y_labels, y_scores)
    sensitivity = tpr
    specificity = 1 - fpr
    max_value = sensitivity + specificity - 1
    best_thresh = thresholds[np.argmax(max_value)]
    return best_thresh


def compute_metrics(y_test, y_pred=None, y_score=None):
    """

    :param y_test: true label
    :param y_preds: predicted label
    :param y_scores: predicted probabilities or score
    :return:
    """
    Metric = namedtuple("metrics", ["auroc", "avg_pre", "acc", "ppv", "npv", "sens", "spes", "balanced_acc"])
    auroc = avg_pre = acc = ppv = npv = sens = spes = balanced_acc = 0

    if y_score is not None:
        auroc = roc_auc_score(y_test, y_score)
        avg_pre = average_precision_score(y_test, y_score)
    if y_pred is not None:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        sens = tp / (tp + fn)
        spes = tn / (fp + tn)
        acc = (tn + tp) / (tn + fp + fn + tp)
        balanced_acc = (sens + spes) / 2

    return Metric(auroc, avg_pre, acc, ppv, npv, sens, spes, balanced_acc)


def f(x, model):
    return shap.links.identity(model.predict_proba(x, validate_features=False)[:, 1])


def model_pipeline_cv(X, y, model,  model_name, output_dir, n_splits=5,
                      scaler=None, imputer=None,
                      feature_reducer=None, calibrated=False, conformal=False,
                      auto_decision_thres_cut_off=False):
    print("\n************************* model %s **************************\n" % model_name)
    final_auroc_result = []
    final_ap_result = []
    final_acc_result = []
    final_ppv_result = []
    final_npv_result = []
    final_sens_result = []
    final_spes_result = []
    final_balanced_acc_result = []

    final_best_threshold_list_val = []

    random_shuffle = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
    splits = 0

    for train_val_idx, test_idx in random_shuffle.split(X, y):
        print("\n************************* splits %d **************************\n" % splits)
        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        np.save(output_dir / 'X_test_on_test_split_{}.npy'.format(splits), X_test)

        auroc_result = []
        avg_pre_result = []
        acc_result = []
        ppv_result = []
        npv_result = []
        sens_result = []
        spes_result = []
        balanced_acc_result = []
        best_threshold_list_val = []

        # cross validation
        cv = KFold(n_splits=5, random_state=1024)
        cv.get_n_splits(X)

        for fold, (train_index, val_index) in enumerate(cv.split(X_train_val, y_train_val)):
            print("\n************************* CV Fold %d **************************\n" % fold)
            X_train, y_train = X_train_val[train_index], y_train_val[train_index]
            X_val, y_val = X_train_val[val_index], y_train_val[val_index]
            np.save(output_dir / 'X_train_on_train_split_{}_run_fold_{}.npy'.format(splits, fold), X_train)

            print("X_train", X_train.shape)
            print("X_val", X_val.shape)
            print("X_test", X_test.shape)

            # fit the model
            if model_name == 'xgb':
                clf = model
                clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100)

            if model_name == 'lightGBM':
                categorical_feature_index = [142, 146, 147, 149, 156, 159]
                # categorical_feature_index = [141, 145, 146, 148, 155, 158]
                clf = model
                clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100,
                        categorical_feature=categorical_feature_index)  # use the validation to early stop

            else:
                estimators = make_pipeline_estimators(model, model_name, scaler, imputer, feature_reducer)
                clf = Pipeline(estimators)
                clf.fit(X_train, y_train)

            y_tr_scores = clf.predict_proba(X_train)[:, 1]
            tr_metrics = compute_metrics(y_train, None, y_tr_scores)
            print(
                "Training AUROC: %.4f" % (
                    tr_metrics.auroc ))

            y_scores = clf.predict_proba(X_val)[:, 1]

            if auto_decision_thres_cut_off is False:
                thres = 0.5
            else:
                thres = find_best_threthold(y_val, y_scores)

            best_threshold_list_val.append(thres)

            y_preds = (y_scores >= thres).astype(bool)
            val_metrics = compute_metrics(y_val, y_preds, y_scores)


            print(
                "Validation AUROC: %.4f, Average Precision: %.4f, ACC: %.4f, PPV: %.4f, NPV: %.4f, Sensitivity: %.4f, Specificity: %.4f, Balanced ACC: %.4f, Best threshold %.4f" % (
                    val_metrics.auroc, val_metrics.avg_pre, val_metrics.acc, val_metrics.ppv, val_metrics.npv,
                    val_metrics.sens, val_metrics.spes, val_metrics.balanced_acc, thres))

            # test the current model
            # test before calibration
            y_before_scores = clf.predict_proba(X_test)[:, 1]
            y_before_preds = (y_before_scores >= thres).astype(bool)
            test_metrics = compute_metrics(y_test, y_before_preds, y_before_scores)

            print(
                "Before Calibration:   Test AUROC: %.4f, PRAUC: %.4f, ACC: %.4f, PPV: %.4f, NPV: %.4f, Sensitivity: %.4f, Specificity: %.4f, balanced acc: %.4f " % (
                    test_metrics.auroc, test_metrics.avg_pre, test_metrics.acc, test_metrics.ppv, test_metrics.npv,
                    test_metrics.sens, test_metrics.spes, test_metrics.balanced_acc))

            auroc_result.append(test_metrics.auroc)
            avg_pre_result.append(test_metrics.avg_pre)
            acc_result.append(test_metrics.acc)
            ppv_result.append(test_metrics.ppv)
            npv_result.append(test_metrics.npv)
            sens_result.append(test_metrics.sens)
            spes_result.append(test_metrics.spes)
            balanced_acc_result.append(test_metrics.balanced_acc)

            if model_name == 'lightGBM':
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_train)
                np.save(output_dir / '{}_shap_values_on_train_split_{}_run_fold_{}.npy'.format(model_name,
                                                                                              splits,
                                                                                              fold), shap_values)
            ##### test after calibration
            ################### need to recompute the threshold on validation set####
            if calibrated:
                calibrator = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
                calibrator.fit(X_val, y_val)
                y_scores = calibrator.predict_proba(X_val)[:, 1]
                if auto_decision_thres_cut_off is False:
                    thres = 0.5
                else:
                    thres = find_best_threthold(y_val, y_scores)

                y_scores_after = calibrator.predict_proba(X_test)[:, 1]
                y_preds_after = (y_scores_after >= thres).astype(bool)

                test_metrics = compute_metrics(y_test, y_preds_after, y_scores_after)

                print(
                    "After Calibration:   Test AUROC: %.4f, PRAUC: %.4f, ACC: %.4f, PPV: %.4f, NPV: %.4f, Sensitivity: %.4f, Specificity: %.4f, Balanced ACC: %.4f" % (
                        test_metrics.auroc, test_metrics.avg_pre, test_metrics.acc, test_metrics.ppv, test_metrics.npv,
                        test_metrics.sens, test_metrics.spes, test_metrics.balanced_acc))

                predictions = pd.DataFrame({'y_true': y_test,
                                            'y_pred': y_preds_after,
                                            'y_score': y_scores_after
                                            })
                predictions.to_csv(
                    output_dir / '{}_calibrated_prediction_on_test_split_{}_run_fold_{}.csv'.format(model_name,
                                                                                                    splits,
                                                                                                    fold),
                    index=False)
                joblib.dump(calibrator,
                            output_dir / '{}_calibrated_split_{}_run_fold_{}.pkl'.format(model_name,
                                                                                         splits, fold))

        result_dict = {'AUROC': auroc_result, 'PRAUC': avg_pre_result, 'ACC': acc_result,
                       'PPV': ppv_result, 'NPV': npv_result, 'Sensitivity': sens_result, 'Specificity': spes_result,
                       'Balanced_ACC': balanced_acc_result, 'Threshold': best_threshold_list_val}

        np.save(output_dir / '{}_summary_result_split_{}.npy'.format(model_name, splits), result_dict)
        splits += 1

        final_auroc_result.extend(auroc_result)
        final_ap_result.extend(avg_pre_result)
        final_acc_result.extend(acc_result)
        final_ppv_result.extend(ppv_result)
        final_npv_result.extend(npv_result)
        final_sens_result.extend(sens_result)
        final_spes_result.extend(spes_result)
        final_balanced_acc_result.extend(balanced_acc_result)
        final_best_threshold_list_val.extend(best_threshold_list_val)

    return final_auroc_result, final_ap_result, final_acc_result, final_ppv_result, final_npv_result, \
           final_sens_result, final_spes_result, final_balanced_acc_result, final_best_threshold_list_val


def run_benchmark():
    df = pd.read_csv(
        Path(DATA_PATH) / 'processed' / 'merged_data_removed_high_missing_features.csv') #'merged_data_removed_high_missing_features.csv'
    X, y = get_X_y(df)
    print(Counter(y))

    results_dir = Path(RESUlT_PATH) / "benchmark"
    mkdir(results_dir)
    result = {}

    for model_class, model_name in models:
        final_auroc_result, final_ap_result, final_acc_result, final_ppv_result, \
        final_npv_result, final_sens_result, final_spes_result, final_balanced_acc_result, final_best_threshold_list_val = model_pipeline_cv(
            X, y,
            model_class, model_name, results_dir,
            n_splits=N_SPLITS,
            scaler=StandardScaler(), imputer=impute_transformer(),
            feature_reducer=None,
            calibrated=True,
            auto_decision_thres_cut_off=True)

        print(len(final_auroc_result))
        print(len(final_balanced_acc_result))

        # create 95% confidence interval for population mean weight
        result[model_name] = ['%.3f (%.3f-%.3f)' % (np.mean(final_auroc_result),
                                                    cal_CI(final_auroc_result)[0],
                                                    cal_CI(final_auroc_result)[1]),
                              '%.3f (%.3f-%.3f)' % (np.mean(final_ap_result),
                                                    cal_CI(final_ap_result)[0],
                                                    cal_CI(final_ap_result)[1]),
                              '%.3f (%.3f-%.3f)' % (np.mean(final_acc_result),
                                                    cal_CI(final_acc_result)[0],
                                                    cal_CI(final_acc_result)[1]),
                              '%.3f (%.3f-%.3f)' % (np.mean(final_balanced_acc_result),
                                                    cal_CI(final_balanced_acc_result)[0],
                                                    cal_CI(final_balanced_acc_result)[1]),
                              '%.3f (%.3f-%.3f)' % (np.mean(final_sens_result),
                                                    cal_CI(final_sens_result)[0],
                                                    cal_CI(final_sens_result)[1]),
                              '%.3f (%.3f-%.3f)' % (np.mean(final_spes_result),
                                                    cal_CI(final_spes_result)[0],
                                                    cal_CI(final_spes_result)[1]),
                              '%.3f (%.3f-%.3f)' % (np.mean(final_ppv_result),
                                                    cal_CI(final_ppv_result)[0],
                                                    cal_CI(final_ppv_result)[1]),
                              '%.3f (%.3f-%.3f)' % (np.mean(final_npv_result),
                                                    cal_CI(final_npv_result)[0],
                                                    cal_CI(final_npv_result)[1]),
                              '%.3f (%.3f-%.3f)' % (
                                  np.mean(final_best_threshold_list_val),
                                  cal_CI(final_best_threshold_list_val)[0],
                                  cal_CI(final_best_threshold_list_val)[1])]

    df_result = pd.DataFrame.from_dict(result, orient='index',
                                       columns=['AUROC', 'AUPRC', 'ACC', 'Balanced_ACC',  'Sensitivity', 'Specificity','PPV', 'NPV',
                                                'Best_threshold'])

    df_result.to_csv(results_dir / 'benchmark_summary_result.csv')


def run_conformal_prediction():
    df = pd.read_csv(
        Path(DATA_PATH) / 'processed' / 'merged_data_removed_high_missing_features.csv')
    X, y, sample_IDs = get_X_y(df, return_sample_unique_IDs=True)
    print(Counter(y))

    results_dir = Path(RESUlT_PATH) / "conformal_vs_non_conformal"
    mkdir(results_dir)

    random_shuffle = ShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=0)
    splits = 0
    for model_class, model_name in models:
        for train_val_idx, test_idx in random_shuffle.split(X, y):
            print("\n************************* splits %d **************************\n" % splits)
            X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_test_subjectIDs = sample_IDs[test_idx]

            # cross validation
            cv = KFold(n_splits=5, random_state=1024)
            cv.get_n_splits(X)

            for fold, (train_index, val_index) in enumerate(cv.split(X_train_val, y_train_val)):
                print("\n************************* CV Fold %d **************************\n" % fold)
                X_train, y_train = X_train_val[train_index], y_train_val[train_index]
                X_val, y_val = X_train_val[val_index], y_train_val[val_index]

                print("X_train", X_train.shape)
                print("X_val", X_val.shape)
                print("X_test", X_test.shape)

                estimators = make_pipeline_estimators(model_class, model_name, scaler=StandardScaler(),
                                                      imputer=impute_transformer(),
                                                      feature_reducer=None)

                ## train without conformal prediction
                clf = Pipeline(estimators)
                clf.fit(X_train, y_train)
                calibrator = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
                calibrator.fit(X_val, y_val)
                y_scores = calibrator.predict_proba(X_val)[:, 1]

                thres = find_best_threthold(y_val, y_scores)

                y_scores_after = calibrator.predict_proba(X_test)[:, 1]
                y_preds_after = (y_scores_after >= thres).astype(bool)

                test_metrics = compute_metrics(y_test, y_preds_after, y_scores_after)

                print(
                    "After Calibration:   Test AUROC: %.4f, PRAUC: %.4f, ACC: %.4f, PPV: %.4f, NPV: %.4f, Sensitivity: %.4f, Specificity: %.4f, Balanced ACC: %.4f" % (
                        test_metrics.auroc, test_metrics.avg_pre, test_metrics.acc, test_metrics.ppv, test_metrics.npv,
                        test_metrics.sens, test_metrics.spes, test_metrics.balanced_acc))

                predictions = pd.DataFrame({'Subject_id': X_test_subjectIDs,
                                            'True_label': y_test,
                                            'Pred_label': y_preds_after,
                                            'Pred_score': y_scores_after
                                            })
                predictions.to_csv(
                    results_dir / '{}_calibrated_prediction_on_test_split_{}_run_fold_{}.csv'.format(model_name,
                                                                                                     splits,
                                                                                                     fold), index=False)

                ## train use conformal prediction
                clf_for_conformal = Pipeline(estimators)
                model_adapter = ClassifierAdapter(clf_for_conformal)
                nc = ClassifierNc(model_adapter, MarginErrFunc())
                icp = IcpClassifier(nc, condition=lambda x: x[1])
                icp.fit(X_train, y_train)
                icp.calibrate(X_val, y_val)

                predict_result = pd.DataFrame(icp.predict_conf(X_test),
                                              columns=['Pred_label', 'Confidence', 'Credibility'])
                predict_result['Subject_id'] = X_test_subjectIDs
                predict_result['True_label'] = y_test
                metrics = compute_metrics(y_test, predict_result['Pred_label'].ravel())
                print(
                    "Conformal prediction: ACC: %.4f, PPV: %.4f, NPV: %.4f, Sensitivity: %.4f, Specificity: %.4f, balanced acc: %.4f " % (
                        metrics.acc, metrics.ppv, metrics.npv, metrics.sens, metrics.spes, metrics.balanced_acc))
                predict_result.to_csv(
                    results_dir / '{}_conformal_prediction_final_on_test_split_{}_run_fold_{}.csv'.format(model_name,
                                                                                                          splits,
                                                                                                          fold),
                    index=False)
                prediction = icp.predict(X_test, significance=0.05)

                predict_result = pd.DataFrame(prediction, columns=['Class0', 'Class1'])
                predict_result['Subject_id'] = X_test_subjectIDs
                predict_result['True_label'] = y_test

                predict_result.to_csv(
                    results_dir / '{}_conformal_prediction_on_test_split_{}_run_fold_{}.csv'.format(model_name,
                                                                                                    splits,
                                                                                                    fold), index=False)
            splits += 1


if __name__ == '__main__':
    N_SPLITS = 5
    # run_benchmark()
    run_conformal_prediction()
