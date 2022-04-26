import numpy as np
import bisect
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted


class InductiveConformalPredictor():
    """
    Standard Conformal Predictor with uncertainty non-conformity score.

    Args:
        predictor: classifier used in upstream task.
    """

    def __init__(self, predictor):
        self.predictor = predictor
        check_is_fitted(self.predictor, attributes=["classes_"])

        self._le = LabelEncoder()
        self.classes = self._le.fit_transform(predictor.classes_)

    def fit(self, X, y):
        self.calibration_score = self._uncertainty_conformity_score(X)
        self.calibration_class = self._le.transform(y)
        return self

    def _uncertainty_conformity_score(self, data):
        uncertainty_score = 1 - self.predictor.predict_proba(data)
        return uncertainty_score

    def predict_proba(self, X, mondrian=True):
        check_is_fitted(self, attributes=["calibration_score"])

        conformity_score = self._uncertainty_conformity_score(X)
        conformal_pred = np.zeros(conformity_score.shape)

        for c in self.classes:
            if mondrian:
                calibration_filt = self.calibration_score[self.calibration_class == c]
                calib = calibration_filt[:, c]
            else:
                calib = self.calibration_score[range(len(self.calibration_class)),
                                               self.calibration_class]

            sorted_calib = np.sort(calib)
            conformal_pred[:, c] = [float(bisect.bisect(sorted_calib, x))/len(calib)
                                    for x in conformity_score[:, c]]

        return conformal_pred

    def predict(self, X, mondrian=True, alpha=0.05):
        _conformal_proba = self.predict_proba(X=X, mondrian=mondrian)
        conformal_pred = (_conformal_proba > alpha).astype(int)

        mlb = MultiLabelBinarizer()
        mlb.fit([self._le.classes_])
        pred = mlb.inverse_transform(conformal_pred)

        return pred