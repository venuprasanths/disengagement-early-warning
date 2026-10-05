"""
Uncertainty Estimation Engine for Disengagement Early-Warning System.

Quantifies prediction confidence and uncertainty intervals via:
1. Bootstrapped ensemble resampling (epistemic uncertainty: model disagreement).
2. Binary entropy calculation (aleatoric uncertainty: inherent signal ambiguity).
3. Sparse-data penalty adjustment (e.g. transfer students with few observation weeks).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.ensemble import HistGradientBoostingClassifier
from src.schema import assert_no_ground_truth_leakage
from src.features import ENGINEERED_FEATURE_NAMES


class BootstrappedUncertaintyEstimator:
    """
    Bootstrapped ensemble estimator that outputs:
    - Point prediction risk score
    - Lower confidence bound (e.g. 10th percentile)
    - Upper confidence bound (e.g. 90th percentile)
    - Epistemic uncertainty (interval width / standard error across bootstrap models)
    - Aleatoric uncertainty (Shannon entropy)
    """

    def __init__(
        self,
        n_bootstraps: int = 10,
        random_state: int = 42,
    ):
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        self.models: List[HistGradientBoostingClassifier] = []
        self.feature_names = ENGINEERED_FEATURE_NAMES
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BootstrappedUncertaintyEstimator":
        """
        Fits B independent bootstrap estimators on resampled training sets.

        Epistemic Uncertainty Rationale:
            Resampling training records with replacement simulates data variability
            and measures model stability. Where training data is dense and unambiguous,
            the B models converge. In sparse or atypical regions, predictions diverge,
            yielding a wider empirical confidence band.

        Parameters:
            X (pd.DataFrame): Training feature matrix.
            y (pd.Series): Binary ground-truth outcome series.

        Returns:
            BootstrappedUncertaintyEstimator: The fitted estimator instance (self).
        """
        assert_no_ground_truth_leakage(list(X.columns))
        X_clean = X[self.feature_names].fillna(0.0).to_numpy()
        y_clean = y.to_numpy()

        n_samples = len(X_clean)
        rng = np.random.default_rng(self.random_state)

        self.models = []
        for b in range(self.n_bootstraps):
            # Bootstrap resample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X_clean[indices]
            y_b = y_clean[indices]

            model = HistGradientBoostingClassifier(
                max_iter=80,
                max_leaf_nodes=12,
                min_samples_leaf=15,
                learning_rate=0.09,
                random_state=self.random_state + b,
            )
            model.fit(X_b, y_b)
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict_with_intervals(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes mean risk, lower bound, upper bound, and interval width across the bootstrap ensemble.

        Mathematical Formulation:
            Let {p_1, ..., p_B} be the predicted probabilities from B models.
            mean_risk = (1 / B) * sum(p_b)
            lower_bound = Percentile_alpha(preds), where alpha = (1 - confidence_level) / 2
            upper_bound = Percentile_{1 - alpha}(preds)
            interval_width = upper_bound - lower_bound

        Parameters:
            X (pd.DataFrame): Input feature matrix.
            confidence_level (float): Target coverage level in (0, 1), default 0.80 (80% CI).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (mean_risk, lower_bound, upper_bound, interval_width)

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before predict_with_intervals.")
        assert_no_ground_truth_leakage(list(X.columns))
        X_clean = X[self.feature_names].fillna(0.0).to_numpy()

        # Gather predictions from each bootstrap model
        preds = np.zeros((self.n_bootstraps, len(X_clean)))
        for b, model in enumerate(self.models):
            preds[b, :] = model.predict_proba(X_clean)[:, 1]

        mean_risk = np.mean(preds, axis=0)

        # Percentile interval bounds
        alpha = (1.0 - confidence_level) / 2.0
        lower_percentile = alpha * 100
        upper_percentile = (1.0 - alpha) * 100

        lower_bound = np.percentile(preds, lower_percentile, axis=0)
        upper_bound = np.percentile(preds, upper_percentile, axis=0)
        interval_width = upper_bound - lower_bound

        return mean_risk, lower_bound, upper_bound, interval_width

    def compute_instance_uncertainty(
        self,
        instance_features: pd.Series,
        weeks_available: int = 16,
    ) -> Dict[str, Any]:
        """
        Computes granular uncertainty metrics for an individual student record.

        Implements Transfer-Student Epistemic Guardrail:
            When a student has fewer than 6 weeks of observed history, an artificial
            epistemic variance penalty expands the interval bounds to warn counselors
            against premature labeling:
                margin = max(interval_width / 2, 0.05 * (6 - weeks_available) + 0.10)

        Parameters:
            instance_features (pd.Series): Single student feature record.
            weeks_available (int): Cumulative weeks of observation recorded for this student.

        Returns:
            Dict[str, Any]:
                - mean_risk (float): Point prediction risk score.
                - interval_lower (float): Lower confidence bound in [0.0, 1.0].
                - interval_upper (float): Upper confidence bound in [0.0, 1.0].
                - interval_width (float): Epistemic interval width (upper - lower).
                - entropy (float): Aleatoric Shannon entropy in [0.0, 1.0].
                - confidence_label (str): 'High Confidence', 'Moderate Confidence', or 'Low Confidence'.
                - sparse_data_flag (bool): True if student has < 6 weeks of data.
        """
        # Verify feature names do not include ground truth
        assert_no_ground_truth_leakage(self.feature_names)
        # Select only model features
        x_row = instance_features[self.feature_names].to_frame().T.fillna(0.0)

        mean_r, low_b, up_b, width = self.predict_with_intervals(x_row)
        score = float(mean_r[0])
        low = float(low_b[0])
        high = float(up_b[0])
        w_val = float(width[0])

        # If data is sparse (< 6 weeks of observation), expand interval to reflect high epistemic uncertainty
        if weeks_available < 6:
            missing_weeks = 6 - weeks_available
            margin = max(w_val / 2.0, 0.05 * missing_weeks + 0.10)
            low = max(0.0, score - margin)
            high = min(1.0, score + margin)
            w_val = high - low

        # Binary Shannon entropy (aleatoric uncertainty)
        p_clipped = np.clip(score, 1e-5, 1.0 - 1e-5)
        entropy = float(-(p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped)))

        # Confidence category
        if w_val < 0.15:
            confidence_label = "High Confidence"
        elif w_val < 0.35:
            confidence_label = "Moderate Confidence"
        else:
            confidence_label = "Low Confidence (Wide Interval)"

        return {
            "mean_risk": round(score, 3),
            "interval_lower": round(low, 3),
            "interval_upper": round(high, 3),
            "interval_width": round(w_val, 3),
            "entropy": round(entropy, 3),
            "confidence_label": confidence_label,
            "sparse_data_flag": bool(weeks_available < 6),
        }
