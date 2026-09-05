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
        """Fits B bootstrap models on resampled training sets."""
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
        Computes mean risk, lower bound, upper bound, and interval width.

        Returns:
            (mean_risk, lower_bound, upper_bound, interval_width)
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
        Computes granular uncertainty metrics for an individual student row.
        Includes missing-data penalty for transfer students.
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
