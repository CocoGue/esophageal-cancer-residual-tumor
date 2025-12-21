"""
Logistic regression model based on statsmodels.

This implementation supports:
- optional intercept
- optional elastic-net regularization
"""

from typing import Optional, List

import pandas as pd
import statsmodels.api as sm
import numpy as np

class LogisticRegressionModel:
    """
    Wrapper around statsmodels GLM for binary logistic regression.
    """

    def __init__(
        self,
        add_intercept: bool = True,
        regularization: Optional[float] = None,
        max_iter: int = 300,
    ) -> None:
        """
        Initialize the logistic regression model.

        Parameters
        ----------
        add_intercept : bool, default=True
            Whether to add an intercept term.
        regularization : float or None, default=None
            Elastic-net mixing parameter.
            - None: no regularization
            - 0.0: pure L2
            - 1.0: pure L1
        max_iter : int, default=300
            Maximum number of optimization iterations.
        """
        self.add_intercept = add_intercept
        self.regularization = regularization
        self.max_iter = max_iter
        self.model_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "LogisticRegressionModel":
        """
        Fit the logistic regression model.
        """
        X_fit = X.copy()

        if self.add_intercept:
            X_fit = sm.add_constant(X_fit, has_constant="add")

        glm = sm.GLM(y, X_fit, family=sm.families.Binomial())

        if self.regularization is not None:
            penalty = self._compute_penalty(X_fit)
            self.model_ = glm.fit_regularized(
                method="elastic_net",
                alpha=penalty,
                L1_wt=self.regularization,
                maxiter=self.max_iter,
            )
        else:
            self.model_ = glm.fit()

        return self

    def get_coefficients(self) -> pd.DataFrame:
        """
        Return model coefficients as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and coefficients.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted first.")

        coef = self.model_.params
        return pd.DataFrame(
            {
                "feature": coef.index,
                "coefficient": coef.values,
            }
        )

    def get_odds_ratios(self) -> pd.DataFrame:
        """
        Return odds ratios for each feature.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and odds ratios.
        """
        coef_df = self.get_coefficients()
        coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
        return coef_df

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict probabilities for the positive class.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model must be fitted before calling predict_proba()."
            )

        X_pred = X.copy()

        if self.add_intercept:
            X_pred = sm.add_constant(X_pred, has_constant="add")

        return self.model_.predict(X_pred)

    def _compute_penalty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute penalty vector for statsmodels elastic-net regularization.

        The intercept (if present) is not penalized.

        Parameters
        ----------
        X : pd.DataFrame
            Training design matrix (with intercept if enabled).

        Returns
        -------
        np.ndarray
            Penalty weights.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        penalty = np.ones(n_features)

        if self.add_intercept:
            penalty[0] = 0.0

        return penalty / n_samples