import numpy as np
import keras

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from keras.wrappers import SKLearnRegressor


class MultiOutputKerasRegressor(SKLearnRegressor, RegressorMixin):
    """
    Keras 3 SKLearnRegressor with proper multi-output regression support.

    The model must be a callable:

        model(X, y, **model_kwargs) -> compiled Keras model

    X and y are passed to the factory so their dimensions can be inferred.
    """

    def fit(self, X, y, **kwargs):
        # Keras' SKLearnRegressor validates y as 1D.
        # We replace that validation with sklearn's multi-output validation.
        X, y = check_X_y(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=False,
            allow_nd=True,
        )

        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1

        # Infer dimensions through the model factory.
        if callable(self.model):
            model_kwargs = self.model_kwargs or {}

            self.model_ = self.model(
                X,
                y,
                **model_kwargs,
            )
        else:
            if not self.warm_start:
                self.model_ = keras.models.clone_model(self.model)
            else:
                self.model_ = self.model

        # Fit arguments supplied to constructor + arguments supplied to fit()
        fit_kwargs = dict(self.fit_kwargs or {})
        fit_kwargs.update(kwargs)

        self.history_ = self.model_.fit(
            X,
            y,
            **fit_kwargs,
        )

        return self

    def predict(self, X):
        X = check_array(
            X,
            ensure_2d=False,
            allow_nd=True,
        )

        return self.model_.predict(
            X,
            verbose=0,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags