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


# ----------------------------------------------------------------------
# Model factory
# ----------------------------------------------------------------------

def make_model(X, y, hidden_units=64):
    """
    Keras model factory.

    X and y are provided automatically by the wrapper.
    """

    n_features = X.shape[-1]
    n_outputs = y.shape[1]

    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(hidden_units, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_outputs),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    # 2000 samples, 10 input features
    X = rng.normal(size=(2000, 10))

    # 3 regression targets
    y = np.column_stack([
        X[:, 0] + 2 * X[:, 1],
        X[:, 2] - X[:, 3],
        X[:, 4] + X[:, 5] + X[:, 6],
    ])

    print("X:", X.shape)
    print("y:", y.shape)

    reg = MultiOutputKerasRegressor(
        model=make_model,
        model_kwargs={
            "hidden_units": 64,
        },
        fit_kwargs={
            "epochs": 10,
            "batch_size": 32,
            "verbose": 0,
            "validation_split": 0.1,
        },
    )

    reg.fit(X, y)

    predictions = reg.predict(X[:5])

    print("Predictions:", predictions.shape)
    print(predictions)

    print()
    print("Keras model:")
    reg.model_.summary()


if __name__ == "__main__":
    main()