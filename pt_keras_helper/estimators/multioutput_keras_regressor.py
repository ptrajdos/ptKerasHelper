import keras

from sklearn.base import RegressorMixin
from keras.wrappers import SKLearnRegressor
from keras.src.wrappers.fixes import _validate_data
from keras.src.wrappers.utils import _check_model

def make_model_default(X, y, hidden_units=64):
    """
    Keras model factory.

    X and y are provided automatically by the wrapper.
    """

    n_features = X.shape[-1]    
    n_outputs = 1 if len(y.shape) == 1 else y.shape[1]

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


class MultiOutputKerasRegressor(SKLearnRegressor, RegressorMixin):
    """
    Keras 3 SKLearnRegressor with proper multi-output regression support.

    The model must be a callable:

        model(X, y, **model_kwargs) -> compiled Keras model

    X and y are passed to the factory so their dimensions can be inferred.
    """
    pass

    def fit(self, X, y, **kwargs):
        X, y = _validate_data(self, X, y, multi_output=True, y_numeric=True, ensure_2d=False, allow_nd=False)
        self.n_features_in_ = X.shape[-1]
        
        y = self._process_target(y, reset=True)
        model = self._get_model(X, y)
        _check_model(model)

        fit_kwargs = self.fit_kwargs or {}
        fit_kwargs.update(kwargs)
        self.history_ = model.fit(X, y, **fit_kwargs)

        self.model_ = model

        
        return self


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags