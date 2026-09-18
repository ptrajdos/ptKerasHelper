import numpy as np
import keras

from pt_keras_helper.estimators.multioutput_keras_regressor import MultiOutputKerasRegressor

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