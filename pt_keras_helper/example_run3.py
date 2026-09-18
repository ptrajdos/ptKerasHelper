import numpy as np
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import check_X_y, check_array
from keras.wrappers import SKLearnRegressor

from pt_keras_helper.estimators.multioutput_keras_regressor import MultiOutputKerasRegressor


# ----------------------------------------------------------------------
# LSTM model factory
# ----------------------------------------------------------------------

def make_lstm(X, y, units=32):
    """
    X: (samples, timesteps, features)
    y: (samples, outputs)
    """

    n_timesteps = X.shape[1]
    n_features = X.shape[2]
    n_outputs = y.shape[1]

    print(
        f"Building LSTM: "
        f"timesteps={n_timesteps}, "
        f"features={n_features}, "
        f"outputs={n_outputs}"
    )

    model = keras.Sequential([
        keras.layers.Input(
            shape=(n_timesteps, n_features)
        ),

        keras.layers.LSTM(units),

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
# Generate example time-series data
# ----------------------------------------------------------------------

def create_data(
    n_samples=2000,
    timesteps=20,
    n_features=8,
    n_outputs=3,
):
    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(n_samples, timesteps, n_features)
    )

    # Create three targets from temporal properties of X.
    y = np.empty((n_samples, n_outputs))

    y[:, 0] = (
        X[:, :, 0].mean(axis=1)
        + 0.5 * X[:, :, 1].mean(axis=1)
    )

    y[:, 1] = (
        X[:, -1, 2]
        + 0.5 * X[:, -1, 3]
    )

    y[:, 2] = (
        X[:, :, 4].max(axis=1)
        - X[:, :, 5].min(axis=1)
    )

    # Add a little noise.
    y += 0.05 * rng.normal(size=y.shape)

    return X, y


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    X, y = create_data()

    print("X:", X.shape)
    print("y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    reg = MultiOutputKerasRegressor(
        model=make_lstm,

        # Optional arguments passed to make_lstm()
        model_kwargs={
            "units": 32,
        },

        # Arguments passed to model.fit()
        fit_kwargs={
            "epochs": 10,
            "batch_size": 32,
            "verbose": 0,
        },
    )

    reg.fit(
        X_train,
        y_train,
    )

    y_pred = reg.predict(X_test)

    print()
    print("y_test:", y_test.shape)
    print("y_pred:", y_pred.shape)

    print()
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 :", r2_score(y_test, y_pred))

    print()
    print("Per-target R2:")

    for i in range(y.shape[1]):
        print(
            f"  target {i}: "
            f"{r2_score(y_test[:, i], y_pred[:, i]):.4f}"
        )

    print()
    reg.model_.summary()


if __name__ == "__main__":
    main()