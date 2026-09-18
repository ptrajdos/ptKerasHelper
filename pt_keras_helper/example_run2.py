import numpy as np
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.wrappers import SKLearnRegressor

def main():
    # ------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------

    def make_model(X, y):
        n_features = X.shape[1]
        n_targets = y.shape[1]

        model = keras.Sequential([
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(n_targets),       # multi-target output
        ])

        model.compile(
            optimizer="adam",
            loss="mse",
        )

        return model


    # ------------------------------------------------------------
    # Generate example data
    # ------------------------------------------------------------

    rng = np.random.default_rng(42)

    n_samples = 2000
    n_features = 10
    n_targets = 3

    X = rng.normal(size=(n_samples, n_features))

    # Three target variables
    y = np.column_stack([
        2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.1, n_samples),
        X[:, 2] - 3 * X[:, 3] + rng.normal(0, 0.1, n_samples),
        X[:, 4] + X[:, 5] + rng.normal(0, 0.1, n_samples),
    ])


    # ------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


    # ------------------------------------------------------------
    # sklearn-style Keras estimator
    # ------------------------------------------------------------

    reg = SKLearnRegressor(
        model=make_model,
        fit_kwargs={
            "epochs": 50,
            "batch_size": 32,
            "verbose": 0,
        },
    )


    # ------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)


    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    print("y_test shape:", y_test.shape)
    print("y_pred shape:", y_pred.shape)

    print("\nR² per target:")
    print(r2_score(y_test, y_pred, multioutput="raw_values"))

    print("\nRMSE per target:")
    print(
        np.sqrt(
            mean_squared_error(
                y_test,
                y_pred,
                multioutput="raw_values",
            )
        )
    )

    print("\nOverall R²:")
    print(r2_score(y_test, y_pred))

if __name__ == "__main__":
    main()