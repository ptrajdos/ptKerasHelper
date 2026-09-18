# keras3_regression.py

import numpy as np
import keras

def main():
    # ----------------------------------------------------------------------
    # 1. Generate data
    # ----------------------------------------------------------------------

    rng = np.random.default_rng(42)

    n_samples = 2000
    n_features = 10

    X = rng.normal(size=(n_samples, n_features))

    # Nonlinear regression target + noise
    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.5 * X[:, 2] ** 2
        + 0.2 * rng.normal(size=n_samples)
    )

    # Train/test split
    n_train = int(0.8 * n_samples)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]


    # ----------------------------------------------------------------------
    # 2. Create Keras 3 model
    # ----------------------------------------------------------------------

    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),  # regression output
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )


    # ----------------------------------------------------------------------
    # 3. Train
    # ----------------------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=1,
    )


    # ----------------------------------------------------------------------
    # 4. Evaluate
    # ----------------------------------------------------------------------

    loss, mae = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    print(f"Test MSE: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")


    # ----------------------------------------------------------------------
    # 5. Predict
    # ----------------------------------------------------------------------

    y_pred = model.predict(X_test, verbose=0).ravel()

    print("First 10 predictions:")
    print(y_pred[:10])

    print("\nFirst 10 targets:")
    print(y_test[:10])


    # ----------------------------------------------------------------------
    # 6. R²
    # ----------------------------------------------------------------------

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

    r2 = 1.0 - ss_res / ss_tot

    print(f"\nTest R²: {r2:.4f}")

if __name__ == "__main__":
    main()