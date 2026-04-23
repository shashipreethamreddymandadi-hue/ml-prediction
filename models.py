def run_ann(train_df, test_df):

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    predictors = ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO"]
    target = "Current"

    tf.random.set_seed(42)
    np.random.seed(42)

    train_data = train_df[predictors]
    train_target = train_df[target]

    test_data = test_df[predictors]
    test_target = test_df[target]

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    y_mean = train_target.mean()
    y_std = train_target.std() + 1e-8

    train_target_scaled = (train_target - y_mean) / y_std
    test_target = np.array(test_target)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(len(predictors),)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    model.fit(
        train_data_scaled,
        train_target_scaled,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[callback],
        verbose=0
    )

    test_pred_scaled = model.predict(test_data_scaled).flatten()
    train_pred_scaled = model.predict(train_data_scaled).flatten()

    test_predictions = test_pred_scaled * y_std + y_mean
    train_predictions = train_pred_scaled * y_std + y_mean

    r2_test = r2_score(test_target, test_predictions)
    rmse_test = np.sqrt(mean_squared_error(test_target, test_predictions))
    r2_train = r2_score(train_target, train_predictions)

    voltages = np.linspace(train_df["Potential"].min(),
                           train_df["Potential"].max(), 500)

    delta_V = voltages.max() - voltages.min()
    mass = 0.002

    scan_rate = train_df["SCAN_RATE"].mean()
    v = scan_rate / 1000

    concentrations = np.linspace(0, 10, 21)

    capacitance_results = []
    cap_plot = []
    sample_cv_curve = None

    for i, conc in enumerate(concentrations):

        cv_input = pd.DataFrame({
            "Potential": voltages,
            "OXIDATION": 1,
            "Zn/Co_Conc": conc,
            "SCAN_RATE": scan_rate,
            "ZN": 1,
            "CO": 0
        })

        cv_scaled = scaler.transform(cv_input)
        pred_scaled = model.predict(cv_scaled).flatten()
        predicted_current = pred_scaled * y_std + y_mean

        if i == 0:
            sample_cv_curve = {
                "voltage": voltages.tolist(),
                "current": predicted_current.tolist()
            }

        area = np.trapezoid(np.abs(predicted_current), voltages)
        C = area / (2 * mass * delta_V * v)

        E = 0.5 * C * (delta_V ** 2) / 3600
        t = delta_V / v
        P = (E * 3600) / t if t != 0 else 0

        capacitance_results.append({"C": C, "E": E, "P": P})
        cap_plot.append(C)

    best_index = np.argmax(cap_plot)
    best = capacitance_results[best_index]

    return {
        "metrics": {
            "r2_test": float(r2_test),
            "rmse_test": float(rmse_test),
            "r2_train": float(r2_train)
        },
        "optimization": {
            "best_concentration": float(concentrations[best_index]),
            "capacitance": float(best["C"]),
            "energy": float(best["E"]),
            "power": float(best["P"])
        },
        "plots": {
            "actual_vs_predicted": {
                "actual": test_target.tolist(),
                "predicted": test_predictions.tolist()
            },
            "cv_curve": sample_cv_curve,
            "capacitance_vs_concentration": {
                "concentration": concentrations.tolist(),
                "capacitance": cap_plot
            }
        }
    }


def run_rf(train_df, test_df):

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score

    predictors = ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO"]
    target = "Current"

    for col in predictors + [target]:
        if col not in train_df.columns or col not in test_df.columns:
            return {"error": f"Missing column: {col}"}

    train_data = train_df[predictors]
    train_target = train_df[target]

    test_data = test_df[predictors]
    test_target = test_df[target]

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=11,
        random_state=42
    )

    scores = cross_val_score(rf_model, train_data, train_target, cv=5, scoring='r2')
    rf_model.fit(train_data, train_target)

    test_predictions = rf_model.predict(test_data)
    train_predictions = rf_model.predict(train_data)

    r2_test = r2_score(test_target, test_predictions)
    rmse_test = np.sqrt(mean_squared_error(test_target, test_predictions))
    r2_train = r2_score(train_target, train_predictions)

    # ✅ FIXED HERE
    feature_importance = {
        k: float(v) for k, v in zip(predictors, rf_model.feature_importances_)
    }

    voltages = np.linspace(train_df["Potential"].min(),
                           train_df["Potential"].max(), 500)

    delta_V = voltages.max() - voltages.min()
    mass = 0.002
    scan_rate = train_df["SCAN_RATE"].mean()
    v = scan_rate / 1000

    concentrations = np.linspace(0, 10, 21)

    capacitance_results = []
    cap_values = []
    cv_curve_data = None

    for i, conc in enumerate(concentrations):

        cv_input = pd.DataFrame({
            "Potential": voltages,
            "OXIDATION": 1,
            "Zn/Co_Conc": conc,
            "SCAN_RATE": scan_rate,
            "ZN": 1,
            "CO": 0
        })[predictors]

        predicted_current = rf_model.predict(cv_input)

        if i == 0:
            cv_curve_data = {
                "voltage": voltages.tolist(),
                "current": predicted_current.tolist()
            }

        area = np.trapezoid(np.abs(predicted_current), voltages)

        C = area / (2 * mass * delta_V * v)
        E = 0.5 * C * (delta_V ** 2) / 3600
        t = delta_V / v
        P = (E * 3600) / t if t != 0 else 0

        capacitance_results.append({"C": C, "E": E, "P": P})
        cap_values.append(C)

    best_index = int(np.argmax(cap_values))
    best = capacitance_results[best_index]

    return {
        "metrics": {
            "r2_test": float(r2_test),
            "rmse_test": float(rmse_test),
            "r2_train": float(r2_train),
            "cv_r2_mean": float(scores.mean())
        },
        "optimization": {
            "best_concentration": float(concentrations[best_index]),
            "capacitance": float(best["C"]),
            "energy": float(best["E"]),
            "power": float(best["P"])
        },
        "feature_importance": feature_importance,
        "plots": {
            "actual_vs_predicted": {
                "actual": test_target.tolist(),
                "predicted": test_predictions.tolist()
            },
            "cv_curve": cv_curve_data,
            "capacitance_vs_concentration": {
                "concentration": concentrations.tolist(),
                "capacitance": cap_values
            }
        }
    }


def run_xgb(train_df, test_df):

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error, r2_score
    from xgboost import XGBRegressor

    predictors = ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO"]
    target = "Current"

    for col in predictors + [target]:
        if col not in train_df.columns or col not in test_df.columns:
            return {"error": f"Missing column: {col}"}

    train_data = train_df[predictors]
    train_target = train_df[target]

    test_data = test_df[predictors]
    test_target = test_df[target]

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'random_state': 42
    }

    xgb_model = XGBRegressor(**params, n_estimators=200)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, train_data, train_target, scoring='r2', cv=cv)

    xgb_model.fit(train_data, train_target)

    test_predictions = xgb_model.predict(test_data)
    train_predictions = xgb_model.predict(train_data)

    r2_test = r2_score(test_target, test_predictions)
    rmse_test = np.sqrt(mean_squared_error(test_target, test_predictions))
    r2_train = r2_score(train_target, train_predictions)

    # ✅ FIXED HERE
    feature_importance = {
        k: float(v) for k, v in zip(predictors, xgb_model.feature_importances_)
    }

    voltages = np.linspace(train_df["Potential"].min(),
                           train_df["Potential"].max(), 500)

    delta_V = voltages.max() - voltages.min()
    mass = 0.002
    scan_rate = train_df["SCAN_RATE"].mean()
    v = scan_rate / 1000

    concentrations = np.linspace(0, 10, 21)

    capacitance_results = []
    cap_values = []
    cv_curve_data = None

    for i, conc in enumerate(concentrations):

        cv_input = pd.DataFrame({
            "Potential": voltages,
            "OXIDATION": 1,
            "Zn/Co_Conc": conc,
            "SCAN_RATE": scan_rate,
            "ZN": 1,
            "CO": 0
        })[predictors]

        predicted_current = xgb_model.predict(cv_input)

        if i == 0:
            cv_curve_data = {
                "voltage": voltages.tolist(),
                "current": predicted_current.tolist()
            }

        area = np.trapezoid(np.abs(predicted_current), voltages)

        C = area / (2 * mass * delta_V * v)
        E = 0.5 * C * (delta_V ** 2) / 3600
        t = delta_V / v
        P = (E * 3600) / t if t != 0 else 0

        capacitance_results.append({
            "C": float(C),
            "E": float(E),
            "P": float(P)
        })

        cap_values.append(float(C))

    best_index = int(np.argmax(cap_values))
    best = capacitance_results[best_index]

    return {
        "metrics": {
            "r2_test": float(r2_test),
            "rmse_test": float(rmse_test),
            "r2_train": float(r2_train),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_scores": cv_scores.tolist()
        },
        "optimization": {
            "best_concentration": float(concentrations[best_index]),
            "capacitance": best["C"],
            "energy": best["E"],
            "power": best["P"]
        },
        "feature_importance": feature_importance,
        "plots": {
            "actual_vs_predicted": {
                "actual": test_target.tolist(),
                "predicted": test_predictions.tolist()
            },
            "cv_curve": cv_curve_data,
            "capacitance_vs_concentration": {
                "concentration": concentrations.tolist(),
                "capacitance": cap_values
            }
        }
    }


def run_all(train_df, test_df):

    models = {}

    try:
        models["ANN"] = run_ann(train_df, test_df)
    except Exception as e:
        models["ANN"] = {"error": str(e)}

    try:
        models["RF"] = run_rf(train_df, test_df)
    except Exception as e:
        models["RF"] = {"error": str(e)}

    try:
        models["XGB"] = run_xgb(train_df, test_df)
    except Exception as e:
        models["XGB"] = {"error": str(e)}

    valid_models = {
        k: v for k, v in models.items()
        if "metrics" in v
    }

    if not valid_models:
        return {"error": "All models failed"}

    best_model = max(
        valid_models,
        key=lambda m: valid_models[m]["metrics"]["r2_test"]
    )

    return {
        "models": models,
        "best_model": best_model
    }