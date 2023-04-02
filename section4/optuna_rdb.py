import optuna


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return 2 * x ** 2 - 1.05 * x ** 4 \
        + x ** 6 / 6 + x * y + y ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///test.db", 
        study_name="test",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    print(
        f"Best value: {study.best_value} "
        f"(params: {study.best_params})"
    )
