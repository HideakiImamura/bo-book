import optuna


def objective(trial):
    x = trial.suggest_float("x", -15, 30)
    y = trial.suggest_float("y", -15, 30)

    c0 = (x - 5) ** 2 + y**2 - 25
    c1 = -((x - 8) ** 2) - (y + 3) ** 2 + 7.7

    trial.set_user_attr("constraint", (c0, c1))

    v0 = 4 * x**2 + 4 * y**2

    return v0


def constraints(trial):
    return trial.user_attrs["constraint"]


if __name__ == "__main__":
    sampler = optuna.integration.BoTorchSampler(
        constraints_func=constraints,
        n_startup_trials=10,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=32)

    print("Number of finished trials: ", len(study.trials))
    best_constraint = study.best_trial.user_attrs["constraint"]
    print(
        f"Best value: {study.best_value} ("
        f"params: {study.best_params}, "
        f"constraint: {best_constraint})"
    )

    feasible_trials = [
        t for t in study.trials 
        if all([c <= 0 for c in t.user_attrs["constraint"]])
    ]
    best_feasible_trial = min(feasible_trials, key=lambda t: t.value)
    
    print("Number of finished feasible trials: ", len(feasible_trials))
    best_constraint = best_feasible_trial.user_attrs["constraint"]
    print(
        f"Best value: {best_feasible_trial.value} ("
        f"params: {best_feasible_trial.params}, "
        f"constraint: {best_constraint})"
    )
