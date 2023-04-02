import numpy as np
import optuna


def objective(trial):
    x = trial.suggest_float("x", 0, 6)
    y = trial.suggest_float("y", 0, 6)

    f0 = np.sin(x) - y
    f1 = np.cos(x) + y**2

    return f0, f1


if __name__ == "__main__":
    sampler = optuna.integration.BoTorchSampler()
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    trials = sorted(study.best_trials, key=lambda t: t.values)

    print("Pareto front: ", len(trials))

    for trial in trials:
        print(f"  Trial#{trial.number}")
        print(f"    Values: Values={trial.values}")
        print(f"    Params: {trial.params}")
    
    fig = optuna.visualization.plot_pareto_front(study)
    fig.show()
