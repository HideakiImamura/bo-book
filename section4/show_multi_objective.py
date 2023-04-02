import optuna
import numpy as np


def objective(trial):
    x = trial.suggest_float("x", 0, 6)
    y = trial.suggest_float("y", 0, 6)

    f0 = np.sin(x) - y
    f1 = np.cos(x) + y**2

    return f0, f1


if __name__ == "__main__":
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, n_trials=1000)

    fig = optuna.visualization.plot_pareto_front(study)
    fig.show()
