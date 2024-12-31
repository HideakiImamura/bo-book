import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return x ** 2


if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(sampler=module.AutoSampler())
    study.optimize(objective, n_trials=300)

    print(study.best_params)
    print(study.best_value)
