from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import optuna
import torch


def default_candidates_func(
    train_x, train_obj, train_con, bounds, pending_x
):

    train_x = normalize(train_x, bounds=bounds)
    
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=train_obj.max(),
        sampler=SobolQMCNormalSampler(torch.Size((256,))),
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def customized_candidates_func(
    train_x, train_obj, train_con, bounds, pending_x
):
        
    train_x = normalize(train_x, bounds=bounds)
    
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = ExpectedImprovement(
        model=model,
        best_f=train_obj.max(),
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def objective(trial):
	x = trial.suggest_float("x", -5, 5)
	y = trial.suggest_float("y", -5, 5)
	return 2 * x ** 2 - 1.05 * x ** 4 \
		+ x ** 6 / 6 + x * y + y ** 2 + np.random.normal(scale=1e-3)


if __name__ == "__main__":

    sampler1 = optuna.integration.BoTorchSampler(candidates_func=default_candidates_func)
    study1 = optuna.create_study(sampler=sampler1)
    study1.optimize(objective, n_trials=50)

    sampler2 = optuna.integration.BoTorchSampler(candidates_func=customized_candidates_func)
    study2 = optuna.create_study(sampler=sampler2)
    study2.optimize(objective, n_trials=50)

    print(f"Study1's best value: {study1.best_value} (params: {study1.best_params})")
    print(f"Study2's best value: {study2.best_value} (params: {study2.best_params})")
