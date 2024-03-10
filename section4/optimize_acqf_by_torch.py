import numpy as np
import optuna
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.initializers import initialize_q_batch_nonneg


def candidates_func(
    train_x: torch.Tensor, 
    train_obj: torch.Tensor,
    train_con: torch.Tensor, 
    bounds: torch.Tensor,
    pending_x: torch.Tensor,
) -> torch.Tensor:

    train_x = normalize(train_x, bounds=bounds)
    
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = ExpectedImprovement(
        model=model, best_f=train_obj.max(),
    )

    n_dim = bounds.shape[0]
    n_restarts = 5
    n_iter = 100
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    Xraw = torch.rand(100 * n_restarts, 1, n_dim)
    Yraw = acqf(Xraw)
    X = initialize_q_batch_nonneg(Xraw, Yraw, n_restarts)
    X.requires_grad_(True)

    optimizer = torch.optim.Adam([X], lr=0.01)

    for i in range(n_iter):
        optimizer.zero_grad()
        losses = -acqf(X)
        loss = losses.sum()
    
        loss.backward()
        optimizer.step()
    
        for i in range(n_dim):
            X.data[..., i].clamp_(0, 1)
    
    candidates = X[torch.argmax(acqf(X))]
    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def objective(trial: optuna.Trial) -> float:
	x = trial.suggest_float("x", -5, 5)
	y = trial.suggest_float("y", -5, 5)
	return 2 * x ** 2 - 1.05 * x ** 4 \
		+ x ** 6 / 6 + x * y + y ** 2 + np.random.normal(scale=1e-3)


if __name__ == "__main__":
    sampler = optuna.integration.BoTorchSampler(candidates_func=candidates_func)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=50)
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
