from typing import Optional

import numpy as np
import optuna
import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood


class ThompsonSampling(AnalyticAcquisitionFunction):
    def __init__(
        self, 
        model: Model, 
        bounds: torch.Tensor, 
        delta: int = 10, 
        posterior_transform: Optional[PosteriorTransform] = None, 
    ) -> None:
        super().__init__(model=model, posterior_transform=posterior_transform)

        n_dim = bounds.shape[-1]
        grids_for_each_dim = []
        for i in range(n_dim):
            grids_for_each_dim.append(torch.linspace(bounds[0, i], bounds[1, i], delta))
        X = torch.cartesian_prod(*grids_for_each_dim)
        
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        Y = posterior.rsample()

        self.candidates = X[torch.argmax(Y)]
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return - ((X - self.candidates) ** 2).sum(axis=(1, 2))


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

    acqf = ThompsonSampling(model=model, bounds=bounds, delta=30)

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
	x = trial.suggest_float("x", 0, 1)
	y = trial.suggest_float("y", 0, 1)
	return np.sin(np.pi * (x ** 2 + y ** 2)) + np.random.normal(scale=1e-3)


if __name__ == "__main__":
    sampler = optuna.integration.BoTorchSampler(candidates_func=candidates_func)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=50)
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
