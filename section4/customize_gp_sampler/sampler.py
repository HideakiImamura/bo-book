from __future__ import annotations

from typing import Any
from typing import cast
import warnings

import numpy as np

import optuna
from optuna._gp import search_space as gp_search_space
from optuna._gp import acqf
from optuna._gp import gp
from optuna.distributions import BaseDistribution
from optuna.samplers._gp import sampler as optuna_gp_sampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


EPS = 1e-10


def _warn_and_convert_inf(
    values: np.ndarray,
) -> np.ndarray:
    if np.any(~np.isfinite(values)):
        warnings.warn(
            "GPSampler cannot handle +/-inf, so we clip them to the best/worst finite value."
        )

        finite_vals = values[np.isfinite(values)]
        best_finite_val = np.max(finite_vals, axis=0, initial=0.0)
        worst_finite_val = np.min(finite_vals, axis=0, initial=0.0)

        return np.clip(values, worst_finite_val, best_finite_val)
    return values


class GPUCBSampler(optuna.samplers.GPSampler):
    def __init__(
        self,
        *,
        dim_coefficient: float = 0.2,
        n_trials_coefficient: float = 2.0,
        seed: int | None = None,
        n_startup_trials: int = 10,
    ) -> None:
        super().__init__(seed=seed, n_startup_trials=n_startup_trials)
        self._dim_coefficient = dim_coefficient
        self._n_trials_coefficient = n_trials_coefficient
    
    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        _sign = -1.0 if study.direction == StudyDirection.MINIMIZE else 1.0
        score_vals = np.array([_sign * cast(float, trial.value) for trial in trials])

        score_vals = _warn_and_convert_inf(score_vals)
        standardized_score_vals = (score_vals - np.mean(score_vals)) / max(EPS, np.std(score_vals))

        if self._kernel_params_cache is not None and len(
            self._kernel_params_cache.inverse_squared_lengthscales
        ) != len(internal_search_space.scale_types):
            # Clear cache if the search space changes.
            self._kernel_params_cache = None

        kernel_params = gp.fit_kernel_params(
            X=normalized_params,
            Y=standardized_score_vals,
            is_categorical=(
                internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            initial_kernel_params=self._kernel_params_cache,
            deterministic_objective=self._deterministic,
        )
        self._kernel_params_cache = kernel_params

        beta = (
            self._dim_coefficient * 
            internal_search_space.bounds.shape[0] * 
            np.log(
                self._n_trials_coefficient * len(trials)
            )
        )
        acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.UCB,
            kernel_params=kernel_params,
            search_space=internal_search_space,
            X=normalized_params,
            Y=standardized_score_vals,
            beta=beta,
        )
        best_params = normalized_params[np.argmax(standardized_score_vals), :]

        normalized_param = self._optimize_acqf(acqf_params, best_params)
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)
