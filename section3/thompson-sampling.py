from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import plotly.graph_objects as go
import torch

if __name__ == "__main__":
    n_trials = 10
    low = 0.
    high = 10.
    x = torch.rand((n_trials, 1)) * (high - low) + low
    y = torch.sin(x) + torch.rand((n_trials, 1)) * 1e-3

    model = SingleTaskGP(x, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    x_model = torch.linspace(low, high, 100)
    posterior = model.posterior(x_model)
    y_model = posterior.mean.squeeze()
    y_upper = y_model + posterior.variance.sqrt().squeeze()
    y_lower = y_model - posterior.variance.sqrt().squeeze()

    sample1 = posterior.rsample().squeeze()
    ind1 = torch.argmin(sample1)

    sample2 = posterior.rsample().squeeze()
    ind2 = torch.argmin(sample2)

    layout = go.Layout(
        title="トンプソン抽出の例",
        xaxis={"title": "x: 探索点"},
        yaxis={"title": "y: 目的関数値"},
        font={"size": 25}
    )

    traces = []
    traces.append(go.Scatter(x=x.squeeze(), y=y.squeeze(), mode="markers", name="履歴上のトライアル"))
    traces.append(go.Scatter(x=x_model.detach().numpy(), y=y_model.detach().numpy(), name="事後分布"))
    traces.append(go.Scatter(x=x_model.detach().numpy(), y=y_upper.detach().numpy(), mode="lines", line=dict(width=0.01), showlegend=False))
    traces.append(go.Scatter(x=x_model.detach().numpy(), y=y_lower.detach().numpy(), mode="none", showlegend=False, fill="tonexty", fillcolor="rgba(255,0,0,0.2)"))
    traces.append(go.Scatter(x=x_model.detach().numpy(), y=sample1.detach().numpy(), name="サンプルされた関数1"))
    traces.append(go.Scatter(x=x_model[ind1].detach().numpy(), y=sample1[ind1].detach().numpy(), showlegend=False, mode="markers", marker=dict(size=12)))
    traces.append(go.Scatter(x=x_model.detach().numpy(), y=sample2.detach().numpy(), name="サンプルされた関数2"))
    traces.append(go.Scatter(x=x_model[ind2].detach().numpy(), y=sample2[ind2].detach().numpy(), showlegend=False, mode="markers", marker=dict(size=12)))


    fig = go.Figure(data=traces, layout=layout)
    fig.show()
