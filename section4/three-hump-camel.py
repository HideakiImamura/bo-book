import numpy as np
import plotly.graph_objects as go


def f(x, y):
    return 2 * x ** 2 - 1.05 * x ** 4 \
		+ x ** 6 / 6 + x * y + y ** 2

width = 2
x = np.linspace(-width, width, 100)
y = np.linspace(-width, width, 100)
z = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        z[i, j] = f(x[i], y[j])

pad=10
fig = go.Figure()
fig.add_trace(go.Surface(x=x, y=y, z=z))
fig.update_layout(
    showlegend=False,
    margin={"b": pad, "l": pad, "r": pad, "t": pad},
)
fig.update_traces(showscale=False)
fig.write_image("./three-hump-camel2.png")
#fig.show()
