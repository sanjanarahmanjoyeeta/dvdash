from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Precompute datasets
# Normal distribution
means = np.arange(-5, 6, 1)
std_devs = np.arange(1, 11, 1)
precomputed_data_normal = {
    (mean, sd): np.random.normal(loc=mean, scale=sd, size=1000)
    for mean in means
    for sd in std_devs
}

# Binomial distribution
n_trials = range(1, 21)
p_success = np.arange(0.1, 1.1, 0.1)
precomputed_data_binomial = {
    (n, round(p, 1)): np.random.binomial(n=n, p=p, size=1000)
    for n in n_trials
    for p in p_success
}

# Poisson distribution
lambdas = np.arange(1, 21, 1)
precomputed_data_poisson = {
    lam: np.random.poisson(lam=lam, size=1000)
    for lam in lambdas
}

# Uniform distribution
a_values = np.arange(0, 11, 1)
b_values = np.arange(1, 12, 1)
precomputed_data_uniform = {
    (a, b): np.random.uniform(low=a, high=b, size=1000)
    for a in a_values
    for b in b_values if a < b
}

# Layout
app.layout = dbc.Container([
    html.H1("Distribution Visualizations", className="text-center my-4"),
    
    # Normal Distribution Panel
    dbc.Row([
        dbc.Col([
            html.H4("Normal Distribution"),
            html.Label("Description:"),
            html.Div(
                "The Normal distribution is widely used in statistics and natural phenomena modeling. "
                "It is symmetric, centered around its mean, with a spread defined by the standard deviation.",
                style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "10px", "border-radius": "5px"}
            ),
            html.Label("Mean:"),
            dcc.Slider(
                id="mean-slider",
                min=-5, max=5, step=1, value=0,
                marks={i: str(i) for i in range(-5, 6)}
            ),
            html.Label("Standard Deviation:"),
            dcc.Slider(
                id="sd-slider",
                min=1, max=10, step=1, value=1,
                marks={i: str(i) for i in range(1, 11)}
            ),
        ], width=4),
        dbc.Col([
            dcc.Graph(id="distribution-graph-normal", style={"height": "300px"}),
            dcc.Graph(id="density-graph-normal", style={"height": "300px"}),
            html.Label("Generated Python Code:"),
            html.Pre(id="generated-code-normal", style={"background-color": "#f8f9fa", "padding": "10px"})
        ], width=8)
    ], className="my-4"),
    
    # Binomial Distribution Panel
    dbc.Row([
        dbc.Col([
            html.H4("Binomial Distribution"),
            html.Label("Description:"),
            html.Div(
                "The Binomial distribution models the number of successes in a fixed number of independent trials. "
                "It is useful for experiments with two possible outcomes (e.g., success or failure).",
                style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "10px", "border-radius": "5px"}
            ),
            html.Label("Number of Trials (n):"),
            dcc.Slider(
                id="n-slider",
                min=1, max=20, step=1, value=10,
                marks={i: str(i) for i in range(1, 21)}
            ),
            html.Label("Probability of Success (p):"),
            dcc.Slider(
                id="p-slider",
                min=0.1, max=1.0, step=0.1, value=0.5,
                marks={round(p, 1): f"{p:.1f}" for p in np.arange(0.1, 1.1, 0.1)}
            ),
        ], width=4),
        dbc.Col([
            dcc.Graph(id="distribution-graph-binomial", style={"height": "300px"}),
            dcc.Graph(id="density-graph-binomial", style={"height": "300px"}),
            html.Label("Generated Python Code:"),
            html.Pre(id="generated-code-binomial", style={"background-color": "#f8f9fa", "padding": "10px"})
        ], width=8)
    ], className="my-4"),
    
    # Poisson Distribution Panel
    dbc.Row([
        dbc.Col([
            html.H4("Poisson Distribution"),
            html.Label("Description:"),
            html.Div(
                "The Poisson distribution models the number of events occurring within a fixed interval. "
                "It is useful for counting occurrences of rare events.",
                style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "10px", "border-radius": "5px"}
            ),
            html.Label("Lambda (λ):"),
            dcc.Slider(
                id="lambda-slider",
                min=1, max=20, step=1, value=5,
                marks={i: str(i) for i in range(1, 21)}
            ),
        ], width=4),
        dbc.Col([
            dcc.Graph(id="distribution-graph-poisson", style={"height": "300px"}),
            dcc.Graph(id="density-graph-poisson", style={"height": "300px"}),
            html.Label("Generated Python Code:"),
            html.Pre(id="generated-code-poisson", style={"background-color": "#f8f9fa", "padding": "10px"})
        ], width=8)
    ], className="my-4"),
    
    # Uniform Distribution Panel
    dbc.Row([
        dbc.Col([
            html.H4("Uniform Distribution"),
            html.Label("Description:"),
            html.Div(
                "The Uniform distribution assumes equal likelihood of outcomes within the range. "
                "It is used when all values in a range are equally likely.",
                style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "10px", "border-radius": "5px"}
            ),
            html.Label("Min (a):"),
            dcc.Slider(
                id="a-slider",
                min=0, max=10, step=1, value=2,
                marks={i: str(i) for i in range(0, 11)}
            ),
            html.Label("Max (b):"),
            dcc.Slider(
                id="b-slider",
                min=1, max=12, step=1, value=8,
                marks={i: str(i) for i in range(1, 13)}
            ),
        ], width=4),
        dbc.Col([
            dcc.Graph(id="distribution-graph-uniform", style={"height": "300px"}),
            dcc.Graph(id="density-graph-uniform", style={"height": "300px"}),
            html.Label("Generated Python Code:"),
            html.Pre(id="generated-code-uniform", style={"background-color": "#f8f9fa", "padding": "10px"})
        ], width=8)
    ], className="my-4"),
])

# Callback for Normal Distribution
@app.callback(
    [Output("distribution-graph-normal", "figure"),
     Output("density-graph-normal", "figure"),
     Output("generated-code-normal", "children")],
    [Input("mean-slider", "value"),
     Input("sd-slider", "value")]
)
def update_normal_distribution(mean, sd):
    data = precomputed_data_normal[(mean, sd)]
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)

    hist_fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
    hist_fig.update_layout(title="Normal Distribution (Histogram)", xaxis_title="Values", yaxis_title="Frequency")
    
    density_fig = go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals), mode='lines')])
    density_fig.update_layout(title="Normal Distribution (Density Plot)", xaxis_title="Values", yaxis_title="Density")
    
    python_code = f"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

mean = {mean}
sd = {sd}
data = np.random.normal(loc=mean, scale=sd, size=1000)

plt.hist(data, bins=30, alpha=0.7, label='Histogram')

kde = gaussian_kde(data)
x_vals = np.linspace(min(data), max(data), 1000)
plt.plot(x_vals, kde(x_vals), label='Density')
plt.legend()
plt.show()
"""
    return hist_fig, density_fig, python_code

# Callback for Binomial Distribution
@app.callback(
    [Output("distribution-graph-binomial", "figure"),
     Output("density-graph-binomial", "figure"),
     Output("generated-code-binomial", "children")],
    [Input("n-slider", "value"),
     Input("p-slider", "value")]
)
def update_binomial_distribution(n, p):
    data = precomputed_data_binomial[(n, round(p, 1))]
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)

    hist_fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
    hist_fig.update_layout(title="Binomial Distribution (Histogram)", xaxis_title="Values", yaxis_title="Frequency")
    
    density_fig = go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals), mode='lines')])
    density_fig.update_layout(title="Binomial Distribution (Density Plot)", xaxis_title="Values", yaxis_title="Density")
    
    python_code = f"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

n = {n}
p = {p}
data = np.random.binomial(n=n, p=p, size=1000)

plt.hist(data, bins=30, alpha=0.7, label='Histogram')

kde = gaussian_kde(data)
x_vals = np.linspace(min(data), max(data), 1000)
plt.plot(x_vals, kde(x_vals), label='Density')
plt.legend()
plt.show()
"""
    return hist_fig, density_fig, python_code

# Callback for Poisson Distribution
@app.callback(
    [Output("distribution-graph-poisson", "figure"),
     Output("density-graph-poisson", "figure"),
     Output("generated-code-poisson", "children")],
    Input("lambda-slider", "value")
)
def update_poisson_distribution(lam):
    data = precomputed_data_poisson[lam]
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)

    hist_fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
    hist_fig.update_layout(title="Poisson Distribution (Histogram)", xaxis_title="Values", yaxis_title="Frequency")
    
    density_fig = go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals), mode='lines')])
    density_fig.update_layout(title="Poisson Distribution (Density Plot)", xaxis_title="Values", yaxis_title="Density")
    
    python_code = f"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

lam = {lam}
data = np.random.poisson(lam=lam, size=1000)

plt.hist(data, bins=30, alpha=0.7, label='Histogram')

kde = gaussian_kde(data)
x_vals = np.linspace(min(data), max(data), 1000)
plt.plot(x_vals, kde(x_vals), label='Density')
plt.legend()
plt.show()
"""
    return hist_fig, density_fig, python_code

# Callback for Uniform Distribution
@app.callback(
    [Output("distribution-graph-uniform", "figure"),
     Output("density-graph-uniform", "figure"),
     Output("generated-code-uniform", "children")],
    [Input("a-slider", "value"),
     Input("b-slider", "value")]
)
def update_uniform_distribution(a, b):
    if a >= b:
        return {}, {}, "Error: Min (a) must be less than Max (b)."
    data = precomputed_data_uniform[(a, b)]
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)

    hist_fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
    hist_fig.update_layout(title="Uniform Distribution (Histogram)", xaxis_title="Values", yaxis_title="Frequency")
    
    density_fig = go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals), mode='lines')])
    density_fig.update_layout(title="Uniform Distribution (Density Plot)", xaxis_title="Values", yaxis_title="Density")
    
    python_code = f"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

a = {a}
b = {b}
data = np.random.uniform(low=a, high=b, size=1000)

plt.hist(data, bins=30, alpha=0.7, label='Histogram')

kde = gaussian_kde(data)
x_vals = np.linspace(min(data), max(data), 1000)
plt.plot(x_vals, kde(x_vals), label='Density')
plt.legend()
plt.show()
"""
    return hist_fig, density_fig, python_code

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)