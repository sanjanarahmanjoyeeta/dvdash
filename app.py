from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# Load datasets
def load_dataset(filepath):
    # Automatically handle 'Unnamed: 0' by setting it as the index or dropping it
    df = pd.read_csv(filepath)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])  # Drop if unnecessary
    return df

iris = load_dataset("data/iris.csv")
mtcars = load_dataset("data/mtcars.csv")
diamonds = load_dataset("data/diamonds.csv")
titanic = load_dataset("data/titanic.csv")

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for Dash compatibility
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)


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
    dcc.Tabs([
        dcc.Tab(label="Home", children=[
            dbc.Container([
                html.Div([
                    html.Img(src="/assets/logo.png", height="100px", style={"display": "block", "margin": "0 auto"}),
                    html.H1("Welcome to Your Data Visualization Playground!", style={"text-align": "center"}),
                    html.H3("Meet Your Guide: Sanjana Rahman", style={"text-align": "center"}),
                    html.P("Hey there! I'm Sanjana, a Ph.D. student at the University of Texas at El Paso. "
                           "I created this app to make learning data visualization accessible, interactive, and engaging for college students."),
                    
                    html.H3("What Can You Do Here?"),
                    html.Ul([
                        html.Li("Create stunning plots, charts, and graphs."),
                        html.Li("Understand distributions and basic statistical concepts."),
                        html.Li("Perform beginner-friendly data analyses."),
                        html.Li("Upload your own datasets and visualize them interactively."),
                        html.Li("Learn the R code behind every visualization to sharpen your coding skills.")
                    ]),
                    
                    html.P("This app is packed with features to help you gain confidence in data visualization and analysis. "
                           "Explore, experiment, and enjoy the process of uncovering insights through data."),
                    
                    html.H3("Datasets Available in This App"),
                    html.P("Below is a detailed description of the datasets you can use in this app, including their variables and what they represent:"),
                    
                    # Dataset Descriptions
                    html.H4("Iris Dataset"),
                    html.Ul([
                        html.Li("Sepal.Length: Length of the sepal in centimeters."),
                        html.Li("Sepal.Width: Width of the sepal in centimeters."),
                        html.Li("Petal.Length: Length of the petal in centimeters."),
                        html.Li("Petal.Width: Width of the petal in centimeters."),
                        html.Li("Species: Species of the iris flower (Setosa, Versicolor, Virginica).")
                    ]),
                    html.P("This dataset is widely used for classification tasks and is ideal for understanding relationships between flower measurements."),
                    
                    html.H4("Mtcars Dataset"),
                    html.Ul([
                        html.Li("mpg: Miles per gallon (fuel efficiency)."),
                        html.Li("cyl: Number of cylinders in the engine."),
                        html.Li("disp: Engine displacement in cubic inches."),
                        html.Li("hp: Gross horsepower."),
                        html.Li("drat: Rear axle ratio."),
                        html.Li("wt: Weight of the car in 1000 lbs."),
                        html.Li("qsec: Quarter-mile time."),
                        html.Li("vs: Engine shape (0 = V-shaped, 1 = straight)."),
                        html.Li("am: Transmission type (0 = automatic, 1 = manual)."),
                        html.Li("gear: Number of forward gears."),
                        html.Li("carb: Number of carburetors.")
                    ]),
                    html.P("This dataset is great for exploring relationships between car specifications and their performance."),
                    
                    html.H4("Diamonds Dataset"),
                    html.Ul([
                        html.Li("carat: Weight of the diamond."),
                        html.Li("cut: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)."),
                        html.Li("color: Diamond color (from D (best) to J (worst))."),
                        html.Li("clarity: Clarity of the diamond (e.g., IF = Internally Flawless, SI1 = Slightly Included)."),
                        html.Li("depth: Total depth percentage = z / mean(x, y)."),
                        html.Li("table: Width of the top of the diamond relative to the widest point."),
                        html.Li("price: Price in US dollars."),
                        html.Li("x: Length of the diamond in mm."),
                        html.Li("y: Width of the diamond in mm."),
                        html.Li("z: Depth of the diamond in mm.")
                    ]),
                    html.P("This dataset is perfect for analyzing how diamond characteristics influence pricing."),
                    
                    html.H4("Titanic Dataset"),
                    html.Ul([
                        html.Li("Class: Passenger class (1st, 2nd, 3rd)."),
                        html.Li("Sex: Gender of the passenger (male, female)."),
                        html.Li("Age: Age of the passenger."),
                        html.Li("Survived: Survival status (0 = did not survive, 1 = survived)."),
                        html.Li("Siblings/Spouses Aboard: Number of siblings or spouses aboard."),
                        html.Li("Parents/Children Aboard: Number of parents or children aboard."),
                        html.Li("Fare: Passenger fare paid.")
                    ]),
                    html.P("This dataset is widely used for predictive modeling and classification tasks, such as predicting survival status."),
                    
                    html.H3("How to Use These Datasets"),
                    html.P("These datasets are preloaded in the app and can be used for various types of analyses, including visualizations, statistical summaries, and machine learning tasks. "
                           "Use the app's interactive features to explore these datasets and gain insights."),
                    
                    html.H3("Let's get started! Happy Visualizing! ✨")
                ])
            ])
        ]),
        # Static Plots Tab
        dcc.Tab(label="Plots", children=[
            dbc.Row([
                # Left Panel
                dbc.Col([
                    html.Label("Choose Dataset:", style={"margin-top": "10px"}),
                    dcc.Dropdown(
                        id="static-dataset",
                        options=[
                            {"label": "Iris", "value": "iris"},
                            {"label": "Mtcars", "value": "mtcars"},
                            {"label": "Diamonds", "value": "diamonds"},
                            {"label": "Titanic", "value": "titanic"}
                        ],
                        value="iris"
                    ),
                    html.Label("Select Plot Type:", style={"margin-top": "10px"}),
                    dcc.Dropdown(
                        id="static-plot-type",
                        options=[
                            {"label": "Scatter Plot", "value": "scatter"},
                            {"label": "Line Plot", "value": "line"},
                            {"label": "Bar Chart", "value": "barchart"},
                            {"label": "Histogram", "value": "histogram"}
                        ],
                        value="scatter"
                    ),
                    html.Label("X-axis Variable:", style={"margin-top": "10px"}),
                    dcc.Dropdown(id="x-var", placeholder="Select X-axis variable"),
                    html.Label("Y-axis Variable:", style={"margin-top": "10px"}),
                    dcc.Dropdown(id="y-var", placeholder="Select Y-axis variable"),
                    html.Label("Color Grouping Variable:", style={"margin-top": "10px"}),
                    dcc.Dropdown(id="color-var", placeholder="Select color grouping variable"),
                    html.Label("Transparency (alpha):", style={"margin-top": "10px"}),
                    dcc.Slider(id="alpha-slider", min=0.1, max=1, step=0.1, value=0.7),
                    html.Div("Tips: Use appropriate variable types for different plot types. Relevant sliders will appear based on the selected plot type.",
                             style={"margin-top": "20px", "font-style": "italic"})
                ], width=3),
                # Right Panel
                dbc.Col([
                    dcc.Graph(id="static-plot"),
                    html.Div([
                        html.H4("Generated Python Code", style={"margin-top": "20px"}),
                        html.Pre(id="generated-code", style={"background-color": "#f8f9fa", "padding": "10px", "border-radius": "5px"})
                    ], style={"margin-top": "20px"})
                ], width=9)
            ])
        ]),
        # Add the Data Analysis Tab
        dcc.Tab(label="Data Analysis", children=[
            dbc.Container([
                dbc.Row([
                    # Left Panel
                    dbc.Col([
                        html.H4("Analysis Options", className="mb-3"),
                        html.Label("Select Dataset:"),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[
                                {"label": "Iris", "value": "iris"},
                                {"label": "Mtcars", "value": "mtcars"},
                                {"label": "Diamonds", "value": "diamonds"},
                                {"label": "Titanic", "value": "titanic"}
                            ],
                            value="iris",
                            clearable=False
                        ),
                        html.Label("Select Analysis:", className="mt-4"),
                        dcc.Dropdown(
                            id="analysis-dropdown",
                            options=[
                                {"label": "Description", "value": "description"},
                                {"label": "PairPlot", "value": "pairplot"},
                                {"label": "Distribution", "value": "distribution"}
                            ],
                            value="description",
                            clearable=False
                        )
                    ], width=3, style={"border-right": "1px solid #ccc", "padding-right": "15px"}),

                    # Right Panel
                    dbc.Col([
                        html.Div(id="analysis-output", className="mt-4")
                    ], width=9)
                ])
            ], fluid=True)
        ]),
        # Tab for Distributions
        dcc.Tab(label="Distributions", children=[
            # Normal Distribution Panel
            dbc.Row([
                dbc.Col([
                    html.H4("Normal Distribution"),
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
                    html.Pre(id="generated-code-normal", style={"background-color": "#f8f9fa", "padding": "10px"})
                ], width=8)
            ], className="my-4"),

            # Binomial Distribution Panel
            dbc.Row([
                dbc.Col([
                    html.H4("Binomial Distribution"),
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
                    html.Pre(id="generated-code-binomial", style={"background-color": "#f8f9fa", "padding": "10px"})
                ], width=8)
            ], className="my-4"),

            # Poisson Distribution Panel
            dbc.Row([
                dbc.Col([
                    html.H4("Poisson Distribution"),
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
                    html.Pre(id="generated-code-poisson", style={"background-color": "#f8f9fa", "padding": "10px"})
                ], width=8)
            ], className="my-4"),

            # Uniform Distribution Panel
            dbc.Row([
                dbc.Col([
                    html.H4("Uniform Distribution"),
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
                    html.Pre(id="generated-code-uniform", style={"background-color": "#f8f9fa", "padding": "10px"})
                ], width=8)
            ], className="my-4"),
        ])
    ])
])

# Callback to update variable options based on selected dataset
@app.callback(
    [Output("x-var", "options"),
     Output("y-var", "options"),
     Output("color-var", "options")],
    Input("static-dataset", "value")
)
def update_variable_options(dataset):
    data = {"iris": iris, "mtcars": mtcars, "diamonds": diamonds, "titanic": titanic}[dataset]
    options = [{"label": col, "value": col} for col in data.columns]
    return options, options, options

# Callback to render the static plot based on user input
@app.callback(
    Output("static-plot", "figure"),
    [Input("static-dataset", "value"),
     Input("static-plot-type", "value"),
     Input("x-var", "value"),
     Input("y-var", "value"),
     Input("color-var", "value"),
     Input("alpha-slider", "value")]
)
def update_static_plot(dataset, plot_type, x_var, y_var, color_var, alpha):
    data = {"iris": iris, "mtcars": mtcars, "diamonds": diamonds, "titanic": titanic}[dataset]
    if plot_type == "scatter" and x_var and y_var:
        fig = px.scatter(data, x=x_var, y=y_var, color=color_var, opacity=alpha)
    elif plot_type == "line" and x_var and y_var:
        fig = px.line(data, x=x_var, y=y_var, color=color_var)
    elif plot_type == "histogram" and x_var:
        fig = px.histogram(data, x=x_var, color=color_var, opacity=alpha)
    elif plot_type == "density" and x_var:
        fig = px.density_heatmap(data, x=x_var, y=y_var)
    elif plot_type == "barchart" and x_var:
        fig = px.bar(data, x=x_var, color=color_var)
    else:
        fig = px.scatter()  # Default empty plot
    return fig

# Callback to generate Python code based on user input
@app.callback(
    Output("generated-code", "children"),
    [Input("static-plot-type", "value"),
     Input("x-var", "value"),
     Input("y-var", "value"),
     Input("color-var", "value"),
     Input("alpha-slider", "value")]
)
def generate_python_code(plot_type, x_var, y_var, color_var, alpha):
    if plot_type == "scatter" and x_var and y_var:
        return f"""import plotly.express as px
fig = px.scatter(data, x='{x_var}', y='{y_var}', color='{color_var}', opacity={alpha})
fig.show()"""
    elif plot_type == "line" and x_var and y_var:
        return f"""import plotly.express as px
fig = px.line(data, x='{x_var}', y='{y_var}', color='{color_var}')
fig.show()"""
    elif plot_type == "histogram" and x_var:
        return f"""import plotly.express as px
fig = px.histogram(data, x='{x_var}', color='{color_var}', opacity={alpha})
fig.show()"""
    elif plot_type == "density" and x_var and y_var:
        return f"""import plotly.express as px
fig = px.density_heatmap(data, x='{x_var}', y='{y_var}')
fig.show()"""
    elif plot_type == "barchart" and x_var:
        return f"""import plotly.express as px
fig = px.bar(data, x='{x_var}', color='{color_var}')
fig.show()"""
    else:
        return "No valid plot selected."

# Callback to update the right panel based on dataset and analysis selection
@app.callback(
    Output("analysis-output", "children"),
    Input("dataset-dropdown", "value"),
    Input("analysis-dropdown", "value")
)
def update_analysis(dataset_name, analysis_type):
    # Load the selected dataset
    datasets = {
        "iris": iris,
        "mtcars": mtcars,
        "diamonds": diamonds,
        "titanic": titanic
    }
    df = datasets[dataset_name]

    if analysis_type == "description":  # Description
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Calculate missing values
        total_missing = df.isnull().sum().sum()
        percent_missing = (total_missing / (df.shape[0] * df.shape[1])) * 100
        missing_summary_table = df.isnull().sum().reset_index()
        missing_summary_table.columns = ["Column", "Missing Count"]
        missing_summary_table["Percentage"] = (
            missing_summary_table["Missing Count"] / df.shape[0] * 100
        )

        description = [
            html.H5("Dataset Description"),
            html.P(f"Name: {dataset_name.capitalize()}"),
            html.P(f"Dimension: {df.shape[0]} rows, {df.shape[1]} columns"),
            html.P(f"Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}"),
            html.P(f"Categorical Columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}"),
        
            # Dataset preview
            html.H5("Dataset Preview"),
            dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True),
        
            # Numeric columns statistics
            html.H5("Numeric Columns Summary"),
            dbc.Table.from_dataframe(
                df[numeric_cols].describe().transpose().reset_index().rename(
                    columns={
                        "index": "Column",
                        "25%": "Q1",
                        "50%": "Median",
                        "75%": "Q3"
                    }
                ),
                striped=True,
                bordered=True,
                hover=True
            ) if numeric_cols else html.P("No numeric columns available."),
        
            # Categorical columns summary
            html.H5("Categorical Columns Summary"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Column"), html.Th("Unique Values"), html.Th("Categories")])),
                html.Tbody([
                    html.Tr([
                        html.Td(col),
                        html.Td(len(df[col].unique())),
                        html.Td(", ".join(map(str, df[col].unique()[:10])) + ("..." if len(df[col].unique()) > 10 else ""))
                    ]) for col in categorical_cols
                ])
            ]) if categorical_cols else html.P("No categorical columns available."),
        
            # Missing values summary
            html.H5("Missing Summary"),
            html.P(f"Total Missing Values: {total_missing} ({percent_missing:.2f}%)"),
            dbc.Table.from_dataframe(
                missing_summary_table,
                striped=True,
                bordered=True,
                hover=True
            ) if total_missing > 0 else html.P("No missing values in the dataset.")
        ]
        return description

    elif analysis_type == "pairplot":  # PairPlot
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            def create_custom_pairplot(data):
                def corrfunc(x, y, **kwargs):
                    corr = np.corrcoef(x, y)[0][1]
                    ax = plt.gca()
                    ax.annotate(f'{corr:.2f}', xy=(0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center', fontsize=10)

                g = sns.PairGrid(data)
                g.map_lower(sns.scatterplot, alpha=0.6)
                g.map_diag(sns.kdeplot, fill=True)
                g.map_upper(corrfunc)
                plt.subplots_adjust(top=0.95)
                g.fig.suptitle(f"Pair Plot of {dataset_name.capitalize()}", fontsize=16)
                buffer = io.BytesIO()
                g.fig.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
                buffer.close()
                return f"data:image/png;base64,{encoded_image}"

            pairplot_image = create_custom_pairplot(df[numeric_cols])
            return html.Img(src=pairplot_image, style={"width": "100%", "height": "auto"})
        else:
            return html.P("Not enough numeric columns for pair plot.")

    elif analysis_type == "distribution":  # Distribution Analysis
        return html.Div([
            html.Label("Select Distribution Type:"),
            dcc.Dropdown(
                id="dist-plot-type-dropdown",
                options=[
                    {"label": "Histogram", "value": "histogram"},
                    {"label": "Density", "value": "density"},
                    {"label": "Boxplot", "value": "boxplot"},
                    {"label": "Barplot", "value": "barplot"}
                ],
                value="histogram",
                clearable=False
            ),
            html.Label("Select Variable:", className="mt-3"),
            dcc.Dropdown(id="dist-variable-dropdown", clearable=False),
            html.Div(id="dist-plot-output", className="mt-4")
        ])

    else:
        return html.P("Invalid analysis type selected.")

# Callback to update variable dropdown based on plot type and dataset
@app.callback(
    Output("dist-variable-dropdown", "options"),
    Input("dataset-dropdown", "value"),
    Input("dist-plot-type-dropdown", "value")
)
def update_variable_dropdown(dataset_name, plot_type):
    datasets = {
        "iris": iris,
        "mtcars": mtcars,
        "diamonds": diamonds,
        "titanic": titanic
    }
    df = datasets[dataset_name]
    if plot_type in ["histogram", "density", "boxplot"]:
        options = [{"label": col, "value": col} for col in df.select_dtypes(include=np.number).columns]
    elif plot_type == "barplot":
        options = [{"label": col, "value": col} for col in df.select_dtypes(include=["object", "category"]).columns]
    else:
        options = []
    return options

# Callback to generate the selected distribution plot
@app.callback(
    Output("dist-plot-output", "children"),
    Input("dataset-dropdown", "value"),
    Input("dist-plot-type-dropdown", "value"),
    Input("dist-variable-dropdown", "value")
)
def update_distribution_plot(dataset_name, plot_type, variable):
    datasets = {
        "iris": iris,
        "mtcars": mtcars,
        "diamonds": diamonds,
        "titanic": titanic
    }
    df = datasets[dataset_name]
    if not variable:
        return html.P("Please select a variable to plot.")
    if plot_type == "histogram":
        fig = px.histogram(df, x=variable, title=f"Histogram of {variable} in {dataset_name.capitalize()}")
    elif plot_type == "density":
        from scipy.stats import gaussian_kde
        import numpy as np
        
        # Data validation
        numeric_data = df[variable].dropna()
        if numeric_data.empty:
            return html.P("No valid data for the selected variable.")
        
        # Generate KDE
        kde = gaussian_kde(numeric_data)
        x_vals = np.linspace(numeric_data.min(), numeric_data.max(), 100)
        y_vals = kde(x_vals)

        # Plot KDE using Plotly
        fig = px.line(x=x_vals, y=y_vals, labels={'x': variable, 'y': 'Density'},
                    title=f"Density Plot of {variable} in {dataset_name.capitalize()}")
        fig.update_layout(height=600)
        return dcc.Graph(figure=fig, style={"height": "600px"})
    elif plot_type == "boxplot":
        fig = px.box(df, y=variable, title=f"Boxplot of {variable} in {dataset_name.capitalize()}")
    elif plot_type == "barplot":
        fig = px.bar(df, x=variable, title=f"Barplot of {variable} in {dataset_name.capitalize()}")
    else:
        return html.P("Invalid plot type selected.")
    return dcc.Graph(figure=fig, style={"height": "600px"})

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
    return (
        go.Figure(data=[go.Histogram(x=data)]),
        go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals))]),
        python_code
    )

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
    return (
        go.Figure(data=[go.Histogram(x=data)]),
        go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals))]),
        python_code
    )

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
    return (
        go.Figure(data=[go.Histogram(x=data)]),
        go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals))]),
        python_code
    )

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
    return (
        go.Figure(data=[go.Histogram(x=data)]),
        go.Figure(data=[go.Scatter(x=x_vals, y=kde(x_vals))]),
        python_code
    )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
