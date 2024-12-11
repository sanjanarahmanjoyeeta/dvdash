from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

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

# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

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
                            {"label": "Histogram", "value": "histogram"},
                            {"label": "Density Plot", "value": "density"},
                            {"label": "Violin Plot", "value": "violin"},
                            {"label": "Dot Plot", "value": "dotplot"},
                            {"label": "Bubble Chart", "value": "bubblechart"},
                            {"label": "Tree Map", "value": "treemap"},
                            {"label": "Heat Map", "value": "heatmap"}
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

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)