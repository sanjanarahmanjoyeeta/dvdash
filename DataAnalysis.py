import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for Dash compatibility
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

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
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

# Layout
app.layout = dbc.Container([
    dcc.Tabs([
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
        ])
    ])
])

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

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)