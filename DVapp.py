import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Load datasets
@st.cache_data
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

iris = load_dataset("data/iris.csv")
mtcars = load_dataset("data/mtcars.csv")
diamonds = load_dataset("data/diamonds.csv")
titanic = load_dataset("data/titanic.csv")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Plots", "Data Analysis", "Distributions"])

# Precompute data for distributions
means = np.arange(-5, 6, 1)
std_devs = np.arange(1, 11, 1)
precomputed_data_normal = {(mean, sd): np.random.normal(loc=mean, scale=sd, size=1000) for mean in means for sd in std_devs}
precomputed_data_binomial = {(n, round(p, 1)): np.random.binomial(n=n, p=p, size=1000) for n in range(1, 21) for p in np.arange(0.1, 1.1, 0.1)}
precomputed_data_poisson = {lam: np.random.poisson(lam=lam, size=1000) for lam in range(1, 21)}
precomputed_data_uniform = {(a, b): np.random.uniform(low=a, high=b, size=1000) for a in range(0, 11) for b in range(1, 12) if a < b}

# Home Page
if page == "Home":
    st.title("Welcome to Your Data Visualization Playground!")
    st.markdown("#### Created by Sanjana Rahman")
    st.image("assets/logo.png", use_column_width=True)
    st.write("Explore data visualizations, analyses, and distributions with preloaded datasets.")
    st.subheader("Datasets Overview")
    st.markdown("""
    - **Iris Dataset**: Flower measurements for classification tasks.
    - **Mtcars Dataset**: Car specifications for performance analysis.
    - **Diamonds Dataset**: Diamond features influencing pricing.
    - **Titanic Dataset**: Predictive modeling on passenger survival.
    """)

# Plots Page
elif page == "Plots":
    st.title("Interactive Plots")
    dataset_name = st.selectbox("Choose Dataset", ["Iris", "Mtcars", "Diamonds", "Titanic"])
    plot_type = st.selectbox("Choose Plot Type", ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Density Plot"])
    df = {"Iris": iris, "Mtcars": mtcars, "Diamonds": diamonds, "Titanic": titanic}[dataset_name]
    
    x_var = st.selectbox("Select X-axis Variable", df.columns)
    y_var = st.selectbox("Select Y-axis Variable", df.columns) if plot_type != "Histogram" else None
    color_var = st.selectbox("Select Color Grouping Variable", [None] + list(df.columns))
    alpha = st.slider("Set Transparency (Alpha)", 0.1, 1.0, 0.7)

    if plot_type == "Scatter Plot" and x_var and y_var:
        fig = px.scatter(df, x=x_var, y=y_var, color=color_var, opacity=alpha)
        st.plotly_chart(fig)
    elif plot_type == "Line Plot" and x_var and y_var:
        fig = px.line(df, x=x_var, y=y_var, color=color_var)
        st.plotly_chart(fig)
    elif plot_type == "Bar Chart" and x_var:
        fig = px.bar(df, x=x_var, y=y_var, color=color_var, opacity=alpha)
        st.plotly_chart(fig)
    elif plot_type == "Histogram" and x_var:
        fig = px.histogram(df, x=x_var, nbins=20, color=color_var, opacity=alpha)
        st.plotly_chart(fig)

# Data Analysis Page
elif page == "Data Analysis":
    st.title("Data Analysis")
    dataset_name = st.selectbox("Select Dataset", ["Iris", "Mtcars", "Diamonds", "Titanic"])
    df = {"Iris": iris, "Mtcars": mtcars, "Diamonds": diamonds, "Titanic": titanic}[dataset_name]
    analysis_type = st.selectbox("Choose Analysis Type", ["Summary", "PairPlot", "Distributions"])

    if analysis_type == "Summary":
        st.write(df.describe())
    elif analysis_type == "PairPlot":
        st.write("Generating pair plot...")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols])
            st.pyplot()

# Distributions Page
elif page == "Distributions":
    st.title("Explore Distributions")
    dist_type = st.selectbox("Select Distribution Type", ["Normal", "Binomial", "Poisson", "Uniform"])
    
    if dist_type == "Normal":
        mean = st.slider("Mean", -5, 5, 0)
        sd = st.slider("Standard Deviation", 1, 10, 1)
        data = precomputed_data_normal[(mean, sd)]
        st.hist(data, bins=30)
