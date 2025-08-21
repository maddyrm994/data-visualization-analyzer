import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Visualization Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- App Title and Description ---
st.title("📊 Data Visualization Analyzer")
st.markdown("""
Welcome to the Data Visualization Analyzer! 

Upload your **cleaned** CSV or XLSX dataset, and this app will automatically:
1.  Identify your column types (numerical vs. categorical).
2.  Generate a variety of relevant visualizations.
3.  Provide actionable insights for each chart.
""")

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from uploaded file into a pandas DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_insights(df_sample_str, plot_type, columns):
    """Generates insights for a given plot using the GROQ API."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except FileNotFoundError:
        st.error("GROQ_API_KEY not found. Please add it to your Streamlit secrets.")
        return "Could not generate insights due to missing API key."
    except KeyError:
        st.error("GROQ_API_KEY not found in secrets.toml. Please add it.")
        return "Could not generate insights due to missing API key."
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert data analyst. Your task is to provide concise, actionable insights based on a data visualization.
    
    Here is the context:
    - Visualization Type: {plot_type}
    - Columns Used: {', '.join(columns)}
    - Data Sample (first 5 rows):
    {df_sample_str}

    Based on this information, please provide 3-4 bulleted insights revealed by this type of visualization. 
    Focus on potential patterns, outliers, distributions, or relationships. Be direct and clear.
    
    Example for a histogram of 'Age':
    - The age distribution is skewed to the right, indicating a larger population of younger individuals.
    - There is a noticeable peak in the 20-30 age bracket, which could be the primary demographic.
    - There are very few individuals over the age of 70, suggesting a potential outlier or a specific characteristic of this dataset.
    
    Now, provide the insights for the given visualization.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192", # Using a fast and capable model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.warning(f"Could not generate insights due to an API error: {e}")
        return "Insights could not be generated at this time."

# --- Main App Logic ---

# 1. File Uploader in the Sidebar
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file", 
        type=["csv", "xlsx"],
        help="Upload a cleaned dataset for analysis. The app works best with data that has clear column headers and no missing values."
    )

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        st.header("📄 Data Preview")
        st.dataframe(df.head())

        # Identify column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.sidebar.header("2. Data Summary")
        st.sidebar.markdown(f"**{df.shape[0]}** rows")
        st.sidebar.markdown(f"**{df.shape[1]}** columns")
        st.sidebar.markdown(f"**{len(numeric_cols)}** numerical columns")
        st.sidebar.markdown(f"**{len(categorical_cols)}** categorical columns")

        st.divider()
        st.header("🎨 Visualizations & Insights")
        st.info("The app is now generating plots and analyzing them. This may take a moment.")
        
        # --- Visualization Generation ---

        # Section 1: Univariate Analysis (Single Column)
        st.subheader("1. Univariate Analysis (Analyzing Single Columns)")
        
        selected_column = st.selectbox("Select a column for univariate analysis:", df.columns)
        
        if selected_column:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"#### Analysis of `{selected_column}`")
                # Plotting logic based on column type
                if selected_column in numeric_cols:
                    # Histogram
                    fig_hist = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Box Plot
                    fig_box = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                    st.plotly_chart(fig_box, use_container_width=True)
                
                elif selected_column in categorical_cols:
                     # Bar Chart
                    fig_bar = px.bar(df, x=selected_column, title=f"Count of {selected_column}")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.write("This column type is not supported for univariate analysis.")

            with col2:
                # Generate Insights for the selected plot
                with st.spinner(f"🤖 Analyzing `{selected_column}`..."):
                    plot_type = "Histogram and Box Plot" if selected_column in numeric_cols else "Bar Chart"
                    df_sample_str = df[[selected_column]].head().to_string()
                    insights = get_insights(df_sample_str, plot_type, [selected_column])
                    
                    st.markdown("#### 🧠 AI-Powered Insights")
                    st.markdown(insights)

        st.divider()
        # Section 2: Bivariate Analysis (Two Columns)
        st.subheader("2. Bivariate Analysis (Analyzing Relationships Between Two Columns)")
        
        if len(df.columns) > 1:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select the X-axis:", df.columns, index=0)
            with col2:
                # Filter out the x-axis column from y-axis options
                y_axis_options = [col for col in df.columns if col != x_axis]
                y_axis = st.selectbox("Select the Y-axis:", y_axis_options, index=min(1, len(y_axis_options)-1) if y_axis_options else 0)

            if x_axis and y_axis:
                plot_container, insight_container = st.columns([1, 1])

                with plot_container:
                    st.markdown(f"#### Analysis of `{x_axis}` vs. `{y_axis}`")
                    # Determine plot type
                    if x_axis in numeric_cols and y_axis in numeric_cols:
                        plot_type = "Scatter Plot"
                        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs. {x_axis}")
                    elif x_axis in categorical_cols and y_axis in numeric_cols:
                        plot_type = "Box Plot"
                        fig = px.box(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                    elif x_axis in numeric_cols and y_axis in categorical_cols:
                        plot_type = "Box Plot"
                        fig = px.box(df, x=y_axis, y=x_axis, title=f"{x_axis} by {y_axis}")
                    elif x_axis in categorical_cols and y_axis in categorical_cols:
                        plot_type = "Density Heatmap"
                        # Create a crosstab for the heatmap
                        crosstab = pd.crosstab(df[x_axis], df[y_axis])
                        fig = px.imshow(crosstab, text_auto=True, title=f"Heatmap of {x_axis} vs. {y_axis}")
                    else:
                        st.warning("Could not determine a suitable plot for the selected combination.")
                        fig = None
                        
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                with insight_container:
                    if fig:
                        with st.spinner(f"🤖 Analyzing relationship between `{x_axis}` and `{y_axis}`..."):
                            df_sample_str = df[[x_axis, y_axis]].head().to_string()
                            insights = get_insights(df_sample_str, plot_type, [x_axis, y_axis])
                            st.markdown("#### 🧠 AI-Powered Insights")
                            st.markdown(insights)
        else:
            st.warning("Bivariate analysis requires at least two columns in the dataset.")

else:
    st.info("Awaiting your dataset. Please upload a file using the sidebar to begin.")