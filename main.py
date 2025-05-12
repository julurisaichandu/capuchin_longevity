import streamlit as st
import pandas as pd
import numpy as np
# import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
import base64
from datetime import datetime, timedelta
import os
import re

# Set page config
st.set_page_config(
    page_title="Capuchin Health Dashboard",
    page_icon="🐒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define reference ranges for capuchin health parameters
# Based on literature values for adult capuchin monkeys
REFERENCE_RANGES = {
    'RBC': {'min': 4.5, 'max': 7.0, 'units': 'M/µL'},
    'Hematocrit': {'min': 35.0, 'max': 50.0, 'units': '%'}, 
    'Hemoglobin': {'min': 12.0, 'max': 17.0, 'units': 'g/dL'},
    'MCV': {'min': 65.0, 'max': 85.0, 'units': 'fL'},
    'MCH': {'min': 20.0, 'max': 28.0, 'units': 'pg'},
    'MCHC': {'min': 30.0, 'max': 36.0, 'units': 'g/dL'},
    'WBC': {'min': 3.5, 'max': 15.0, 'units': 'K/µL'},
    'Neutrophils': {'min': 1.5, 'max': 8.0, 'units': 'K/µL'},
    'Lymphocytes': {'min': 0.8, 'max': 4.0, 'units': 'K/µL'},
    'Monocytes': {'min': 0.1, 'max': 0.8, 'units': 'K/µL'},
    'Eosinophils': {'min': 0.0, 'max': 0.6, 'units': 'K/µL'},
    'Basophils': {'min': 0.0, 'max': 0.2, 'units': 'K/µL'},
    'Platelets': {'min': 200.0, 'max': 600.0, 'units': 'K/µL'},
    'Glucose': {'min': 60.0, 'max': 140.0, 'units': 'mg/dL'},
    'BUN': {'min': 8.0, 'max': 30.0, 'units': 'mg/dL'},
    'Creatinine': {'min': 0.4, 'max': 1.0, 'units': 'mg/dL'},
    'Phosphorus': {'min': 2.5, 'max': 6.0, 'units': 'mg/dL'},
    'Calcium': {'min': 8.0, 'max': 11.0, 'units': 'mg/dL'},
    'Sodium': {'min': 140.0, 'max': 155.0, 'units': 'mmol/L'},
    'Potassium': {'min': 3.0, 'max': 5.0, 'units': 'mmol/L'},
    'Chloride': {'min': 100.0, 'max': 120.0, 'units': 'mmol/L'},
    'Total Protein': {'min': 5.5, 'max': 8.0, 'units': 'g/dL'},
    'Albumin': {'min': 3.0, 'max': 5.0, 'units': 'g/dL'},
    'Globulin': {'min': 1.5, 'max': 3.5, 'units': 'g/dL'},
    'ALT': {'min': 10.0, 'max': 50.0, 'units': 'U/L'},
    'AST': {'min': 10.0, 'max': 50.0, 'units': 'U/L'},
    'ALP': {'min': 30.0, 'max': 120.0, 'units': 'U/L'},
    'GGT': {'min': 10.0, 'max': 70.0, 'units': 'U/L'},
    'Bilirubin Total': {'min': 0.0, 'max': 0.3, 'units': 'mg/dL'},
    'Cholesterol': {'min': 100.0, 'max': 250.0, 'units': 'mg/dL'}
}

# Age-specific adjustments (percentage adjustments to apply to reference ranges)
AGE_ADJUSTMENTS = {
    'juvenile': {  # 0-5 years
        'RBC': {'min': 0.90, 'max': 1.10},
        'Hematocrit': {'min': 0.90, 'max': 1.05},
        'Hemoglobin': {'min': 0.90, 'max': 1.05},
        'ALP': {'min': 1.20, 'max': 1.50},  # Higher in juveniles
        'Glucose': {'min': 0.95, 'max': 1.10},
        'Phosphorus': {'min': 1.10, 'max': 1.20},  # Higher in juveniles
    },
    'geriatric': {  # 25+ years
        'RBC': {'min': 0.90, 'max': 0.95},  # Lower in geriatric
        'Hematocrit': {'min': 0.90, 'max': 0.95},  # Lower in geriatric
        'Hemoglobin': {'min': 0.90, 'max': 0.95},  # Lower in geriatric
        'Creatinine': {'min': 1.05, 'max': 1.10},  # Higher in geriatric
        'BUN': {'min': 1.05, 'max': 1.15},  # Higher in geriatric
    }
}

# Function to get age category
def get_age_category(age):
    if age is None or pd.isna(age):
        return 'adult'  # Default to adult if age is unknown

    try:
        age = float(age)
    except (ValueError, TypeError):
        return 'adult'  # If age can't be converted, assume adult

    if age < 5:
        return 'juvenile'
    elif age > 25:
        return 'geriatric'
    else:
        return 'adult'


# Function to get adjusted reference range based on age
def get_reference_range(parameter, age=None):
    if parameter not in REFERENCE_RANGES:
        return None
    
    base_range = REFERENCE_RANGES[parameter].copy()
    
    # Apply age-specific adjustments if applicable
    age_category = get_age_category(age)
    if age_category != 'adult' and parameter in AGE_ADJUSTMENTS.get(age_category, {}):
        adjustments = AGE_ADJUSTMENTS[age_category][parameter]
        base_range['min'] *= adjustments['min']
        base_range['max'] *= adjustments['max']
    
    return base_range

# Function to check if a value is outside reference range
def is_outside_range(value, parameter, age=None):
    range_info = get_reference_range(parameter, age)
    if range_info is None or pd.isna(value):
        return False
    
    # Handle string values like '<0.1'
    if isinstance(value, str):
        if value.startswith('<'):
            value = float(value[1:]) - 0.01  # Just below the threshold
        elif value.startswith('>'):
            value = float(value[1:]) + 0.01  # Just above the threshold
        else:
            try:
                value = float(value)
            except:
                return False
    
    return value < range_info['min'] or value > range_info['max']

# Function to get the direction of deviation (High/Low)
def get_deviation(value, parameter, age=None):
    range_info = get_reference_range(parameter, age)
    if range_info is None or pd.isna(value):
        return ''
    
    # Handle string values
    if isinstance(value, str):
        if value.startswith('<'):
            value = float(value[1:]) - 0.01
        elif value.startswith('>'):
            value = float(value[1:]) + 0.01
        else:
            try:
                value = float(value)
            except:
                return ''
    
    if value < range_info['min']:
        return 'Low'
    elif value > range_info['max']:
        return 'High'
    else:
        return 'Normal'

# Data loading and processing functions
def clean_value(value):
    """Convert value to float if possible"""
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # Handle string values like '<0.1'
    if isinstance(value, str):
        if value.startswith('<'):
            # Return a value just below the threshold
            try:
                return float(value[1:]) - 0.01
            except:
                return np.nan
        elif value.startswith('>'):
            # Return a value just above the threshold
            try:
                return float(value[1:]) + 0.01
            except:
                return np.nan
        else:
            # Try to convert to float
            try:
                return float(value)
            except:
                return np.nan
    
    return np.nan

def preprocess_data(df):
    """Clean and preprocess the uploaded data"""
    # Ensure date column is in datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Clean up result values
    if 'result' in df.columns:
        df['result_numeric'] = df['result'].apply(clean_value)
    
    return df

def pivot_data_for_clustering(df):
    """Pivot data into a format suitable for clustering"""
    # Create a pivot table with animal IDs as rows and tests as columns
    pivot_df = df.pivot_table(
        index=['id', 'date'],
        columns='test',
        values='result_numeric',
        aggfunc='first'
    ).reset_index()
    
    # Get the most recent test for each animal
    most_recent = pivot_df.sort_values('date').groupby('id').last().reset_index()
    
    return most_recent

def perform_clustering(df, n_clusters=3):
    """Perform hierarchical clustering on the data"""
    # Get only the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'age_years']]
    
    # Handle missing values
    analysis_df = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(analysis_df)
    
    # Perform hierarchical clustering
    Z = linkage(X_scaled, method='ward')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Add cluster information to the original dataframe
    result_df = df.copy()
    result_df['cluster'] = clusters
    
    return result_df, Z

def perform_pca(df):
    """Perform PCA on the data"""
    # Get only the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'age_years', 'cluster']]
    
    # Handle missing values
    analysis_df = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(analysis_df)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create a dataframe with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=['PC1', 'PC2']
    )
    
    # Add the animal ID and cluster information
    pca_df['id'] = df['id'].values
    if 'cluster' in df.columns:
        pca_df['cluster'] = df['cluster'].values
    
    # Feature importance for each principal component
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=numeric_cols
    )
    
    return pca_df, loadings, pca.explained_variance_ratio_

# UI Functions
def create_download_link(df, filename):
    """Creates a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def plot_time_series(df, animal_id, parameter, age=None):
    """Create a time series plot for a parameter with reference ranges"""
    # Filter data for the specific animal and parameter
    animal_data = df[(df['id'] == animal_id) & (df['test'] == parameter)]
    
    if len(animal_data) == 0:
        st.warning(f"No data available for {parameter} for this animal.")
        return None
    
    # Sort by date
    animal_data = animal_data.sort_values('date')
    
    # Get the reference range
    ref_range = get_reference_range(parameter, age)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the time series
    fig.add_trace(go.Scatter(
        x=animal_data['date'],
        y=animal_data['result_numeric'],
        mode='lines+markers',
        name=parameter,
        line=dict(color='royalblue', width=2)
    ))
    
    # Add reference range if available
    if ref_range:
        # Add reference range as a shaded area
        fig.add_shape(
            type="rect",
            x0=animal_data['date'].min(),
            x1=animal_data['date'].max(),
            y0=ref_range['min'],
            y1=ref_range['max'],
            fillcolor="rgba(0,255,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # Add reference lines
        fig.add_shape(
            type="line",
            x0=animal_data['date'].min(),
            x1=animal_data['date'].max(),
            y0=ref_range['min'],
            y1=ref_range['min'],
            line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=animal_data['date'].min(),
            x1=animal_data['date'].max(),
            y0=ref_range['max'],
            y1=ref_range['max'],
            line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash")
        )
    
    # Update layout
    fig.update_layout(
        title=f"{parameter} Over Time for {animal_id}",
        xaxis_title="Date",
        yaxis_title=f"{parameter} ({ref_range['units']})" if ref_range else parameter,
        template="plotly_white"
    )
    
    return fig

def plot_parameter_distribution(df, parameter):
    """Create a distribution plot for a parameter across all animals"""
    # Filter data for the specific parameter
    param_data = df[df['test'] == parameter]
    
    if len(param_data) == 0:
        st.warning(f"No data available for {parameter}.")
        return None
    
    # Get the reference range
    ref_range = get_reference_range(parameter)
    
    # Create the figure
    fig = px.histogram(
        param_data, 
        x='result_numeric', 
        color='id', 
        barmode='overlay',
        opacity=0.7,
        marginal="box"
    )
    
    # Add reference range if available
    if ref_range:
        # Add reference range as vertical lines
        fig.add_shape(
            type="line",
            x0=ref_range['min'],
            x1=ref_range['min'],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=1, dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=ref_range['max'],
            x1=ref_range['max'],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=1, dash="dash")
        )
        
        # Add reference range annotation
        fig.add_annotation(
            x=(ref_range['min'] + ref_range['max']) / 2,
            y=1,
            text=f"Reference Range: {ref_range['min']} - {ref_range['max']} {ref_range['units']}",
            showarrow=False,
            yref="paper"
        )
    
    # Update layout
    fig.update_layout(
        title=f"{parameter} Distribution Across All Animals",
        xaxis_title=f"{parameter} ({ref_range['units']})" if ref_range else parameter,
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig

def plot_clustered_data(pca_df, loadings, explained_variance):
    """Plot the results of the clustering analysis using PCA"""
    # Create figure with two subplots - PCA scatter plot and loadings plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA Scatter Plot", "Feature Importance"))
    
    # Add PCA scatter plot
    if 'cluster' in pca_df.columns:
        # Add scatter plot colored by cluster
        scatter = px.scatter(
            pca_df, 
            x='PC1', 
            y='PC2', 
            color='cluster',
            text='id',
            hover_data=['id', 'cluster']
        )
        
        for trace in scatter.data:
            fig.add_trace(trace, row=1, col=1)
    else:
        # Add scatter plot without cluster information
        scatter = px.scatter(
            pca_df, 
            x='PC1', 
            y='PC2', 
            text='id',
            hover_data=['id']
        )
        
        for trace in scatter.data:
            fig.add_trace(trace, row=1, col=1)
    
    # Add loadings plot (top 10 features by absolute value for PC1)
    top_loadings = loadings['PC1'].abs().sort_values(ascending=False).head(10).index
    loadings_sorted = loadings.loc[top_loadings, 'PC1'].sort_values()
    
    fig.add_trace(
        go.Bar(
            y=loadings_sorted.index,
            x=loadings_sorted.values,
            orientation='h',
            name='PC1'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="PCA Analysis Results",
        template="plotly_white",
        height=600,
        annotations=[
            dict(
                x=0.25, y=1.05,
                text=f"PC1 ({explained_variance[0]:.2%} variance explained)",
                showarrow=False,
                xref="paper", yref="paper"
            ),
            dict(
                x=0.25, y=1.0,
                text=f"PC2 ({explained_variance[1]:.2%} variance explained)",
                showarrow=False,
                xref="paper", yref="paper"
            )
        ]
    )
    
    return fig

def plot_health_radar(df, animal_id):
    """Create a radar chart for the most recent health parameters of an animal"""
    # Get the most recent data for the animal
    animal_data = df[df['id'] == animal_id]
    
    if len(animal_data) == 0:
        st.warning(f"No data available for {animal_id}.")
        return None
    
    # Get the most recent date
    most_recent_date = animal_data['date'].max()
    recent_data = animal_data[animal_data['date'] == most_recent_date]
    
    # Convert to a format suitable for radar chart
    radar_data = []
    labels = []
    reference_mins = []
    reference_maxes = []
    
    for _, row in recent_data.iterrows():
        parameter = row['test']
        if parameter in REFERENCE_RANGES:
            ref_range = get_reference_range(parameter, animal_data['age_years'].iloc[0])
            labels.append(parameter)
            
            # Normalize the value between 0 and 1 for the radar chart
            value = row['result_numeric']
            min_val = ref_range['min']
            max_val = ref_range['max']
            range_size = max_val - min_val
            
            # Calculate normalized value (0.5 is middle of reference range)
            if pd.isna(value):
                normalized = 0.5  # Default to middle if unknown
            else:
                normalized = (value - min_val) / range_size if range_size > 0 else 0.5
            
            radar_data.append(normalized)
            reference_mins.append(0)  # Normalized reference min is always 0
            reference_maxes.append(1)  # Normalized reference max is always 1
    
    # Create the radar chart
    fig = go.Figure()
    
    # Add reference range
    fig.add_trace(go.Scatterpolar(
        r=reference_maxes,
        theta=labels,
        fill='toself',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(color='rgba(0,255,0,0.5)'),
        name='Reference Range'
    ))
    
    # Add animal data
    fig.add_trace(go.Scatterpolar(
        r=radar_data,
        theta=labels,
        fill='toself',
        line=dict(color='royalblue'),
        name=animal_id
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Health Parameters for {animal_id} ({most_recent_date.date()})",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2]
            )
        ),
        showlegend=True
    )
    
    return fig

def create_dashboard():
    """Main function to create the dashboard"""
    st.title("Capuchin Monkey Health Dashboard")
    
    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")
    
    # File uploader
    st.sidebar.subheader("Upload Data")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files with monkey health data", 
        type="csv", 
        accept_multiple_files=True
    )
    
    # Show a sample of expected CSV format
    with st.sidebar.expander("Expected CSV Format"):
        st.write("""
        Your CSV should include these columns:
        - `date`: Date of the test (YYYY-MM-DD)
        - `test`: Test name (e.g., 'RBC', 'Glucose')
        - `result`: Test result value
        - `units`: Units for the result (e.g., 'g/dL')
        - `age_years`: Age of the animal in years (optional)
        - `sex`: Sex of the animal ('Male' or 'Female')
        - `id`: Unique identifier for the animal
        """)
    
    # Initialize session state if not done already
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Load the data if files are uploaded
    if uploaded_files:
        # Read and concatenate all uploaded files
        dfs = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                st.sidebar.success(f"Successfully loaded {file.name}")
            except Exception as e:
                st.sidebar.error(f"Error loading {file.name}: {e}")
        
        if dfs:
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Preprocess the data
            st.session_state.data = preprocess_data(combined_df)
            
            # Show data info
            st.sidebar.info(f"Loaded data for {st.session_state.data['id'].nunique()} animals with {len(st.session_state.data)} records.")
    
    # Main content area with tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Individual Analysis", 
        "Population Overview", 
        "Comparative Analysis", 
        "Pattern Detection"
    ])
    
    # If data is loaded, populate the dashboard
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Tab 1: Individual Animal Analysis
        with tab1:
            st.header("Individual Animal Analysis")
            
            # Select an animal to analyze
            animal_options = data['id'].unique().tolist()
            selected_animal = st.selectbox("Select Animal", animal_options)
            
            # Get data for the selected animal
            animal_data = data[data['id'] == selected_animal]
            
            # Get unique tests for this animal
            tests = animal_data['test'].unique()
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sex = animal_data['sex'].iloc[0] if 'sex' in animal_data.columns else 'Unknown'
                st.metric("Sex", sex)
            
            with col2:
                age = animal_data['age_years'].iloc[0] if 'age_years' in animal_data.columns and not pd.isna(animal_data['age_years'].iloc[0]) else 'Unknown'
                st.metric("Age (years)", age)
            
            with col3:
                records = len(animal_data)
                st.metric("Number of Records", records)
            
            # Display the most recent test results
            st.subheader("Most Recent Test Results")
            
            # Get the most recent date
            most_recent_date = animal_data['date'].max()
            recent_data = animal_data[animal_data['date'] == most_recent_date]
            
            # Create a dataframe for display
            display_df = recent_data[['test', 'result', 'units']].copy()
            
            # Add reference range and status
            display_df['reference_range'] = display_df['test'].apply(
                lambda x: f"{get_reference_range(x, animal_data['age_years'].iloc[0])['min']} - {get_reference_range(x, animal_data['age_years'].iloc[0])['max']}" if x in REFERENCE_RANGES else 'N/A'
            )
            
            display_df['status'] = display_df.apply(
                lambda row: get_deviation(row['result'], row['test'], animal_data['age_years'].iloc[0]), 
                axis=1
            )
            
            # Use custom styling to highlight abnormal values
            def highlight_status(val):
                if val == 'High':
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                elif val == 'Low':
                    return 'background-color: rgba(0, 0, 255, 0.2)'
                else:
                    return ''
            
            # Display the styled dataframe
            st.dataframe(
                display_df.style.applymap(highlight_status, subset=['status']), 
                height=400
            )
            
            # Time series plots for selected parameters
            st.subheader("Parameter Trends Over Time")
            
            # Allow selection of parameters to view
            selected_params = st.multiselect(
                "Select Parameters to Plot",
                options=tests,
                default=tests[0] if len(tests) > 0 else None
            )
            
            # Create time series plots for selected parameters
            for param in selected_params:
                fig = plot_time_series(data, selected_animal, param, age)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Health radar chart
            st.subheader("Health Parameter Radar")
            radar_fig = plot_health_radar(data, selected_animal)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            
        # Tab 2: Population Overview
        with tab2:
            st.header("Population Health Overview")
            
            # Display demographics
            st.subheader("Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sex distribution
                if 'sex' in data.columns:
                    sex_counts = data.drop_duplicates('id')['sex'].value_counts().reset_index()
                    sex_counts.columns = ['Sex', 'Count']
                    
                    fig = px.pie(sex_counts, values='Count', names='Sex', title='Sex Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sex data not available.")
            
            with col2:
                # Age distribution
                if 'age_years' in data.columns and not data['age_years'].isna().all():
                    age_data = data.drop_duplicates('id')
                    
                    fig = px.histogram(
                        age_data, 
                        x='age_years',
                        nbins=10,
                        title='Age Distribution'
                    )
                    fig.update_layout(xaxis_title="Age (years)", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Age data not available.")
            
            # Overall health metrics
            st.subheader("Population Health Metrics")
            
            # Get the most recent record for each animal
            most_recent_records = data.sort_values('date').groupby('id').last().reset_index()
            
            # Calculate the percentage of abnormal values for key parameters
            key_params = ['RBC', 'WBC', 'Hemoglobin', 'Glucose', 'ALT', 'Creatinine']
            abnormal_counts = []
            
            for param in key_params:
                param_data = most_recent_records[most_recent_records['test'] == param]
                if len(param_data) > 0:
                    total = len(param_data)
                    abnormal = sum(param_data.apply(
                        lambda row: is_outside_range(row['result_numeric'], row['test'], row['age_years'] if 'age_years' in row else None),
                        axis=1
                    ))
                    abnormal_counts.append({
                        'Parameter': param,
                        'Normal': total - abnormal,
                        'Abnormal': abnormal,
                        'Total': total,
                        'Percent Abnormal': (abnormal / total) * 100 if total > 0 else 0
                    })
            
            if abnormal_counts:
                abnormal_df = pd.DataFrame(abnormal_counts)
                
                # Create a stacked bar chart
                fig = px.bar(
                    abnormal_df,
                    x='Parameter',
                    y=['Normal', 'Abnormal'],
                    title='Normal vs. Abnormal Values in Population',
                    labels={'value': 'Count', 'variable': 'Status'},
                    barmode='stack'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for population health metrics.")
        
        # Tab 3: Comparative Analysis
        with tab3:
            st.header("Comparative Analysis")
            
            # Select parameters for comparison
            all_tests = data['test'].unique()
            compare_param = st.selectbox("Select Parameter to Compare", all_tests)
            
            # Create a distribution plot
            dist_fig = plot_parameter_distribution(data, compare_param)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
            
            # Allow comparison between specific animals
            st.subheader("Compare Specific Animals")
            
            # Select animals to compare
            compare_animals = st.multiselect(
                "Select Animals to Compare",
                options=animal_options,
                default=animal_options[:min(2, len(animal_options))]
            )
            
            if len(compare_animals) > 0:
                # Select test for comparison
                compare_test = st.selectbox("Select Test for Comparison", all_tests, key="compare_test")
                
                # Create comparison plot
                compare_data = data[data['id'].isin(compare_animals) & (data['test'] == compare_test)]
                
                if len(compare_data) > 0:
                    fig = px.line(
                        compare_data,
                        x='date',
                        y='result_numeric',
                        color='id',
                        markers=True,
                        title=f"{compare_test} Comparison"
                    )
                    
                    # Add reference range if available
                    ref_range = get_reference_range(compare_test)
                    if ref_range:
                        # Add reference range as a shaded area
                        fig.add_shape(
                            type="rect",
                            x0=compare_data['date'].min(),
                            x1=compare_data['date'].max(),
                            y0=ref_range['min'],
                            y1=ref_range['max'],
                            fillcolor="rgba(0,255,0,0.1)",
                            line=dict(width=0),
                            layer="below"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data available for {compare_test} for the selected animals.")
            
            # Add time-normalized view to compare animals at same age
            if 'age_years' in data.columns and not data['age_years'].isna().all():
                st.subheader("Age-Based Comparison")
                
                # Select test for age comparison
                age_compare_test = st.selectbox("Select Test", all_tests, key="age_compare_test")
                
                # Create age-based comparison plot
                age_compare_data = data[data['test'] == age_compare_test]
                
                if len(age_compare_data) > 0:
                    fig = px.scatter(
                        age_compare_data,
                        x='age_years',
                        y='result_numeric',
                        color='id',
                        title=f"{age_compare_test} by Age"
                    )
                    
                    # Add reference range if available
                    ref_range = get_reference_range(age_compare_test)
                    if ref_range:
                        # Add reference range as horizontal lines
                        fig.add_shape(
                            type="line",
                            x0=age_compare_data['age_years'].min(),
                            x1=age_compare_data['age_years'].max(),
                            y0=ref_range['min'],
                            y1=ref_range['min'],
                            line=dict(color="red", width=1, dash="dash")
                        )
                        fig.add_shape(
                            type="line",
                            x0=age_compare_data['age_years'].min(),
                            x1=age_compare_data['age_years'].max(),
                            y0=ref_range['max'],
                            y1=ref_range['max'],
                            line=dict(color="red", width=1, dash="dash")
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data available for {age_compare_test} with age information.")
        
        # Tab 4: Pattern Detection
        with tab4:
            st.header("Pattern Detection")
            
            # Only proceed if we have enough animals
            if data['id'].nunique() >= 2:
                # Prepare data for clustering
                st.subheader("Health Pattern Clustering")
                
                try:
                    # Pivot the data for clustering
                    pivoted_data = pivot_data_for_clustering(data)
                    
                    # Select number of clusters
                    n_clusters = st.slider("Number of Clusters", 2, min(10, data['id'].nunique()), 3)
                    
                    # Perform clustering
                    clustered_data, linkage_matrix = perform_clustering(pivoted_data, n_clusters)
                    
                    # Display dendrogram
                    st.subheader("Hierarchical Clustering Dendrogram")
                    
                    plt.figure(figsize=(10, 6))
                    plt.title('Hierarchical Clustering Dendrogram')
                    plt.xlabel('Animal ID')
                    plt.ylabel('Distance')
                    
                    # Create dendrogram
                    dendrogram_plot = dendrogram(
                        linkage_matrix,
                        leaf_rotation=90,
                        leaf_font_size=10,
                        labels=pivoted_data['id'].values
                    )
                    
                    st.pyplot(plt)
                    
                    # Perform PCA and visualize clusters
                    st.subheader("Principal Component Analysis")
                    
                    pca_df, loadings, explained_variance = perform_pca(clustered_data)
                    
                    pca_fig = plot_clustered_data(pca_df, loadings, explained_variance)
                    st.plotly_chart(pca_fig, use_container_width=True)
                    
                    # Display cluster membership
                    st.subheader("Cluster Membership")
                    
                    cluster_membership = clustered_data[['id', 'cluster']]
                    st.dataframe(cluster_membership)
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    st.write("Top parameters contributing to variation in the data:")
                    
                    # Display top loadings for PC1
                    top_loadings = loadings['PC1'].abs().sort_values(descending=True).head(10)
                    
                    loading_fig = px.bar(
                        x=top_loadings.values,
                        y=top_loadings.index,
                        orientation='h',
                        title="Top Parameters Contributing to PC1",
                        labels={'x': 'Loading Magnitude', 'y': 'Parameter'}
                    )
                    loading_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    st.plotly_chart(loading_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in clustering analysis: {e}")
                    st.info("Make sure your data has sufficient numeric values across multiple animals for clustering to work properly.")
            else:
                st.info("Pattern detection requires data from at least 2 different animals.")
            
            # Anomaly detection section
            st.subheader("Health Anomaly Detection")
            
            # Get the most recent test results for each animal and test
            most_recent = data.sort_values('date').groupby(['id', 'test']).last().reset_index()
            
            # Check for anomalies
            anomalies = []
            for _, row in most_recent.iterrows():
                if is_outside_range(row['result_numeric'], row['test'], row['age_years'] if 'age_years' in row else None):
                    deviation = get_deviation(row['result_numeric'], row['test'], row['age_years'] if 'age_years' in row else None)
                    ref_range = get_reference_range(row['test'], row['age_years'] if 'age_years' in row else None)
                    anomalies.append({
                        'Animal ID': row['id'],
                        'Parameter': row['test'],
                        'Value': row['result'],
                        'Units': row['units'] if 'units' in row else ref_range['units'] if ref_range else '',
                        'Reference Range': f"{ref_range['min']} - {ref_range['max']}" if ref_range else 'N/A',
                        'Status': deviation,
                        'Date': row['date']
                    })
            
            if anomalies:
                anomaly_df = pd.DataFrame(anomalies)
                
                # Display anomalies
                st.markdown("### Detected Anomalies")
                
                # Use custom styling to highlight high/low values
                def highlight_anomaly(val):
                    if val == 'High':
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    elif val == 'Low':
                        return 'background-color: rgba(0, 0, 255, 0.2)'
                    else:
                        return ''
                
                st.dataframe(
                    anomaly_df.style.applymap(highlight_anomaly, subset=['Status']), 
                    height=400
                )
                
                # Option to download anomaly report
                anomaly_csv = anomaly_df.to_csv(index=False)
                b64 = base64.b64encode(anomaly_csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="capuchin_anomalies.csv">Download Anomaly Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.success("No health anomalies detected in the most recent test results.")

# Custom CSS to improve the dashboard appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: rgba(245, 245, 245, 0.5);
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f0f0;
        font-weight: bold;
    }
    [data-testid="stMetric"] {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 5px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    create_dashboard()