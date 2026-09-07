import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_categorical_dtype


st.set_page_config(
    page_title="EDA Interface",
    page_icon="📊",
    layout="wide"
)

st.title("Exploratory Data Analysis Interface")

# --- Sidebar: Dataset Controls ---
st.sidebar.header("Dataset Controls")
st.sidebar.write("Upload CSV File for Analysis")
uploaded_file = st.sidebar.file_uploader("Limit 200MB per file • CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        

        st.header("Dataset Preview & Metadata")
        

        st.write("**First 5 Rows:**")
        st.dataframe(df.head())
        
        # Display dataset dimensions (shape)
        st.markdown(f"**Shape:** `<span style='color:green'> {df.shape} </span>`", unsafe_allow_html=True)
        
        # Display column data types
        st.write("**Column Data Types:**")
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str)
        st.dataframe(dtypes_df, use_container_width=True)
        
        # Display missing values per column
        st.write("**Missing Values per Column:**")
        missing_df = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_df, use_container_width=True)
        
        # Display statistical summary for numerical attributes
        st.write("**Statistical Summary (Numerical Attributes):**")
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            stats_df = df[num_cols].describe().T[['mean', '50%', 'min', 'max']]
            stats_df.rename(columns={'50%': 'median'}, inplace=True)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numerical attributes available for statistical summary.")
            
        # ---------------------------------------------------------
        # 2. Attribute Selection & Classification
        # ---------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.header("Attribute Selection")
        selected_col = st.sidebar.selectbox("Select Attribute for Visualization", df.columns)
        
        # ---------------------------------------------------------
        # 3. Visualization Module
        # ---------------------------------------------------------
        st.markdown("---")
        st.header("Visualization")
        
        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Conditional Rendering based on data type
        if is_numeric_dtype(df[selected_col]):
            # Numerical: Histogram with KDE (matching the screenshot styling)
            sns.histplot(df[selected_col], kde=True, color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(f'Histogram of {selected_col}', fontsize=16)
            ax.set_xlabel(selected_col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
        
        else:
            # Categorical: Bar chart of frequencies
            # Get value counts to calculate percentages later
            val_counts = df[selected_col].value_counts()
            
            sns.barplot(x=val_counts.index, y=val_counts.values, palette='viridis', ax=ax)
            ax.set_title(f'Frequency Chart of {selected_col}', fontsize=16)
            ax.set_xlabel(selected_col, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            
            # Optional: Rotate x-axis labels if there are many categories
            plt.xticks(rotation=45, ha='right')
            
            # Add percentage labels on top of bars
            total = len(df[selected_col].dropna())
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                x = p.get_x() + p.get_width() / 2 - 0.1
                y = p.get_y() + p.get_height() + (p.get_height()*0.02)
                ax.annotate(percentage, (x, y), size=10)

        # Render the plot in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Please upload a CSV file from the sidebar to begin exploratory data analysis.")