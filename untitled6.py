import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
import plotly.graph_objects as go
import numpy as np

# --- Set Page Config ---
st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# --- Load Data ---
sao = pd.read_excel("ADAinB.xlsx")
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
combined_data = pd.read_excel("Combined_BTP_ONS_Quarterly_Data.xlsx")

# --- Preprocessing ---
sao['Month'] = pd.to_datetime(sao['Month'], format="%Y-%m")
sao['Quarter'] = sao['Month'].dt.to_period('Q').astype(str)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔗 Crime vs Well-being",
    "🗺️ Region Explorer", "📄 Raw Data", "🧪 Predictive Insights", "⚙️ Settings"
])

# --- TAB 5: Region Explorer ---
with tab5:
    st.header("🗺️ Region Explorer")
    st.markdown("Explore average well-being metrics across UK regions.")
    area_avg = ons_area.groupby("Area").agg({
        "Life_Satisfaction_Mean_Score": "mean",
        "Anxiety_Mean_Score": "mean"
    }).reset_index()

    col14, col15 = st.columns(2)

    with col14:
        fig11 = px.bar(
            area_avg.sort_values(by='Life_Satisfaction_Mean_Score'),
            x='Life_Satisfaction_Mean_Score',
            y='Area',
            orientation='h',
            title="Average Life Satisfaction by Area",
            color='Life_Satisfaction_Mean_Score',
            color_continuous_scale='Blues')
        st.plotly_chart(fig11, use_container_width=True)

    with col15:
        fig12 = px.bar(
            area_avg.sort_values(by='Anxiety_Mean_Score'),
            x='Anxiety_Mean_Score',
            y='Area',
            orientation='h',
            title="Average Anxiety by Area",
            color='Anxiety_Mean_Score',
            color_continuous_scale='Purples')
        st.plotly_chart(fig12, use_container_width=True)
    st.markdown("---")

# --- TAB 6: Raw Data Viewer ---
with tab6:
    st.header("📄 Raw Data Viewer")
    dataset_choice = st.radio("Select Dataset", ("Combined", "BTP", "ONS"))

    if dataset_choice == "Combined":
        st.dataframe(combined_data)
        csv_data = combined_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Combined Data", data=csv_data, file_name="Combined_Data.csv", mime="text/csv")
    elif dataset_choice == "BTP":
        st.dataframe(sao)
        csv_data = sao.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download BTP Data", data=csv_data, file_name="BTP_Data.csv", mime="text/csv")
    else:
        st.dataframe(ons_area)
        csv_data = ons_area.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download ONS Area Data", data=csv_data, file_name="ONS_Area_Data.csv", mime="text/csv")

# --- TAB 7: Predictive Insights ---
with tab7:
    st.header("🧪 Predictive Insights")
    st.markdown("Using linear regression to project future crime trends.")

    # Create a numerical representation of Quarter for prediction
    combined_data['Quarter_Num'] = range(1, len(combined_data) + 1)
    X = combined_data[['Quarter_Num']]
    y = combined_data['Total_Crimes']

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Prediction
    combined_data['Predicted_Crimes'] = model.predict(X)

    fig13 = px.line(combined_data, x='Quarter', y=['Total_Crimes', 'Predicted_Crimes'],
                    markers=True, title="Crime Trend Prediction")
    fig13.update_traces(mode='lines+markers')
    st.plotly_chart(fig13, use_container_width=True)

# --- TAB 8: Settings ---
with tab8:
    st.header("⚙️ Settings")
    mode = st.radio("Select Theme Mode", ["Light", "Dark"])

    if mode == "Dark":
        st.markdown("""
        <style>
            body, .stApp {
                background-color: #111;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            body, .stApp {
                background-color: white;
                color: black;
            }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("Current theme and filter settings can be shown here.")
