# FINAL DASHBOARD: UK Crime and Well-being Trends (2022–2024)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# --- CONFIG ---
st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# --- LOAD DATA ---
sao = pd.read_excel("ADAinB.xlsx")
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
ons_age = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Age Group")
combined_data = pd.read_excel("Combined_BTP_ONS_Quarterly_Data.xlsx")

# --- PREPROCESSING ---
sao['Month'] = pd.to_datetime(sao['Month'], format="%Y-%m")
sao['Quarter'] = sao['Month'].dt.to_period('Q').astype(str)
sao['Quarter_dt'] = sao['Month'].dt.to_period('Q').dt.to_timestamp() + pd.offsets.QuarterEnd(0)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "🌍 Region Explorer", "📄 Raw Data", "🧪 Predictive Insights", "⚙️ Settings"
])

# --- OVERVIEW ---
with tab1:
    st.title("🚨 UK Crime and Public Well-being Dashboard")
    st.markdown("### Summary Overview (2022–2024)")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Reported Crimes", f"{sao.shape[0]:,}")
    col2.metric("😊 Avg. Life Satisfaction", f"{combined_data['Life_Satisfaction_Mean_Score'].mean():.2f}")
    col3.metric("😟 Avg. Anxiety", f"{combined_data['Anxiety_Mean_Score'].mean():.2f}")

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        fig1 = px.bar(combined_data, x='Quarter', y='Total_Crimes', color='Total_Crimes',
                      title="Total Crimes per Quarter", color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)

    with col5:
        top_crimes = sao['Crime type'].value_counts().nlargest(6).reset_index()
        top_crimes.columns = ['Crime type', 'Count']
        fig2 = px.pie(top_crimes, values='Count', names='Crime type', title="Top 6 Crime Types Distribution")
        st.plotly_chart(fig2, use_container_width=True)

# --- CRIME TRENDS ---
with tab2:
    st.header("📈 Crime Trends Explorer")
    crime_types = sorted(sao['Crime type'].dropna().unique())
    selected_crimes = st.multiselect("Select Crime Types", crime_types, default=crime_types[:3])

    if selected_crimes:
        filtered = sao[sao['Crime type'].isin(selected_crimes)].copy()
        crime_trend = filtered.groupby(['Quarter', 'Crime type']).size().reset_index(name='Count')

        fig3 = px.bar(
            crime_trend, x='Crime type', y='Count', animation_frame='Quarter',
            color='Crime type', title="Animated Crime Trends by Type Over Quarters"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🗺️ Crime Locations Map")
        latest_q = filtered['Quarter'].max()
        last_map = filtered[filtered['Quarter'] == latest_q]
        map_data = last_map[['Latitude', 'Longitude']].dropna().rename(columns={"Latitude": "lat", "Longitude": "lon"})
        if not map_data.empty:
            st.map(map_data, zoom=5)
        else:
            st.warning("No map data available for selected crime types in latest quarter.")
    else:
        st.warning("Please select at least one crime type.")

# --- WELL-BEING TRENDS ---
with tab3:
    st.header("😊 Well-being Trends (ONS Area Data)")
    selected_area = st.selectbox("Select Region (Area)", sorted(ons_area['Area'].dropna().unique()))
    area_data = ons_area[ons_area['Area'] == selected_area]

    fig4 = px.line(
        area_data, x='Quarter',
        y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
        markers=True, title=f"Well-being Trends in {selected_area}"
    )
    st.plotly_chart(fig4, use_container_width=True)

# --- DEEP DIVE ---
with tab4:
    st.header("🔍 Deep Dive: Crime vs Well-being")
    selected_type = st.selectbox("Select Crime Type", sao['Crime type'].dropna().unique())
    st.markdown(f"### Correlation with '{selected_type}'")

    crime_q = sao[sao['Crime type'] == selected_type].groupby('Quarter').size().reset_index(name='Count')
    merged = pd.merge(combined_data, crime_q, on='Quarter', how='left').fillna(0)

    fig5 = px.scatter(merged, x='Count', y='Life_Satisfaction_Mean_Score', trendline='ols',
                      title="Crime Count vs Life Satisfaction")
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.scatter(merged, x='Count', y='Anxiety_Mean_Score', trendline='ols',
                      title="Crime Count vs Anxiety")
    st.plotly_chart(fig6, use_container_width=True)

    corr1, p1 = pearsonr(merged['Count'], merged['Life_Satisfaction_Mean_Score'])
    corr2, p2 = pearsonr(merged['Count'], merged['Anxiety_Mean_Score'])
    st.metric("Correlation (Crime & Life Satisfaction)", f"{corr1:.2f}", delta=f"p = {p1:.3f}")
    st.metric("Correlation (Crime & Anxiety)", f"{corr2:.2f}", delta=f"p = {p2:.3f}")

# --- REGION EXPLORER ---
with tab5:
    st.header("🌍 Region Explorer")
    region = st.selectbox("Select a Region", ons_area['Area'].unique())
    area_avg = ons_area[ons_area['Area'] == region]

    st.subheader(f"📊 Summary for {region}")
    total_crimes = sao[sao['Falls within'] == region].shape[0]
    st.metric("Total BTP Crimes", f"{total_crimes:,}")
    st.metric("Avg. Life Satisfaction", f"{area_avg['Life_Satisfaction_Mean_Score'].mean():.2f}")
    st.metric("Avg. Anxiety", f"{area_avg['Anxiety_Mean_Score'].mean():.2f}")

    st.markdown("### 📈 Well-being Over Time")
    fig7 = px.line(area_avg, x='Quarter',
                   y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
                   markers=True, title=f"Well-being Over Time in {region}")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("### 🧑‍🤝‍🧑 Age Group Comparison")
    age_group_avg = ons_age.groupby('Age_Group').agg({
        'Life_Satisfaction_Mean_Score': 'mean',
        'Anxiety_Mean_Score': 'mean'
    }).reset_index()

    fig8 = px.bar(age_group_avg, x='Age_Group', y='Life_Satisfaction_Mean_Score',
                  color='Life_Satisfaction_Mean_Score', title="Avg. Life Satisfaction by Age Group")
    st.plotly_chart(fig8, use_container_width=True)

# --- RAW DATA ---
with tab6:
    st.header("📄 Raw Data")
    dataset = st.radio("Select Dataset", ("Combined", "BTP", "ONS Area", "ONS Age"))
    if dataset == "Combined":
        st.dataframe(combined_data)
    elif dataset == "BTP":
        st.dataframe(sao)
    elif dataset == "ONS Area":
        st.dataframe(ons_area)
    else:
        st.dataframe(ons_age)

# --- PREDICTIVE INSIGHTS ---
with tab7:
    st.header("🧪 Predictive Insights")
    combined_data['Quarter_Index'] = range(1, len(combined_data)+1)
    X = combined_data[['Quarter_Index']]
    y = combined_data['Total_Crimes']
    model = LinearRegression().fit(X, y)
    combined_data['Predicted_Crimes'] = model.predict(X)

    fig9 = px.line(
        combined_data, x='Quarter',
        y=['Total_Crimes', 'Predicted_Crimes'],
        title="Actual vs Predicted Crime Trends"
    )
    st.plotly_chart(fig9, use_container_width=True)
    st.success(f"Model R² Score: {model.score(X, y):.2f}")

# --- SETTINGS ---
with tab8:
    st.header("⚙️ Dashboard Settings")
    dark_mode = st.toggle("🌙 Toggle Dark Mode")
    if dark_mode:
        st.markdown("""
            <style>body, .stApp { background-color: #111; color: white; }</style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>body, .stApp { background-color: #fff; color: black; }</style>
        """, unsafe_allow_html=True)

    st.markdown("Developed for Strategic Analytics – UEA NBS7096A")