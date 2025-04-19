import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Config ---
st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# --- Load Data ---
sao = pd.read_excel("ADAinB.xlsx")
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
combined_data = pd.read_excel("Combined_BTP_ONS_Quarterly_Data_Updated.xlsx")

# --- Preprocess ---
sao['Month'] = pd.to_datetime(sao['Month'], errors='coerce')
sao['Quarter'] = sao['Month'].dt.to_period('Q').astype(str)
sao['Quarter_dt'] = sao['Month'].dt.to_period('Q').dt.to_timestamp() + pd.offsets.QuarterEnd(0)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "🌍 Region Explorer", "📄 Raw Data", "🧪 Predictive Insights", "⚙️ Settings"
])

# --- TAB 1: Overview ---
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
        fig1 = px.bar(
            combined_data,
            x='Quarter', y='Total_Crimes',
            title="Total Crimes per Quarter",
            color='Total_Crimes',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col5:
        top_crimes = sao['Crime type'].value_counts().nlargest(6).reset_index()
        top_crimes.columns = ['Crime type', 'Count']
        fig2 = px.pie(
            top_crimes,
            values='Count',
            names='Crime type',
            title="Top 6 Crime Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- TAB 2: Crime Trends + Map ---
with tab2:
    st.header("📈 Crime Trends Explorer")
    crime_types = sorted(sao['Crime type'].dropna().unique())
    selected_crimes = st.multiselect("Select Crime Types", crime_types, default=crime_types[:3])

    if selected_crimes:
        filtered = sao[sao['Crime type'].isin(selected_crimes)].copy()
        crime_trend = filtered.groupby(['Quarter', 'Quarter_dt', 'Crime type']).size().reset_index(name='Count')

        fig3 = px.bar(
            crime_trend,
            x='Crime type', y='Count',
            animation_frame='Quarter',
            color='Crime type',
            title="Animated Crime Trends by Type Over Quarters"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🗺️ Crime Locations Map")
        map_data = filtered[['Latitude', 'Longitude']].dropna().rename(
            columns={"Latitude": "latitude", "Longitude": "longitude"}
        )
        if not map_data.empty:
            st.map(map_data, zoom=5)
        else:
            st.warning("No location data available.")
    else:
        st.warning("Please select at least one crime type.")

# --- TAB 3: Well-being Trends ---
with tab3:
    st.header("😊 Well-being Trends (ONS Area Data)")

    available_areas = sorted(ons_area['Area'].dropna().unique())
    selected_area = st.selectbox("Select Region (Area)", available_areas)
    filtered_area = ons_area[ons_area['Area'] == selected_area]

    fig4 = px.line(
        filtered_area,
        x='Quarter',
        y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
        title=f"Well-being Trends in {selected_area}",
        markers=True
    )
    st.plotly_chart(fig4, use_container_width=True)

# --- TAB 4: Deep Dive ---
with tab4:
    st.header("🔍 Deep Dive: Crime vs Well-being")

    selected_type = st.selectbox("Select Crime Type", sorted(sao['Crime type'].dropna().unique()))
    crime_by_q = sao[sao['Crime type'] == selected_type].groupby('Quarter').size().reset_index(name='Count')
    merged = pd.merge(crime_by_q, combined_data, on='Quarter', how='left')

    fig7 = px.scatter(
        merged,
        x='Count', y='Life_Satisfaction_Mean_Score',
        trendline='ols',
        title=f"{selected_type} vs Life Satisfaction"
    )
    st.plotly_chart(fig7, use_container_width=True)

    fig8 = px.scatter(
        merged,
        x='Count', y='Anxiety_Mean_Score',
        trendline='ols',
        title=f"{selected_type} vs Anxiety"
    )
    st.plotly_chart(fig8, use_container_width=True)

# --- TAB 5: Region Explorer ---
with tab5:
    st.header("🌍 Region Explorer")

    region_list = sorted(combined_data['Region'].dropna().unique())
    selected_region = st.selectbox("Select a Region", region_list)
    region_data = combined_data[combined_data['Region'] == selected_region]

    st.subheader(f"📊 Summary for {selected_region}")
    st.metric("Total BTP Crimes", int(region_data['Total_Crimes'].sum()))
    st.metric("Avg. Life Satisfaction", f"{region_data['Life_Satisfaction_Mean_Score'].mean():.2f}")
    st.metric("Avg. Anxiety", f"{region_data['Anxiety_Mean_Score'].mean():.2f}")

    if not region_data[['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score']].isnull().values.all():
        radar_data = region_data.groupby("Region")[["Life_Satisfaction_Mean_Score", "Anxiety_Mean_Score"]].mean().reset_index()
        uk_avg = combined_data[combined_data['Region'] != "UK (BTP)"].groupby("Region")[["Life_Satisfaction_Mean_Score", "Anxiety_Mean_Score"]].mean().mean()
        radar_data = radar_data.append({'Region': 'UK Average', 
                                        'Life_Satisfaction_Mean_Score': uk_avg['Life_Satisfaction_Mean_Score'], 
                                        'Anxiety_Mean_Score': uk_avg['Anxiety_Mean_Score']}, ignore_index=True)

        fig_radar = px.line_polar(
            radar_data.melt(id_vars="Region"),
            r="value", theta="variable", color="Region", line_close=True,
            title="Regional vs UK Average Well-being"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# --- TAB 6: Raw Data ---
with tab6:
    st.header("📄 Raw Data")
    dataset_choice = st.radio("Select Dataset", ("Combined", "BTP", "ONS"))

    if dataset_choice == "Combined":
        st.dataframe(combined_data)
    elif dataset_choice == "BTP":
        st.dataframe(sao)
    else:
        st.dataframe(ons_area)

# --- TAB 7: Predictive Insights ---
with tab7:
    st.header("🧪 Predictive Insights")

    st.markdown("Forecast total crimes using linear regression")
    df = combined_data.copy()
    df = df[df['Total_Crimes'].notnull()]
    df['Quarter_Index'] = range(1, len(df) + 1)
    X = df[['Quarter_Index']]
    y = df['Total_Crimes']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    df['Predicted_Crimes'] = y_pred

    fig_pred = px.line(
        df,
        x='Quarter',
        y=['Total_Crimes', 'Predicted_Crimes'],
        title="Actual vs Predicted Crime Trends"
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    st.success(f"Model R² Score: {model.score(X, y):.2f}")

# --- TAB 8: Settings ---
with tab8:
    st.header("⚙️ Dashboard Settings")
    dark_mode = st.toggle("🌙 Toggle Dark Mode")

    if dark_mode:
        st.markdown("""
            <style>
            body, .stApp { background-color: #121212; color: white; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body, .stApp { background-color: #ffffff; color: black; }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**App Developed for Analytical Exploration of Crime & Well-being Trends in the UK**")