# FINAL STREAMLIT DASHBOARD

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# CONFIG
st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# LOAD DATA
btp = pd.read_excel("ADAinB.xlsx")
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
ons_age = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Age Group")
combined = pd.read_excel("Combined_BTP_ONS_Quarterly_RegionMapped.xlsx")

# PREPROCESS
btp['Month'] = pd.to_datetime(btp['Month'], errors='coerce')
btp['Quarter'] = btp['Month'].dt.to_period('Q').astype(str)
btp['Quarter_dt'] = btp['Month'].dt.to_period('Q').dt.to_timestamp() + pd.offsets.QuarterEnd(0)

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "🌍 Region Explorer", "📄 Raw Data", "🧪 Predictive Insights", "⚙️ Settings"
])

# TAB 1: Overview
with tab1:
    st.title("UK Crime and Well-being Dashboard (2022–2024)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Crimes", f"{btp.shape[0]:,}")
    col2.metric("Avg. Life Satisfaction", f"{combined['Life_Satisfaction_Mean_Score'].mean():.2f}")
    col3.metric("Avg. Anxiety", f"{combined['Anxiety_Mean_Score'].mean():.2f}")

    st.markdown("### Crime Overview")
    col4, col5 = st.columns(2)

    fig1 = px.bar(combined, x='Quarter', y='Total_Crimes', color='Total_Crimes',
                  title="Total Crimes Per Quarter", color_continuous_scale='Reds')
    col4.plotly_chart(fig1, use_container_width=True)

    top_crimes = btp['Crime type'].value_counts().nlargest(6).reset_index()
    top_crimes.columns = ['Crime type', 'Count']
    fig2 = px.pie(top_crimes, names='Crime type', values='Count', title="Top 6 Crime Types")
    col5.plotly_chart(fig2, use_container_width=True)

# TAB 2: Crime Trends
with tab2:
    st.header("📈 Crime Trends Explorer")
    selected_county = st.selectbox("Select County", sorted(btp['County'].dropna().unique()))
    filtered = btp[btp['County'] == selected_county]
    crime_types = sorted(filtered['Crime type'].dropna().unique())
    selected_types = st.multiselect("Select Crime Types", crime_types, default=crime_types[:3])

    if selected_types:
        f = filtered[filtered['Crime type'].isin(selected_types)].copy()
        trend = f.groupby(['Quarter', 'Crime type']).size().reset_index(name='Count')
        fig3 = px.bar(trend, x='Crime type', y='Count', animation_frame='Quarter',
                      color='Crime type', title=f"Animated Trends in {selected_county}")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📍 Crime Location Map")
        loc_data = f[['Latitude', 'Longitude']].dropna()
        if not loc_data.empty:
            st.map(loc_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))
        else:
            st.warning("No map data for selected filters.")
    else:
        st.info("Please select at least one crime type.")

# TAB 3: Well-being Trends
with tab3:
    st.header("😊 Well-being Trends by Area")
    area = st.selectbox("Select Area", sorted(ons_area['Area'].unique()))
    area_df = ons_area[ons_area['Area'] == area]

    fig4 = px.line(area_df, x='Quarter', y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
                   markers=True, title=f"Well-being Over Time: {area}")
    st.plotly_chart(fig4, use_container_width=True)

# TAB 4: Deep Dive
with tab4:
    st.header("🔍 Deep Dive: Crime vs Well-being")
    crime_option = st.selectbox("Select Crime Type", sorted(btp['Crime type'].dropna().unique()))
    crime_df = btp[btp['Crime type'] == crime_option].groupby('Quarter').size().reset_index(name='Count')
    merged = pd.merge(combined, crime_df, on='Quarter', how='left').fillna(0)

    fig5 = px.scatter(merged, x='Count', y='Life_Satisfaction_Mean_Score', trendline='ols',
                      title=f"{crime_option} vs Life Satisfaction")
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.scatter(merged, x='Count', y='Anxiety_Mean_Score', trendline='ols',
                      title=f"{crime_option} vs Anxiety")
    st.plotly_chart(fig6, use_container_width=True)

    corr1, _ = pearsonr(merged['Count'], merged['Life_Satisfaction_Mean_Score'])
    corr2, _ = pearsonr(merged['Count'], merged['Anxiety_Mean_Score'])
    st.metric("Correlation (Crime & Life Satisfaction)", f"{corr1:.2f}")
    st.metric("Correlation (Crime & Anxiety)", f"{corr2:.2f}")


# TAB 5: Region Explorer
with tab5:
    st.header("🌍 Region Explorer by Country or County")
    region_level = st.radio("Select Level", ["County", "Country"])
    region_col = "County" if region_level == "County" else "Country"
    regions = sorted(btp[region_col].dropna().unique())
    selected_region = st.selectbox(f"Select {region_level}", regions)
    region_df = btp[btp[region_col] == selected_region]

    st.metric("Total Crimes", f"{region_df.shape[0]:,}")
    crime_summary = region_df['Crime type'].value_counts().nlargest(5).reset_index()
    fig7 = px.bar(crime_summary, x='index', y='Crime type', color='Crime type',
                  title=f"Top 5 Crime Types in {selected_region}")
    st.plotly_chart(fig7, use_container_width=True)

    if region_level == "County":
        region_ons = ons_area[ons_area['Area'] == selected_region]
        if not region_ons.empty:
            fig8 = px.line(region_ons, x='Quarter',
                           y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
                           title=f"Well-being in {selected_region}", markers=True)
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.warning("Well-being data not available for this County.")

# TAB 6: Raw Data
with tab6:
    st.header("📄 Raw Data Viewer")
    dataset = st.radio("Choose Dataset", ["BTP", "ONS Area", "ONS Age", "Combined"])
    if dataset == "BTP":
        st.dataframe(btp)
    elif dataset == "ONS Area":
        st.dataframe(ons_area)
    elif dataset == "ONS Age":
        st.dataframe(ons_age)
    else:
        st.dataframe(combined)

# TAB 7: Predictive Insights
with tab7:
    st.header("🧪 Predictive Insights")
    combined['Quarter_Index'] = range(1, len(combined) + 1)
    X = combined[['Quarter_Index']]
    y = combined['Total_Crimes']
    model = LinearRegression()
    model.fit(X, y)
    combined['Predicted_Crimes'] = model.predict(X)

    fig9 = px.line(combined, x='Quarter',
                   y=['Total_Crimes', 'Predicted_Crimes'],
                   title="Actual vs Predicted Crime Volume")
    st.plotly_chart(fig9, use_container_width=True)
    st.success(f"R² Score: {model.score(X, y):.2f}")

# TAB 8: Settings
with tab8:
    st.header("⚙️ Dashboard Settings")
    dark_mode = st.toggle("🌙 Enable Dark Mode")

    if dark_mode:
        st.markdown("""
            <style>
            .stApp { background-color: #111 !important; color: #fff; }
            .css-1cpxqw2, .stMarkdown, .stDataFrame { color: white !important; }
            .stSelectbox, .stTextInput { background-color: #222; color: white; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #ffffff !important; color: black; }
            .css-1cpxqw2, .stMarkdown, .stDataFrame { color: black !important; }
            .stSelectbox, .stTextInput { background-color: #f9f9f9; color: black; }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("Developed for UEA NBS7096A – Strategic Cyber Analytics Project")
