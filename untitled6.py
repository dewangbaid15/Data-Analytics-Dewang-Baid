import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# Load data
btp = pd.read_excel("ADAinB.xlsx")
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
ons_age = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Age Group")
combined = pd.read_excel("Combined_BTP_ONS_Quarterly_RegionMapped.xlsx")

# Filter only British regions in BTP dataset
valid_states = ['England', 'Scotland', 'Wales']
btp = btp[btp['State'].isin(valid_states)]

# Preprocess
btp['Month'] = pd.to_datetime(btp['Month'], errors='coerce')
btp['Quarter'] = btp['Month'].dt.to_period("Q").astype(str)
btp['Quarter_dt'] = btp['Month'].dt.to_period("Q").dt.to_timestamp() + pd.offsets.QuarterEnd(0)
btp['County'] = btp['County'].astype(str)
btp['Country'] = btp['Country'].astype(str)
btp['State'] = btp['State'].astype(str)
btp['City'] = btp['City'].astype(str)

# Merge for Deep Dive
crime_counts = btp.groupby(['Quarter', 'Crime type']).size().reset_index(name='Crime_Count')
merged = pd.merge(crime_counts, combined, on='Quarter', how='inner')

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠  Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "🌍 Location Insights","🧪 Predictive Insights", "📄 Raw Data", "⚙️ Settings"
])

# Overview
with tab1:
    st.markdown("### 🔍 Overview: Setting the Scene")
    st.markdown("Begin your journey with a high-level snapshot. Understand the scale of crime and public well-being across UK rail transport—setting the stage for deeper analysis.")
    st.title("🚨 UK Crime and Public Well-being Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Crimes", f"{btp.shape[0]:,}")
    col2.metric("😊 Avg. Life Satisfaction", f"{combined['Life_Satisfaction_Mean_Score'].mean():.2f}")
    col3.metric("😟 Avg. Anxiety", f"{combined['Anxiety_Mean_Score'].mean():.2f}")

    col4, col5 = st.columns(2)
    with col4:
        fig1 = px.bar(combined, x='Quarter', y='Total_Crimes', color='Total_Crimes',
                      title="Total Crimes per Quarter", color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)
    with col5:
        top_crimes = btp['Crime type'].value_counts().nlargest(6).reset_index()
        top_crimes.columns = ['Crime type', 'Count']
        fig2 = px.pie(top_crimes, values='Count', names='Crime type', title="Top 6 Crime Types")
        st.plotly_chart(fig2, use_container_width=True)

# Crime Trends
with tab2:
    st.markdown("### 🕵️ Crime Trends: What Happened When?")
    st.markdown("Explore how crime types have changed over time. Animated trends and maps reveal seasonality and hot zones. This helps us understand which types of incidents are rising—and when.")
    st.header("📈 Crime Trends Explorer")
    st.subheader("📱 Animated Trends by Crime Type")
    crime_types = sorted(btp['Crime type'].dropna().unique())
    selected_crimes = st.multiselect("Select Crime Types for Animation", crime_types, default=crime_types[:2])

    if selected_crimes:
        filtered = btp[btp['Crime type'].isin(selected_crimes)]
        grouped = filtered.groupby(['Quarter', 'Crime type']).size().reset_index(name='Count')
        fig3 = px.bar(grouped, x="Crime type", y="Count", animation_frame="Quarter", color="Crime type",
                      title="Animated Crime Trends")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Please select at least one crime type to show animated trends.")

    st.subheader("🗺️ Crime Location Map")
    crime_map_type = st.selectbox("Select Crime Type for Map", crime_types)
    map_df = btp[btp['Crime type'] == crime_map_type][['Latitude', 'Longitude']].dropna()
    map_df = map_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    if not map_df.empty:
        st.map(map_df, zoom=5)
    else:
        st.warning("No location data available for this crime type.")

# Well-being Trends
with tab3:
    st.markdown("### 😊 Well-being Trends: Understanding the People")
    st.markdown("Track how life satisfaction and anxiety vary across UK regions. These emotional metrics help reveal if rising crime is mirrored by a drop in public well-being.")
    st.header("🙌 Well-being Trends by Area")
    areas = sorted(ons_area['Area'].dropna().unique())
    selected_area = st.selectbox("Select Region (Area)", areas)
    area_df = ons_area[ons_area['Area'] == selected_area]
    fig4 = px.line(area_df, x='Quarter',
                   y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
                   title=f"Well-being in {selected_area}", markers=True)
    st.plotly_chart(fig4, use_container_width=True)

# Deep Dive
with tab4:
    st.markdown("### 🔬 Deep Dive: Correlation Between Crime & Well-being")
    st.markdown("Do more crimes mean more anxiety? Use data-driven correlation plots to examine how specific types of crime may impact public sentiment. This tab connects our past observations with underlying relationships.")
    st.header("😊 Deep Dive: Crime Type vs Well-being")
    st.markdown("Explore how specific types of crime correlate with public well-being over time.")
    crime_options = sorted(merged['Crime type'].dropna().unique())
    selected_crime = st.selectbox("Select a Crime Type", crime_options)
    filtered_merged = merged[merged['Crime type'] == selected_crime]

    fig_ls = px.scatter(filtered_merged, x='Crime_Count', y='Life_Satisfaction_Mean_Score',
                        trendline='ols', title=f"{selected_crime} vs Life Satisfaction")
    st.plotly_chart(fig_ls, use_container_width=True)

    fig_anx = px.scatter(filtered_merged, x='Crime_Count', y='Anxiety_Mean_Score',
                         trendline='ols', title=f"{selected_crime} vs Anxiety")
    st.plotly_chart(fig_anx, use_container_width=True)

    if not filtered_merged.empty:
        corr_ls, p_ls = pearsonr(filtered_merged['Crime_Count'], filtered_merged['Life_Satisfaction_Mean_Score'])
        corr_anx, p_anx = pearsonr(filtered_merged['Crime_Count'], filtered_merged['Anxiety_Mean_Score'])
        st.metric("Correlation (Crime & Life Satisfaction)", f"{corr_ls:.2f}", delta=f"p = {p_ls:.3f}")
        st.metric("Correlation (Crime & Anxiety)", f"{corr_anx:.2f}", delta=f"p = {p_anx:.3f}")
    else:
        st.warning("No data available for this crime type.")

# Location Insights
with tab5:
    st.markdown("### 🌍 Location Insights: Where Is It Happening?")
    st.markdown("Drill down into regional crime data to explore where incidents are concentrated. From national trends to specific counties and cities, this tab localises the patterns and helps pinpoint hotspots.")
    st.header("📍 Location Insights")
    state_options = sorted(btp['State'].dropna().unique())
    selected_state = st.selectbox("Select State", state_options)
    filtered_state = btp[btp['State'] == selected_state]
    county_options = sorted(filtered_state['County'].dropna().unique())
    selected_county = st.selectbox("Select County", county_options)
    filtered_county = filtered_state[filtered_state['County'] == selected_county]
    city_options = sorted(filtered_county['City'].dropna().unique())
    selected_city = st.selectbox("Select City", city_options)
    filtered_data = filtered_county[filtered_county['City'] == selected_city]

    st.subheader(f"🚨 Crime Summary for {selected_city}, {selected_county}, {selected_state}")
    col1, col2 = st.columns(2)
    col1.metric("Total Crimes", f"{filtered_data.shape[0]:,}")
    top_crime = filtered_data['Crime type'].value_counts().idxmax() if not filtered_data.empty else "N/A"
    col2.metric("Top Crime Type", top_crime)

    st.markdown("### 🚩 Crime Types Distribution")
    if not filtered_data.empty:
        crime_counts = filtered_data['Crime type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        fig_crime_bar = px.bar(crime_counts, x='Crime Type', y='Count',
                               title=f"Crime Types in {selected_city}", color='Count',
                               color_continuous_scale='Inferno')
        st.plotly_chart(fig_crime_bar, use_container_width=True)

    st.markdown(f"### 🗾 Crime Treemap by County in {selected_state}")
    treemap_df = filtered_state.groupby('County')['Crime type'].count().reset_index(name='Crime Count')
    fig_treemap = px.treemap(treemap_df, path=['County'], values='Crime Count',
                             title=f"Crime Volume by County in {selected_state}")
    st.plotly_chart(fig_treemap, use_container_width=True)

    st.markdown("### 🧭 Crime Hotspots Map")
    map_df = filtered_data[['Latitude', 'Longitude']].dropna()
    if not map_df.empty:
        st.map(map_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}), zoom=7)
    else:
        st.warning("No location data available for this city.")

with tab6:
    st.markdown("### 🔮 Predictive Insights: What Comes Next?")
    st.markdown("Can we foresee crime trends before they escalate? This tab uses linear regression to forecast crime for the next 2 years — tailored by crime type and region.")

    st.header("📊 Forecast Crime Volume (Next 8 Quarters)")

    # --- Filters with unique keys to avoid Streamlit widget conflicts
    crime_types = sorted(btp['Crime type'].dropna().unique())
    selected_crime = st.selectbox("Select Crime Type", crime_types, key="forecast_crime")

    states = sorted(btp['State'].dropna().unique())
    selected_state = st.selectbox("Select State", states, key="forecast_state")

    # --- Filter dataset
    forecast_df = btp[(btp['Crime type'] == selected_crime) & (btp['State'] == selected_state)]

    if forecast_df.empty:
        st.warning("No data available for selected crime type and state.")
    else:
        # Aggregate by Quarter
        forecast_df['Quarter'] = pd.to_datetime(forecast_df['Month'], errors='coerce').dt.to_period('Q').astype(str)
        trend = forecast_df.groupby('Quarter').size().reset_index(name='Crime_Count')
        trend = trend.sort_values('Quarter')
        trend['Quarter_Index'] = range(1, len(trend) + 1)

        # Linear Regression Model
        X = trend[['Quarter_Index']]
        y = trend['Crime_Count']
        model = LinearRegression().fit(X, y)

        # Forecast next 8 quarters
        future_index = pd.DataFrame({'Quarter_Index': range(len(trend)+1, len(trend)+9)})
        future_index['Crime_Count'] = model.predict(future_index[['Quarter_Index']]).round().astype(int)

        # Fix quarter labels based on last actual quarter
        last_q = trend['Quarter'].iloc[-1]
        last_period = pd.Period(last_q, freq='Q')
        future_quarters = [str(last_period + i) for i in range(1, 9)]
        future_index['Quarter'] = future_quarters
        future_index['Type'] = 'Forecast'

        # Actual data
        trend['Type'] = 'Actual'

        # Combine actual and forecast
        combined = pd.concat([
            trend[['Quarter', 'Crime_Count', 'Type']],
            future_index[['Quarter', 'Crime_Count', 'Type']]
        ], ignore_index=True)

        # Plot with continuous line
        fig = px.line(
            combined,
            x='Quarter',
            y='Crime_Count',
            color='Type',
            title=f"{selected_crime} Forecast in {selected_state}",
            markers=True,
            line_shape='linear'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Forecast complete. Model trained on {len(trend)} quarters. Forecasting next 8 (till 2026 Q4).")

            
#Raw Data
with tab7:
    st.markdown("### 📂 Raw Data: Trust the Source")
    st.markdown("Explore the full datasets powering the dashboard. Transparency matters — and this tab lets users audit, verify, or reuse the underlying data for further analysis.")
    st.header("📈 Raw Data")
    dataset = st.radio("Choose Dataset", ["BTP", "ONS Area", "ONS Age", "Combined"])
    if dataset == "BTP":
        st.subheader("British Transport Police (BTP) Crime Data")
        st.dataframe(btp)
    elif dataset == "ONS Area":
        st.subheader("ONS Well-being Data by Area")
        st.dataframe(ons_area)
    elif dataset == "ONS Age":
        st.subheader("ONS Well-being Data by Age Group")
        st.dataframe(ons_age)
    else:
        st.subheader("Combined BTP & ONS Data")
        st.write("**Columns in Combined Dataset:**", combined.columns.tolist())
        st.dataframe(combined)

# Settings
with tab8:
    st.markdown("### 🎛️ Settings: Personalise Your View")
    st.markdown("Toggle between light and dark mode to suit your visual preferences. This tab ensures your interaction with the dashboard is both accessible and user-friendly.")
    st.header("📲 Dashboard Settings")
    light_mode = st.toggle("🔅 Enable Light Mode")
    if light_mode:
        st.markdown("<style>body, .stApp { background-color: #ffffff; color: black; }</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body, .stApp { background-color: #121212; color: white; }</style>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("App Developed for **Advanced Topics in Data Analytics** at UEA.")
