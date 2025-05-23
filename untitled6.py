import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr

st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# Load data with caching
@st.cache_data
def load_data():
    btp = pd.read_excel("ADAinB.xlsx")
    ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
    ons_age = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Age Group")
    combined = pd.read_excel("Combined_BTP_ONS_Quarterly_RegionMapped.xlsx")
    return btp, ons_area, ons_age, combined

btp, ons_area, ons_age, combined = load_data()

# Preprocess BTP data
btp['Month'] = pd.to_datetime(btp['Month'], errors='coerce')
btp['Quarter'] = btp['Month'].dt.to_period("Q").astype(str)
btp['Quarter_dt'] = btp['Month'].dt.to_period("Q").dt.to_timestamp()
btp['County'] = btp['County'].astype(str)
btp['Country'] = btp['Country'].astype(str)
btp['State'] = btp['State'].astype(str)
btp['City'] = btp['City'].astype(str)

# Merge for Deep Dive
crime_counts = btp.groupby(['Quarter', 'Crime type']).size().reset_index(name='Crime_Count')
merged = pd.merge(crime_counts, combined, on='Quarter', how='inner')

# Sidebar filters
st.sidebar.header("🔍 Global Filters")
crime_types = sorted(btp['Crime type'].dropna().unique())
selected_crime = st.sidebar.selectbox("Select Crime Type", crime_types)

states = sorted(btp['State'].dropna().unique())
selected_state = st.sidebar.selectbox("Select State", states)

# Tabs layout
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "🌍 Location Insights", "🧪 Predictive Insights", "📄 Raw Data", "⚙️ Settings"
])

# -------------------- Tab 1: Overview --------------------
with tab1:
    st.markdown("### 🔍 Overview: Setting the Scene")
    st.markdown("Understand the scale of crime and public well-being across UK rail transport.")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Crimes", f"{btp.shape[0]:,}")
    col2.metric("😊 Avg. Life Satisfaction", f"{combined['Life_Satisfaction_Mean_Score'].mean():.2f}")
    col3.metric("😟 Avg. Anxiety", f"{combined['Anxiety_Mean_Score'].mean():.2f}")

    col4, col5 = st.columns(2)
    with col4:
        fig1 = px.bar(
            combined, x='Quarter', y='Total_Crimes', color='Total_Crimes',
            title="Total Crimes per Quarter", color_continuous_scale='Reds',
            hover_data={'Quarter': True, 'Total_Crimes': True}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col5:
        top_crimes = btp['Crime type'].value_counts().nlargest(6).reset_index()
        top_crimes.columns = ['Crime type', 'Count']
        fig2 = px.pie(
            top_crimes, values='Count', names='Crime type',
            title="Top 6 Crime Types", hover_data=['Count']
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info("🔑 **Key takeaway:** Crime trends are seasonal and certain crime types dominate, such as theft and violence.")

    
# -------------------- Tab 2: Crime Trends --------------------
with tab2:
    st.markdown("### 🕵️ Crime Trends: What Happened When?")
    st.markdown("Explore how selected crime types evolved over time and geography.")

    st.header("📈 Animated Crime Trends")
    filtered = btp[(btp['Crime type'] == selected_crime)]

    if filtered.empty:
        st.warning("No data available for the selected crime type.")
    else:
        grouped = filtered.groupby(['Quarter', 'Crime type']).size().reset_index(name='Count')
        fig3 = px.bar(
            grouped, x="Crime type", y="Count", animation_frame="Quarter",
            color="Crime type", title="Animated Crime Trends by Quarter",
            hover_data=['Crime type', 'Count']
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.header("🗺️ Crime Location Map")
    map_df = filtered[['Latitude', 'Longitude']].dropna().rename(
        columns={'Latitude': 'latitude', 'Longitude': 'longitude'}
    )
    if not map_df.empty:
        st.map(map_df, zoom=5)
    else:
        st.warning("No location data available for this crime type.")

    st.info("🔑 **Key takeaway:** Crime volumes vary over time, with peaks during certain quarters. Mapping reveals high-density areas.")


# -------------------- Tab 3: Well-being Trends --------------------
with tab3:
    st.markdown("### 😊 Well-being Trends: Understanding the People")
    st.markdown("Track changes in life satisfaction and anxiety across selected UK regions over time.")

    st.header("📉 Life Satisfaction and Anxiety Over Time")
    areas = sorted(ons_area['Area'].dropna().unique())
    selected_area = st.selectbox("Select Region (Area)", areas, key="wellbeing_area")
    area_df = ons_area[ons_area['Area'] == selected_area]

    if area_df.empty:
        st.warning("No well-being data available for the selected region.")
    else:
        fig4 = px.line(
            area_df,
            x='Quarter',
            y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
            title=f"Well-being Trends in {selected_area}",
            markers=True,
            hover_data={'Quarter': True, 'Life_Satisfaction_Mean_Score': True, 'Anxiety_Mean_Score': True}
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.info("🔑 **Key takeaway:** Well-being indicators such as life satisfaction and anxiety show meaningful shifts over time and differ regionally.")


# -------------------- Tab 4: Deep Dive --------------------
with tab4:
    st.markdown("### 🔬 Deep Dive: Correlation Between Crime & Well-being")
    st.markdown("Do more crimes lead to lower life satisfaction or higher anxiety? Use this tab to explore correlations.")

    st.header("📊 Correlation Visuals")
    crime_options = sorted(merged['Crime type'].dropna().unique())
    selected_corr_crime = st.selectbox("Select a Crime Type", crime_options, key="corr_crime_type")
    filtered_merged = merged[merged['Crime type'] == selected_corr_crime]

    if filtered_merged.empty:
        st.warning("No data available for the selected crime type.")
    else:
        # Scatter plot: Crime Count vs Life Satisfaction
        fig_ls = px.scatter(
            filtered_merged,
            x='Crime_Count',
            y='Life_Satisfaction_Mean_Score',
            trendline='ols',
            title=f"{selected_corr_crime} vs Life Satisfaction",
            hover_data=['Crime_Count', 'Life_Satisfaction_Mean_Score']
        )
        st.plotly_chart(fig_ls, use_container_width=True)

        # Scatter plot: Crime Count vs Anxiety
        fig_anx = px.scatter(
            filtered_merged,
            x='Crime_Count',
            y='Anxiety_Mean_Score',
            trendline='ols',
            title=f"{selected_corr_crime} vs Anxiety",
            hover_data=['Crime_Count', 'Anxiety_Mean_Score']
        )
        st.plotly_chart(fig_anx, use_container_width=True)

        # Pearson correlation metrics
        corr_ls, p_ls = pearsonr(filtered_merged['Crime_Count'], filtered_merged['Life_Satisfaction_Mean_Score'])
        corr_anx, p_anx = pearsonr(filtered_merged['Crime_Count'], filtered_merged['Anxiety_Mean_Score'])

        st.metric("Correlation (Crime & Life Satisfaction)", f"{corr_ls:.2f}", delta=f"p = {p_ls:.3f}")
        st.metric("Correlation (Crime & Anxiety)", f"{corr_anx:.2f}", delta=f"p = {p_anx:.3f}")

    st.info("🔑 **Key takeaway:** Some crime types are significantly correlated with public sentiment. Stronger correlations may indicate deeper social impacts.")


# -------------------- Tab 5: Location Insights --------------------
with tab5:
    st.markdown("### 🌍 Location Insights: Where Is It Happening?")
    st.markdown("Drill down by state, county, and city to explore where selected crimes are most common.")

    st.header("📌 Region Selector")
    filtered_state = btp[btp['State'] == selected_state]
    counties = sorted(filtered_state['County'].dropna().unique())
    selected_county = st.selectbox("Select County", counties, key="loc_county")
    filtered_county = filtered_state[filtered_state['County'] == selected_county]

    cities = sorted(filtered_county['City'].dropna().unique())
    selected_city = st.selectbox("Select City", cities, key="loc_city")
    filtered_data = filtered_county[filtered_county['City'] == selected_city]

    st.subheader(f"🚨 Crime Summary for {selected_city}, {selected_county}, {selected_state}")
    col1, col2 = st.columns(2)
    col1.metric("Total Crimes", f"{filtered_data.shape[0]:,}")
    top_crime = filtered_data['Crime type'].value_counts().idxmax() if not filtered_data.empty else "N/A"
    col2.metric("Top Crime Type", top_crime)

    st.markdown("### 🔢 Crime Types Distribution")
    if not filtered_data.empty:
        crime_counts = filtered_data['Crime type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        fig_bar = px.bar(
            crime_counts, x='Crime Type', y='Count', color='Count',
            title=f"Crime Types in {selected_city}",
            hover_data=['Crime Type', 'Count'], color_continuous_scale='Inferno'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"### 🗺️ Crime Treemap by County in {selected_state}")
    treemap_df = filtered_state.groupby('County')['Crime type'].count().reset_index(name='Crime Count')
    fig_tree = px.treemap(
        treemap_df, path=['County'], values='Crime Count',
        title=f"Crime Volume by County in {selected_state}",
        hover_data=['Crime Count']
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("### 📍 Crime Hotspots Map")
    map_df = filtered_data[['Latitude', 'Longitude']].dropna()
    if not map_df.empty:
        st.map(map_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}), zoom=7)
    else:
        st.warning("No location data available for this city.")

    st.info("🔑 **Key takeaway:** Crime intensity and types vary greatly by region, enabling targeted interventions.")


# -------------------- Tab 6: Predictive Insights --------------------
with tab6:
    st.markdown("### 🔮 Predictive Insights: What Comes Next?")
    st.markdown("Forecast future crime trends using a linear regression model — filtered by crime type and region.")

    st.header("📈 Forecasting Future Crime (Up to 12 Quarters Ahead)")

    # Forecast range slider
    forecast_horizon = st.slider("Select number of future quarters to forecast", 4, 12, 8)

    forecast_df = btp[(btp['Crime type'] == selected_crime) & (btp['State'] == selected_state)]
    if forecast_df.empty:
        st.warning("No data available for selected filters.")
    else:
        forecast_df['Quarter'] = pd.to_datetime(forecast_df['Month'], errors='coerce').dt.to_period('Q').astype(str)
        trend = forecast_df.groupby('Quarter').size().reset_index(name='Crime_Count')
        trend = trend.sort_values('Quarter')
        trend['Quarter_Index'] = range(1, len(trend) + 1)
        trend['Quarter_dt'] = trend['Quarter'].apply(lambda q: pd.Period(q, freq='Q').start_time)

        # Train model
        X = trend[['Quarter_Index']]
        y = trend['Crime_Count']
        model = LinearRegression().fit(X, y)
        r2_score = model.score(X, y)

        # Generate future quarters
        last_q = trend['Quarter'].iloc[-1]
        last_period = pd.Period(last_q, freq='Q')
        start_date = last_period.start_time
        future_dates = [start_date + pd.DateOffset(months=3 * i) for i in range(0, forecast_horizon + 1)]
        future_index = pd.DataFrame({'Quarter_dt': future_dates})
        future_index['Quarter_Index'] = range(len(trend), len(trend) + forecast_horizon + 1)
        future_index['Crime_Count'] = model.predict(future_index[['Quarter_Index']]).round().astype(int)
        future_index['Type'] = 'Forecast'

        # Label actuals
        trend['Type'] = 'Actual'

        # Combine data
        combined = pd.concat([
            trend[['Quarter_dt', 'Crime_Count', 'Type']],
            future_index[['Quarter_dt', 'Crime_Count', 'Type']]
        ], ignore_index=True)

        # Plot
        fig = px.line(
            combined,
            x='Quarter_dt',
            y='Crime_Count',
            color='Type',
            title=f"{selected_crime} Forecast in {selected_state}",
            markers=True,
            line_shape='linear',
            labels={'Quarter_dt': 'Quarter', 'Crime_Count': 'Crime Volume'}
        )
        fig.update_xaxes(tickformat="%b %Y")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Model R² Score", f"{r2_score:.2f}")
        st.success(f"Forecast complete: Model trained on {len(trend)} quarters. Forecasting next {forecast_horizon}.")

    st.info("🔑 **Key takeaway:** The forecast allows crime analysts to anticipate upcoming trends and plan preemptive strategies.")

            
# -------------------- Tab 7: Raw Data --------------------
with tab7:
    st.markdown("### 📂 Raw Data: Trust the Source")
    st.markdown("Explore the datasets behind this dashboard. Use filters and sorting to inspect underlying values.")

    st.header("📈 Raw Data Viewer")
    dataset = st.radio("Choose Dataset", ["BTP", "ONS Area", "ONS Age", "Combined"], key="raw_dataset")

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
        st.dataframe(combined)

    st.info("🔑 **Key takeaway:** Raw data access enables transparency and reusability in evidence-based research.")

# -------------------- Tab 8: Settings --------------------
with tab8:
    st.markdown("### 🎛️ Settings: Personalise Your View")
    st.markdown("Toggle dark/light mode and other visual options.")

    st.header("📲 Dashboard Settings")
    light_mode = st.toggle("🔅 Enable Light Mode")

    if light_mode:
        st.markdown("<style>body, .stApp { background-color: #ffffff; color: black; }</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body, .stApp { background-color: #121212; color: white; }</style>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("App Developed for **Advanced Topics in Data Analytics** at UEA.")
    st.info("🔑 **Key takeaway:** Dashboard personalization enhances accessibility and comfort for every user.")
