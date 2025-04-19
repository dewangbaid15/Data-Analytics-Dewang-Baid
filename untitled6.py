import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# Load data
btp = pd.read_excel("ADAinB.xlsx")

# --- Task 1: Filter only British regions ---
valid_states = ['England', 'Scotland', 'Wales']
btp = btp[btp['State'].isin(valid_states)]
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
ons_age = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Age Group")
combined = pd.read_excel("Combined_BTP_ONS_Quarterly_RegionMapped.xlsx")

# --- Optional: Limit combined data to match filtered BTP regions ---
combined = combined[combined['Region'].isin(btp['County'].unique())]

# Preprocess
btp['Month'] = pd.to_datetime(btp['Month'], errors='coerce')
btp['Quarter'] = btp['Month'].dt.to_period("Q").astype(str)
btp['Quarter_dt'] = btp['Month'].dt.to_period("Q").dt.to_timestamp() + pd.offsets.QuarterEnd(0)
btp['County'] = btp['County'].astype(str)
btp['Country'] = btp['Country'].astype(str)
btp['State'] = btp['State'].astype(str)
btp['City'] = btp['City'].astype(str)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "📍 Location Insights", "📄 Raw Data", "🧪 Predictive Insights", "⚙️ Settings"
])

# TAB 1: Overview
with tab1:
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

# TAB 2: Crime Trends
with tab2:
    st.header("📈 Crime Trends Explorer")

    # --- Animated Crime Trends (Multiselect) ---
    st.subheader("📊 Animated Trends by Crime Type")
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

    # --- Crime Map (Separate dropdown) ---
    st.subheader("🗺️ Crime Location Map")
    crime_map_type = st.selectbox("Select Crime Type for Map", crime_types)
    map_df = btp[btp['Crime type'] == crime_map_type][['Latitude', 'Longitude']].dropna()
    map_df = map_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

    if not map_df.empty:
        st.map(map_df, zoom=5)
    else:
        st.warning("No location data available for this crime type.")
        
        
# TAB 3: Well-being Trends
with tab3:
    st.header("😊 Well-being Trends by Area")
    areas = sorted(ons_area['Area'].dropna().unique())
    selected_area = st.selectbox("Select Region (Area)", areas)
    area_df = ons_area[ons_area['Area'] == selected_area]
    fig4 = px.line(area_df, x='Quarter',
                   y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
                   title=f"Well-being in {selected_area}", markers=True)
    st.plotly_chart(fig4, use_container_width=True)

# TAB 4: Deep Dive
with tab4:
    st.header("🔍 Deep Dive: Crime vs Well-being")

    # --- Task 2: Add circular crime type selector ---
    crime_types = sorted(btp['Crime type'].dropna().unique())
    crime_types = ["All Crimes"] + crime_types
    selected_crime = st.radio("Select Crime Type:", crime_types, horizontal=True)

    if selected_crime != "All Crimes":
        filtered_combined = combined[combined['Crime type'] == selected_crime]
    else:
        filtered_combined = combined.copy()

    # Plot: Crime vs Life Satisfaction
    fig5 = px.scatter(filtered_combined, x='Total_Crimes', y='Life_Satisfaction_Mean_Score',
                      trendline='ols', title=f"{selected_crime} vs Life Satisfaction")
    st.plotly_chart(fig5, use_container_width=True)

    # Plot: Crime vs Anxiety
    fig6 = px.scatter(filtered_combined, x='Total_Crimes', y='Anxiety_Mean_Score',
                      trendline='ols', title=f"{selected_crime} vs Anxiety")
    st.plotly_chart(fig6, use_container_width=True)

    # Correlations
    if not filtered_combined.empty:
        corr_ls, p_ls = pearsonr(filtered_combined['Total_Crimes'], filtered_combined['Life_Satisfaction_Mean_Score'])
        corr_anx, p_anx = pearsonr(filtered_combined['Total_Crimes'], filtered_combined['Anxiety_Mean_Score'])

        st.metric("Correlation (Crime & Life Satisfaction)", f"{corr_ls:.2f}", delta=f"p = {p_ls:.3f}")
        st.metric("Correlation (Crime & Anxiety)", f"{corr_anx:.2f}", delta=f"p = {p_anx:.3f}")
    else:
        st.warning("No data available for the selected crime type.")
        
        
# ---------------- TAB 5: Location Insights --------------------------
with tab5:
    st.header("📍 Location Insights")
    st.markdown("Explore how crime varies by State, County, and City across the UK.")

    # Dropdown filters
    state_options = sorted(btp['State'].dropna().unique())
    selected_state = st.selectbox("Select State", state_options)

    filtered_state = btp[btp['State'] == selected_state]
    county_options = sorted(filtered_state['County'].dropna().unique())
    selected_county = st.selectbox("Select County", county_options)

    filtered_county = filtered_state[filtered_state['County'] == selected_county]
    city_options = sorted(filtered_county['City'].dropna().unique())
    selected_city = st.selectbox("Select City", city_options)

    # Filter final data
    filtered_data = filtered_county[filtered_county['City'] == selected_city]

    st.subheader(f"📊 Crime Summary for {selected_city}, {selected_county}, {selected_state}")
    col1, col2 = st.columns(2)
    col1.metric("Total Crimes", f"{filtered_data.shape[0]:,}")
    top_crime = filtered_data['Crime type'].value_counts().idxmax() if not filtered_data.empty else "N/A"
    col2.metric("Top Crime Type", top_crime)

    # Bar chart of crime type distribution
    st.markdown("### 🔍 Crime Types Distribution")
    if not filtered_data.empty:
        crime_counts = filtered_data['Crime type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        fig_crime_bar = px.bar(
            crime_counts, x='Crime Type', y='Count',
            title=f"Crime Types in {selected_city}",
            color='Count', color_continuous_scale='Inferno'
        )
        st.plotly_chart(fig_crime_bar, use_container_width=True)
    else:
        st.warning("No data available for the selected location.")

    # Treemap by county in selected state
    st.markdown(f"### 🌳 Crime Treemap by County in {selected_state}")
    treemap_df = filtered_state.groupby('County')['Crime type'].count().reset_index(name='Crime Count')
    fig_treemap = px.treemap(
        treemap_df, path=['County'], values='Crime Count',
        title=f"Crime Volume by County in {selected_state}"
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # Map of crimes
    st.markdown("### 🗺️ Crime Hotspots Map")
    map_df = filtered_data[['Latitude', 'Longitude']].dropna()
    if not map_df.empty:
        st.map(map_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}), zoom=7)
    else:
        st.warning("No location data available for this city.")
# TAB 6: Raw Data
with tab6:
    st.header("📄 Raw Data")
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

    fig8 = px.line(combined, x='Quarter', y=['Total_Crimes', 'Predicted_Crimes'],
                   title="Actual vs Predicted Crime Volume")
    st.plotly_chart(fig8, use_container_width=True)
    st.success(f"Model R² Score: {model.score(X, y):.2f}")

# TAB 8: Settings
with tab8:
    st.header("⚙️ Dashboard Settings")
    light_mode = st.toggle(" Enable Light Mode")

    if light_mode:
        st.markdown("""
            <style>
            body, .stApp { background-color: #ffffff; color: black; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body, .stApp { background-color: #121212; color: white; }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("App Developed for **Advanced Topics in Data Analytics** at UEA.")
