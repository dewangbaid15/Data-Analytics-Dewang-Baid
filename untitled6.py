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

# Preprocess BTP
btp['Month'] = pd.to_datetime(btp['Month'], errors='coerce')
btp['Quarter'] = btp['Month'].dt.to_period("Q").astype(str)
btp['Quarter_dt'] = btp['Month'].dt.to_period("Q").dt.to_timestamp() + pd.offsets.QuarterEnd(0)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "🌍 Region Explorer", "📄 Raw Data", "🧪 Predictive Insights", "⚙️ Settings"
])

# -------------------------- TAB 1: Overview --------------------------
with tab1:
    st.title("🚨 UK Crime and Public Well-being Dashboard")
    st.markdown("### Summary Overview (2022–2024)")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Reported Crimes", f"{btp.shape[0]:,}")
    col2.metric("😊 Avg. Life Satisfaction", f"{combined['Life_Satisfaction_Mean_Score'].mean():.2f}")
    col3.metric("😟 Avg. Anxiety", f"{combined['Anxiety_Mean_Score'].mean():.2f}")

    col4, col5 = st.columns(2)
    with col4:
        fig1 = px.bar(
            combined, x='Quarter', y='Total_Crimes', color='Total_Crimes',
            title="Total Crimes per Quarter", color_continuous_scale='Reds'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col5:
        top_crimes = btp['Crime type'].value_counts().nlargest(6).reset_index()
        top_crimes.columns = ['Crime type', 'Count']
        fig2 = px.pie(top_crimes, values='Count', names='Crime type', title="Top 6 Crime Types")
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------- TAB 2: Crime Trends --------------------------
with tab2:
    st.header("📈 Crime Trends Explorer")
    crime_types = sorted(btp['Crime type'].dropna().unique())
    selected_crimes = st.multiselect("Select Crime Types", crime_types, default=crime_types[:2])

    if selected_crimes:
        filtered = btp[btp['Crime type'].isin(selected_crimes)].copy()
        grouped = filtered.groupby(['Quarter', 'Crime type']).size().reset_index(name='Count')
        fig3 = px.bar(
            grouped, x="Crime type", y="Count", animation_frame="Quarter",
            color="Crime type", title="Animated Crime Trends"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🗺️ Crime Locations Map")
        map_df = filtered[['Latitude', 'Longitude']].dropna().rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
        if not map_df.empty:
            st.map(map_df, zoom=5)
        else:
            st.warning("No location data available.")
    else:
        st.warning("Please select at least one crime type.")

# -------------------------- TAB 3: Well-being Trends --------------------------
with tab3:
    st.header("😊 Well-being Trends by Area")
    areas = sorted(ons_area['Area'].dropna().unique())
    selected_area = st.selectbox("Select Region (Area)", areas)

    area_df = ons_area[ons_area['Area'] == selected_area]
    fig4 = px.line(
        area_df, x='Quarter',
        y=['Life_Satisfaction_Mean_Score', 'Anxiety_Mean_Score'],
        title=f"Well-being in {selected_area}",
        markers=True
    )
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------- TAB 4: Deep Dive --------------------------
with tab4:
    st.header("🔍 Deep Dive: Crime vs Well-being")

    fig5 = px.scatter(
        combined, x='Total_Crimes', y='Life_Satisfaction_Mean_Score',
        trendline='ols', title="Crime vs Life Satisfaction"
    )
    fig6 = px.scatter(
        combined, x='Total_Crimes', y='Anxiety_Mean_Score',
        trendline='ols', title="Crime vs Anxiety"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)

    corr_ls, p_ls = pearsonr(combined['Total_Crimes'], combined['Life_Satisfaction_Mean_Score'])
    corr_anx, p_anx = pearsonr(combined['Total_Crimes'], combined['Anxiety_Mean_Score'])

    st.metric("Correlation (Crime & Life Satisfaction)", f"{corr_ls:.2f}", delta=f"p = {p_ls:.3f}")
    st.metric("Correlation (Crime & Anxiety)", f"{corr_anx:.2f}", delta=f"p = {p_anx:.3f}")

# -------------------------- TAB 5: Region Explorer --------------------------
with tab5:
    st.header("🌍 Region Explorer")
    selected_county = st.selectbox("Select County", sorted(btp['County'].dropna().unique()))

    filtered_btp = btp[btp['County'] == selected_county]
    filtered_combined = combined[combined['Region'] == selected_county]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Crimes", f"{filtered_btp.shape[0]:,}")
    col2.metric("Avg. Life Satisfaction", f"{filtered_combined['Life_Satisfaction_Mean_Score'].mean():.2f}" if not filtered_combined.empty else "N/A")
    col3.metric("Avg. Anxiety", f"{filtered_combined['Anxiety_Mean_Score'].mean():.2f}" if not filtered_combined.empty else "N/A")

    if not filtered_combined.empty:
        fig7 = px.line(
            filtered_combined,
            x="Quarter", y=["Life_Satisfaction_Mean_Score", "Anxiety_Mean_Score"],
            title=f"Well-being Over Time in {selected_county}"
        )
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("Well-being data not available for this County.")

# -------------------------- TAB 6: Raw Data --------------------------
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

# -------------------------- TAB 7: Predictive Insights --------------------------
with tab7:
    st.header("🧪 Predictive Insights")

    combined['Quarter_Index'] = range(1, len(combined) + 1)
    X = combined[['Quarter_Index']]
    y = combined['Total_Crimes']

    model = LinearRegression()
    model.fit(X, y)
    combined['Predicted_Crimes'] = model.predict(X)

    fig8 = px.line(
        combined, x='Quarter', y=['Total_Crimes', 'Predicted_Crimes'],
        title="Actual vs Predicted Crime Volume"
    )
    st.plotly_chart(fig8, use_container_width=True)
    st.success(f"Model R² Score: {model.score(X, y):.2f}")

# -------------------------- TAB 8: Settings --------------------------
with tab8:
    st.header("⚙️ Dashboard Settings")
    dark_mode = st.toggle("🌙 Enable Dark Mode")

    if dark_mode:
        st.markdown("""
            <style>
            body, .stApp {
                background-color: #121212;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body, .stApp {
                background-color: #ffffff;
                color: black;
            }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("App Developed for **Advanced Topics in Data Analytics** at UEA.")
