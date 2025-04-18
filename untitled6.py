import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Set Page Config (Must be first Streamlit call) ---
st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# --- Load Data ---
sao = pd.read_excel("ADAinB.xlsx")
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
combined_data = pd.read_excel("Combined_BTP_ONS_Quarterly_Data.xlsx")

# --- Preprocess BTP data for tab 2 ---
sao['Month'] = pd.to_datetime(sao['Month'], format="%Y-%m")
sao['Quarter'] = sao['Month'].dt.to_period('Q').astype(str)

# --- Tabs for Navigation ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔗 Crime vs Well-being",
    "🗺️ Region Explorer", "📄 Raw Data"
])

# --- TAB 1: Overview ---
with tab1:
    st.title("🚨 UK Crime and Public Well-being Dashboard")
    st.markdown("### Overview (2022–2024)")
    st.markdown("Gain insights into transport-related crime trends and well-being metrics across the UK.")

    # KPI Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Reported Crimes", f"{sao.shape[0]:,}")
    col2.metric("😊 Avg. Life Satisfaction", f"{combined_data['Life_Satisfaction_Mean_Score'].mean():.2f}")
    col3.metric("😟 Avg. Anxiety", f"{combined_data['Anxiety_Mean_Score'].mean():.2f}")

    st.markdown("---")
    col4, col5 = st.columns(2)

    # Bar chart: Total crimes per quarter
    with col4:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=combined_data, x='Quarter', y='Total_Crimes', ax=ax1, palette='Reds')
        ax1.set_title("Total Crimes per Quarter")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    # Pie chart: Top 6 Crime Types
    with col5:
        top_crimes = sao['Crime type'].value_counts().nlargest(6)
        fig2, ax2 = plt.subplots()
        ax2.pie(top_crimes, labels=top_crimes.index, autopct='%1.1f%%', startangle=140)
        ax2.set_title("Top 6 Crime Types Distribution")
        st.pyplot(fig2)

    st.markdown("---")
    col6, col7 = st.columns(2)

    # Life Satisfaction bar chart
    with col6:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=combined_data, x='Quarter', y='Life_Satisfaction_Mean_Score', ax=ax3, palette='Blues')
        ax3.set_title("Life Satisfaction per Quarter")
        ax3.tick_params(axis='x', rotation=45)
        st.pyplot(fig3)

    # Anxiety bar chart
    with col7:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=combined_data, x='Quarter', y='Anxiety_Mean_Score', ax=ax4, palette='Purples')
        ax4.set_title("Anxiety per Quarter")
        ax4.tick_params(axis='x', rotation=45)
        st.pyplot(fig4)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Data: British Transport Police & ONS")


# --- TAB 2: Crime Trends ---
with tab2:
    st.header("📈 Crime Trends Over Time")

    # Dropdown to filter by crime type
    crime_types = sorted(sao['Crime type'].dropna().unique())
    selected_crime = st.selectbox("Select Crime Type", crime_types)

    # Filter by crime type and group by quarter
    filtered = sao[sao['Crime type'] == selected_crime].copy()
    trend = filtered.groupby('Quarter').size().reset_index(name='Count')

    # Line chart: Crime trend over time
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=trend, x='Quarter', y='Count', marker='o', ax=ax5)
    ax5.set_title(f"{selected_crime} Trend Over Time")
    ax5.tick_params(axis='x', rotation=45)
    st.pyplot(fig5)

    # 🗺️ Map of recent crimes
    st.subheader(f"🗺️ Locations of {selected_crime} (Latest Quarter Only)")
    latest_quarter = filtered['Quarter'].sort_values().iloc[-1]

    map_data = filtered[filtered['Quarter'] == latest_quarter][['Latitude', 'Longitude']].dropna()

    # ✅ Rename columns for Streamlit compatibility
    map_data = map_data.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})

    # ✅ Ensure correct data types
    map_data["latitude"] = map_data["latitude"].astype(float)
    map_data["longitude"] = map_data["longitude"].astype(float)

    if not map_data.empty:
        st.map(map_data, zoom=6)
    else:
        st.warning("No location data available for this crime type in the latest quarter.")

