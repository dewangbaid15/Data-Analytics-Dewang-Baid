import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔗 Crime vs Well-being",
    "🗺️ Region Explorer", "📄 Raw Data"
])

# --- TAB 1: Overview ---
with tab1:
    st.title("🚨 UK Crime and Public Well-being Dashboard")
    st.markdown("### Overview (2022–2024)")
    st.markdown("Gain insights into transport-related crime trends and well-being metrics across the UK.")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Reported Crimes", f"{sao.shape[0]:,}")
    col2.metric("😊 Avg. Life Satisfaction", f"{combined_data['Life_Satisfaction_Mean_Score'].mean():.2f}")
    col3.metric("😟 Avg. Anxiety", f"{combined_data['Anxiety_Mean_Score'].mean():.2f}")

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=combined_data, x='Quarter', y='Total_Crimes', ax=ax1, palette='Reds')
        ax1.set_title("Total Crimes per Quarter")
        ax1.tick_params(axis='x', rotation=45)
        for i, val in enumerate(combined_data['Total_Crimes']):
            ax1.text(i, val + 200, f"{val:,}", ha='center', va='bottom', fontsize=8)
        st.pyplot(fig1)

    with col5:
        top_crimes = sao['Crime type'].value_counts().nlargest(6)
        fig2, ax2 = plt.subplots()
        ax2.pie(top_crimes, labels=top_crimes.index, autopct='%1.1f%%', startangle=140)
        ax2.set_title("Top 6 Crime Types Distribution")
        st.pyplot(fig2)

    st.markdown("---")
    col6, col7 = st.columns(2)

    with col6:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=combined_data, x='Quarter', y='Life_Satisfaction_Mean_Score', ax=ax3, palette='Blues')
        ax3.set_title("Life Satisfaction per Quarter")
        ax3.tick_params(axis='x', rotation=45)
        for i, val in enumerate(combined_data['Life_Satisfaction_Mean_Score']):
            ax3.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
        st.pyplot(fig3)

    with col7:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=combined_data, x='Quarter', y='Anxiety_Mean_Score', ax=ax4, palette='Purples')
        ax4.set_title("Anxiety per Quarter")
        ax4.tick_params(axis='x', rotation=45)
        for i, val in enumerate(combined_data['Anxiety_Mean_Score']):
            ax4.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)
        st.pyplot(fig4)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Data: British Transport Police & ONS")

# --- TAB 2: Crime Trends ---
with tab2:
    st.header("📈 Crime Trends Over Time")
    crime_types = sorted(sao['Crime type'].dropna().unique())
    selected_crime = st.selectbox("Select Crime Type", crime_types)

    filtered = sao[sao['Crime type'] == selected_crime].copy()
    trend = filtered.groupby('Quarter').size().reset_index(name='Count')

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=trend, x='Quarter', y='Count', marker='o', ax=ax5)
    ax5.set_title(f"{selected_crime} Trend Over Time")
    ax5.tick_params(axis='x', rotation=45)
    st.pyplot(fig5)

    st.subheader(f"🗺️ Locations of {selected_crime} (Latest Quarter Only)")
    latest_quarter = filtered['Quarter'].sort_values().iloc[-1]
    map_data = filtered[filtered['Quarter'] == latest_quarter][['Latitude', 'Longitude']].dropna()
    map_data = map_data.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    map_data["latitude"] = map_data["latitude"].astype(float)
    map_data["longitude"] = map_data["longitude"].astype(float)

    if not map_data.empty:
        st.map(map_data, zoom=6)
    else:
        st.warning("No location data available for this crime type in the latest quarter.")



# --- TAB 3: Well-being Trends ---
with tab3:
    st.header("😊 Well-being Trends (ONS Area Data)")
    st.success("✅ Tab 3 loaded successfully")

    # Area Filter
    available_areas = sorted(ons_area['Area'].dropna().unique())
    selected_area = st.selectbox("Select Region (Area)", available_areas)

    filtered_area = ons_area[ons_area['Area'] == selected_area]

    # Line Chart: Life Satisfaction and Anxiety
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_area, x='Quarter', y='Life_Satisfaction_Mean_Score', label='Life Satisfaction', marker='o')
    sns.lineplot(data=filtered_area, x='Quarter', y='Anxiety_Mean_Score', label='Anxiety', marker='o')
    ax6.set_title(f"Well-being Trends in {selected_area}")
    ax6.set_ylabel("Mean Score")
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend()
    st.pyplot(fig6)

    st.markdown("---")

    # Bar chart comparing areas in selected quarter
    selected_quarter = st.selectbox("Compare by Quarter", sorted(ons_area['Quarter'].unique()))
    compare_df = ons_area[ons_area['Quarter'] == selected_quarter]

    col8, col9 = st.columns(2)

    with col8:
        fig7, ax7 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=compare_df, x='Life_Satisfaction_Mean_Score', y='Area', palette='Blues', ax=ax7)
        ax7.set_title(f"Life Satisfaction by Area ({selected_quarter})")
        for i, val in enumerate(compare_df['Life_Satisfaction_Mean_Score']):
            ax7.text(val + 0.02, i, f"{val:.2f}", va='center')
        st.pyplot(fig7)

    with col9:
        fig8, ax8 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=compare_df, x='Anxiety_Mean_Score', y='Area', palette='Purples', ax=ax8)
        ax8.set_title(f"Anxiety by Area ({selected_quarter})")
        for i, val in enumerate(compare_df['Anxiety_Mean_Score']):
            ax8.text(val + 0.02, i, f"{val:.2f}", va='center')
        st.pyplot(fig8)



# --- TAB 4: Crime vs Well-being ---
with tab4:
    st.header("🔗 Relationship Between Crime & Well-being")

    # Scatter Plot: Crimes vs Life Satisfaction
    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=combined_data, x='Total_Crimes', y='Life_Satisfaction_Mean_Score', ax=ax9)
    ax9.set_title("Total Crimes vs Life Satisfaction")
    st.pyplot(fig9)

    # Scatter Plot: Crimes vs Anxiety
    fig10, ax10 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=combined_data, x='Total_Crimes', y='Anxiety_Mean_Score', ax=ax10)
    ax10.set_title("Total Crimes vs Anxiety")
    st.pyplot(fig10)

    st.markdown("---")

    # Correlation (Pearson)
    from scipy.stats import pearsonr
    corr_ls, p_ls = pearsonr(combined_data['Total_Crimes'], combined_data['Life_Satisfaction_Mean_Score'])
    corr_anx, p_anx = pearsonr(combined_data['Total_Crimes'], combined_data['Anxiety_Mean_Score'])

    col12, col13 = st.columns(2)
    with col12:
        st.metric("📊 Correlation (Crime & Life Satisfaction)", f"{corr_ls:.2f}", delta=f"p = {p_ls:.3f}")
    with col13:
        st.metric("📊 Correlation (Crime & Anxiety)", f"{corr_anx:.2f}", delta=f"p = {p_anx:.3f}")
        
        
        
        
# --- TAB 5: Region Explorer (Map & Area Comparison) ---
with tab5:
    st.header("🗺️ Region Explorer")

    # Aggregated averages by area
    area_avg = ons_area.groupby("Area").agg({
        "Life_Satisfaction_Mean_Score": "mean",
        "Anxiety_Mean_Score": "mean"
    }).reset_index()

    col14, col15 = st.columns(2)

    with col14:
        fig11, ax11 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=area_avg.sort_values(by='Life_Satisfaction_Mean_Score'),
            x='Life_Satisfaction_Mean_Score',
            y='Area',
            palette='coolwarm',
            ax=ax11
        )
        ax11.set_title("Avg. Life Satisfaction by Area")
        for i, val in enumerate(area_avg.sort_values(by='Life_Satisfaction_Mean_Score')['Life_Satisfaction_Mean_Score']):
            ax11.text(val + 0.02, i, f"{val:.2f}", va='center')
        st.pyplot(fig11)

    with col15:
        fig12, ax12 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=area_avg.sort_values(by='Anxiety_Mean_Score'),
            x='Anxiety_Mean_Score',
            y='Area',
            palette='magma',
            ax=ax12
        )
        ax12.set_title("Avg. Anxiety by Area")
        for i, val in enumerate(area_avg.sort_values(by='Anxiety_Mean_Score')['Anxiety_Mean_Score']):
            ax12.text(val + 0.02, i, f"{val:.2f}", va='center')
        st.pyplot(fig12)



# --- TAB 6: Raw Data Viewer ---
with tab6:
    st.header("📄 Raw Data Explorer & Download")

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

