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

# --- TAB 3: Well-being Trends ---
with tab3:
    st.header("😊 Well-being Trends (ONS Area Data)")
    st.success("✅ Tab 3 loaded successfully")

    available_areas = sorted(ons_area['Area'].dropna().unique())
    selected_area = st.selectbox("Select Region (Area)", available_areas)

    filtered_area = ons_area[ons_area['Area'] == selected_area]

    fig6, ax6 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_area, x='Quarter', y='Life_Satisfaction_Mean_Score', label='Life Satisfaction', marker='o')
    sns.lineplot(data=filtered_area, x='Quarter', y='Anxiety_Mean_Score', label='Anxiety', marker='o')
    ax6.set_title(f"Well-being Trends in {selected_area}")
    ax6.set_ylabel("Mean Score")
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend()
    st.pyplot(fig6)

    st.markdown("---")
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

    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=combined_data, x='Total_Crimes', y='Life_Satisfaction_Mean_Score', ax=ax9)
    ax9.set_title("Total Crimes vs Life Satisfaction")
    st.pyplot(fig9)

    fig10, ax10 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=combined_data, x='Total_Crimes', y='Anxiety_Mean_Score', ax=ax10)
    ax10.set_title("Total Crimes vs Anxiety")
    st.pyplot(fig10)

    st.markdown("---")
    corr_ls, p_ls = pearsonr(combined_data['Total_Crimes'], combined_data['Life_Satisfaction_Mean_Score'])
    corr_anx, p_anx = pearsonr(combined_data['Total_Crimes'], combined_data['Anxiety_Mean_Score'])

    st.metric("📊 Correlation (Crime & Life Satisfaction)", f"{corr_ls:.2f}")
    st.metric("📊 Correlation (Crime & Anxiety)", f"{corr_anx:.2f}")

# --- TAB 5: Region Explorer (Map) ---
with tab5:
    st.header("🗺️ Map & Region Explorer")
    st.write("*Explore average well-being and crime data across regions.*")

    area_avg = ons_area.groupby("Area").agg({
        "Life_Satisfaction_Mean_Score": "mean",
        "Anxiety_Mean_Score": "mean"
    }).reset_index()

    col10, col11 = st.columns(2)
    with col10:
        fig11, ax11 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=area_avg.sort_values(by='Life_Satisfaction_Mean_Score'), x='Life_Satisfaction_Mean_Score', y='Area', palette='coolwarm', ax=ax11)
        ax11.set_title("Avg. Life Satisfaction by Area")
        st.pyplot(fig11)

    with col11:
        fig12, ax12 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=area_avg.sort_values(by='Anxiety_Mean_Score'), x='Anxiety_Mean_Score', y='Area', palette='magma', ax=ax12)
        ax12.set_title("Avg. Anxiety by Area")
        st.pyplot(fig12)

# --- TAB 6: Raw Data ---
with tab6:
    st.header("📄 View & Download Raw Data")
    data_option = st.radio("Select Dataset", ("Combined Data", "BTP Data", "ONS Area Data"))

    if data_option == "Combined Data":
        st.dataframe(combined_data)
    elif data_option == "BTP Data":
        st.dataframe(sao)
    else:
        st.dataframe(ons_area)

    st.download_button("📥 Download Displayed Data as CSV", data=eval(data_option.replace(" ", "_"))
                      .to_csv(index=False).encode('utf-8'),
                      file_name=f"{data_option.replace(' ', '_')}.csv")
