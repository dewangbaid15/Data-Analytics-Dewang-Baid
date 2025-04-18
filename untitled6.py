import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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
    
    # Bar chart: Total crimes per quarter
    with col4:
        fig1 = px.bar(
            combined_data, x='Quarter', y='Total_Crimes',
            title="Total Crimes per Quarter",
            labels={'Total_Crimes': 'Total Crimes'},
            color='Total_Crimes',
            color_continuous_scale='Reds'
        )
        fig1.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    # Pie chart: Top 6 Crime Types
    with col5:
        top_crimes = sao['Crime type'].value_counts().nlargest(6).reset_index()
        top_crimes.columns = ['Crime type', 'Count']
        fig2 = px.pie(top_crimes, names='Crime type', values='Count', title="Top 6 Crime Types Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col6, col7 = st.columns(2)

    # Life Satisfaction bar chart
    with col6:
        fig3 = px.bar(
            combined_data, x='Quarter', y='Life_Satisfaction_Mean_Score',
            title="Life Satisfaction per Quarter",
            labels={'Life_Satisfaction_Mean_Score': 'Mean Score'},
            color='Life_Satisfaction_Mean_Score',
            color_continuous_scale='Blues'
        )
        fig3.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # Anxiety bar chart
    with col7:
        fig4 = px.bar(
            combined_data, x='Quarter', y='Anxiety_Mean_Score',
            title="Anxiety per Quarter",
            labels={'Anxiety_Mean_Score': 'Mean Score'},
            color='Anxiety_Mean_Score',
            color_continuous_scale='Purples'
        )
        fig4.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, Plotly, and pandas | Data: British Transport Police & ONS")

        
        
        
# --- TAB 2: Crime Trends ---
with tab2:
    st.header("📈 Crime Trends by Type & Quarter")
    
    # Filter and animate
    crime_types = sorted(sao['Crime type'].dropna().unique())
    selected_crimes = st.multiselect("Select Crime Types", crime_types, default=crime_types[:3])
    filtered_sao = sao[sao['Crime type'].isin(selected_crimes)]

    crime_anim = filtered_sao.groupby(['Quarter', 'Crime type']).size().reset_index(name='Count')

    fig = px.bar(
        crime_anim,
        x="Crime type",
        y="Count",
        color="Crime type",
        animation_frame="Quarter",
        range_y=[0, crime_anim['Count'].max() + 500],
        title="Crime Trends Over Time"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    

# --- TAB 3: Well-being Trends ---
# --- TAB 3: Well-being Trends ---
with tab3:
    st.header("😊 Well-being Trends by Region")

    # Select Area
    areas = sorted(ons_area['Area'].dropna().unique())
    selected_areas = st.multiselect("Select Region(s)", areas, default=areas[:3])
    filtered_ons = ons_area[ons_area['Area'].isin(selected_areas)]

    # Line Charts: Trends Over Time
    fig5 = px.line(
        filtered_ons,
        x="Quarter",
        y="Life_Satisfaction_Mean_Score",
        color="Area",
        title="Life Satisfaction Trends"
    )
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.line(
        filtered_ons,
        x="Quarter",
        y="Anxiety_Mean_Score",
        color="Area",
        title="Anxiety Trends"
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # Additional Bar Charts: Compare by Quarter
    st.subheader("📊 Compare Regions in a Specific Quarter")
    selected_quarter = st.selectbox("Select Quarter", sorted(ons_area['Quarter'].unique()))

    compare_df = ons_area[ons_area['Quarter'] == selected_quarter]

    col8, col9 = st.columns(2)

    with col8:
        fig7 = px.bar(
            compare_df,
            x='Life_Satisfaction_Mean_Score',
            y='Area',
            orientation='h',
            title=f"Life Satisfaction by Area ({selected_quarter})",
            color='Life_Satisfaction_Mean_Score',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col9:
        fig8 = px.bar(
            compare_df,
            x='Anxiety_Mean_Score',
            y='Area',
            orientation='h',
            title=f"Anxiety by Area ({selected_quarter})",
            color='Anxiety_Mean_Score',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig8, use_container_width=True)




# --- TAB 4: Crime vs Well-being ---
with tab4:
    st.header("🔗 Crime vs Well-being Correlation")

    # Scatter Plot: Crimes vs Life Satisfaction
    fig9 = px.scatter(
        combined_data,
        x='Total_Crimes',
        y='Life_Satisfaction_Mean_Score',
        trendline='ols',
        title='Total Crimes vs Life Satisfaction',
        labels={'Total_Crimes': 'Total Crimes', 'Life_Satisfaction_Mean_Score': 'Life Satisfaction'}
    )
    st.plotly_chart(fig9, use_container_width=True)

    # Scatter Plot: Crimes vs Anxiety
    fig10 = px.scatter(
        combined_data,
        x='Total_Crimes',
        y='Anxiety_Mean_Score',
        trendline='ols',
        title='Total Crimes vs Anxiety',
        labels={'Total_Crimes': 'Total Crimes', 'Anxiety_Mean_Score': 'Anxiety'}
    )
    st.plotly_chart(fig10, use_container_width=True)

    # Correlation Stats
    st.markdown("---")
    corr_ls, p_ls = pearsonr(combined_data['Total_Crimes'], combined_data['Life_Satisfaction_Mean_Score'])
    corr_anx, p_anx = pearsonr(combined_data['Total_Crimes'], combined_data['Anxiety_Mean_Score'])

    st.metric("📈 Correlation with Life Satisfaction", f"{corr_ls:.2f}", delta=f"p = {p_ls:.3f}")
    st.metric("📉 Correlation with Anxiety", f"{corr_anx:.2f}", delta=f"p = {p_anx:.3f}")



# --- TAB 5: Region Explorer ---
with tab5:
    st.header("🗺️ Region Explorer")

    st.markdown("Explore average well-being metrics across UK regions.")

    # Aggregated averages by area
    area_avg = ons_area.groupby("Area").agg({
        "Life_Satisfaction_Mean_Score": "mean",
        "Anxiety_Mean_Score": "mean"
    }).reset_index()

    col14, col15 = st.columns(2)

    with col14:
        fig11 = px.bar(
            area_avg.sort_values(by='Life_Satisfaction_Mean_Score'),
            x='Life_Satisfaction_Mean_Score',
            y='Area',
            orientation='h',
            title="Average Life Satisfaction by Area",
            color='Life_Satisfaction_Mean_Score',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig11, use_container_width=True)

    with col15:
        fig12 = px.bar(
            area_avg.sort_values(by='Anxiety_Mean_Score'),
            x='Anxiety_Mean_Score',
            y='Area',
            orientation='h',
            title="Average Anxiety by Area",
            color='Anxiety_Mean_Score',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig12, use_container_width=True)

    st.markdown("---")




# --- TAB 6: Raw Data Viewer ---
with tab6:
    st.header("📄 Raw Data Viewer")

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
