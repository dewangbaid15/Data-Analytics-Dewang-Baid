import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Set Page Config ---
st.set_page_config(page_title="UK Crime & Well-being Dashboard", layout="wide")

# --- Load Data ---
sao = pd.read_excel("ADAinB.xlsx")
ons_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
combined_data = pd.read_excel("Combined_BTP_ONS_Quarterly_Data.xlsx")

# --- Preprocessing ---
sao['Month'] = pd.to_datetime(sao['Month'], format="%Y-%m")
sao['Quarter'] = sao['Month'].dt.to_period('Q').astype(str)
sao['Quarter_dt'] = sao['Month'].dt.to_period('Q').dt.to_timestamp() + pd.offsets.QuarterEnd(0)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview", "📈 Crime Trends", "😊 Well-being Trends", "🔍 Deep Dive",
    "🌍 Region Map", "📄 Raw Data", "🧪 Predictive Insights", "⚙️ Settings"
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

# --- TAB 2: Crime Trends ---
with tab2:
    st.header("📈 Crime Trends Explorer")
    crime_types = sorted(sao['Crime type'].dropna().unique())
    selected_crimes = st.multiselect("Select Crime Types", crime_types, default=crime_types[:2])

    if selected_crimes:
        filtered = sao[sao['Crime type'].isin(selected_crimes)].copy()
        crime_trend = filtered.groupby(['Quarter', 'Crime type']).size().reset_index(name='Count')

        fig3 = px.bar(
            crime_trend,
            x='Crime type', y='Count',
            color='Crime type',
            animation_frame='Quarter',
            title="Animated Crime Trends by Type Over Quarters",
            range_y=[0, crime_trend['Count'].max() * 1.2],
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        
        st.subheader("📊 Crime Type Composition Over Time (Percentage)")
        fig_area = px.area(
            crime_trend,
            x="Quarter", y="Count", color="Crime type",
            title="Proportional Crime Type Trends Over Time",
            groupnorm="percent"
            )
        st.plotly_chart(fig_area, use_container_width=True)


        st.subheader("🗺️ Crime Locations Map")
        latest_quarter = filtered['Quarter'].sort_values().iloc[-1]
        last_map = filtered[
            (filtered['Quarter'] == latest_quarter) & 
            (filtered['Crime type'] == selected_crimes[0])
        ]
        map_data = last_map[['Latitude', 'Longitude']].dropna().rename(
            columns={"Latitude": "latitude", "Longitude": "longitude"}
        )
        if not map_data.empty:
            st.map(map_data, zoom=5)
        else:
            st.warning("No location data available for this crime type in the latest quarter.")
    else:
        st.warning("Please select at least one crime type to view trends and map.")

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

    st.markdown("---")
    selected_quarter = st.selectbox("Compare by Quarter", sorted(ons_area['Quarter'].unique()))
    compare_df = ons_area[ons_area['Quarter'] == selected_quarter]

    col6, col7 = st.columns(2)

    with col6:
        fig5 = px.bar(
            compare_df,
            x='Life_Satisfaction_Mean_Score', y='Area',
            orientation='h',
            color='Life_Satisfaction_Mean_Score',
            color_continuous_scale='Blues',
            title=f"Life Satisfaction by Area ({selected_quarter})"
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col7:
        fig6 = px.bar(
            compare_df,
            x='Anxiety_Mean_Score', y='Area',
            orientation='h',
            color='Anxiety_Mean_Score',
            color_continuous_scale='Purples',
            title=f"Anxiety by Area ({selected_quarter})"
        )
        st.plotly_chart(fig6, use_container_width=True)

# --- TAB 4: Deep Dive ---
with tab4:
    st.header("🔍 Deep Dive: Crime vs Well-being")
    fig7 = px.scatter(
        combined_data,
        x='Total_Crimes', y='Life_Satisfaction_Mean_Score',
        trendline='ols',
        title="Crime vs Life Satisfaction"
    )
    st.plotly_chart(fig7, use_container_width=True)

    fig8 = px.scatter(
        combined_data,
        x='Total_Crimes', y='Anxiety_Mean_Score',
        trendline='ols',
        title="Crime vs Anxiety"
    )
    st.plotly_chart(fig8, use_container_width=True)

    corr_ls, p_ls = pearsonr(combined_data['Total_Crimes'], combined_data['Life_Satisfaction_Mean_Score'])
    corr_anx, p_anx = pearsonr(combined_data['Total_Crimes'], combined_data['Anxiety_Mean_Score'])

    st.metric("Correlation (Crime & Life Satisfaction)", f"{corr_ls:.2f}", delta=f"p = {p_ls:.3f}")
    st.metric("Correlation (Crime & Anxiety)", f"{corr_anx:.2f}", delta=f"p = {p_anx:.3f}")

# --- TAB 5: Region Map ---
with tab5:
    st.header("🌍 Regional Overview")
    
    # --- Heatmap Prep ---
    heatmap_df = ons_area.pivot(index='Area', columns='Quarter', values='Life_Satisfaction_Mean_Score')

    area_avg = ons_area.groupby("Area").agg({
        "Life_Satisfaction_Mean_Score": "mean",
        "Anxiety_Mean_Score": "mean"
    }).reset_index()

    col8, col9 = st.columns(2)

    with col8:
        fig9 = px.bar(
            area_avg.sort_values(by='Life_Satisfaction_Mean_Score'),
            x='Life_Satisfaction_Mean_Score', y='Area',
            color='Life_Satisfaction_Mean_Score',
            title="Avg. Life Satisfaction by Area",
            orientation='h',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig9, use_container_width=True)

    with col9:
        fig10 = px.bar(
            area_avg.sort_values(by='Anxiety_Mean_Score'),
            x='Anxiety_Mean_Score', y='Area',
            color='Anxiety_Mean_Score',
            title="Avg. Anxiety by Area",
            orientation='h',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig10, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🧊 Heatmap: Life Satisfaction by Area and Quarter")
        st.dataframe(heatmap_df.style.background_gradient(cmap='Blues'))

# --- TAB 6: Raw Data Viewer ---
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
    st.markdown("Predict crime volume trends using simple regression.")

    # Prepare model input
    combined_data['Quarter_Index'] = range(1, len(combined_data) + 1)
    X = combined_data[['Quarter_Index']]
    y = combined_data['Total_Crimes']

    # Fit regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    combined_data['Predicted_Crimes'] = y_pred

    # Forecast future quarters
    st.subheader("🔮 Forecast Future Quarters")
    future = st.slider("How many quarters ahead to predict?", 1, 6, 3)
    future_index = pd.DataFrame({'Quarter_Index': range(len(X)+1, len(X)+1+future)})
    future_preds = model.predict(future_index)

    # Generate future quarter labels
    last_known_quarter = combined_data['Quarter'].iloc[-1]
    last_year = int(last_known_quarter[:4])
    last_q = int(last_known_quarter[-1])
    future_quarters = []
    for i in range(1, future + 1):
        q = (last_q + i - 1) % 4 + 1
        y = last_year + ((last_q + i - 1) // 4)
        future_quarters.append(f"{y}Q{q}")

    # Create future DataFrame
    future_df = pd.DataFrame({
        "Quarter": future_quarters,
        "Total_Crimes": [None]*future,
        "Predicted_Crimes": future_preds
    })

    # Combine with original data
    plot_df = pd.concat([
        combined_data[['Quarter', 'Total_Crimes', 'Predicted_Crimes']],
        future_df
    ], ignore_index=True)

    # Reshape for Plotly
    plot_long = plot_df.melt(id_vars='Quarter', value_vars=['Total_Crimes', 'Predicted_Crimes'],
                             var_name='Type', value_name='Crimes')
    plot_long = plot_long.dropna(subset=['Crimes'])

    # Plot
    fig_pred = px.line(
        plot_long,
        x='Quarter',
        y='Crimes',
        color='Type',
        markers=True,
        title="📉 Actual vs Predicted Crime Trends (With Forecast)"
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
    r2_score = model.score(X, y)
    st.success(f"Model R² Score: {r2_score:.2f}")
    
    
# --- TAB 8: Settings ---
with tab8:
    st.header("⚙️ Dashboard Settings")
    dark_mode = st.toggle("🌙 Toggle Dark Mode")

    if dark_mode:
        st.markdown("""
            <style>
            body, .stApp { background-color: #121212; color: white; }
            .css-18ni7ap { background-color: #121212 !important; }
            .css-1cpxqw2, .stMarkdown, .stDataFrame {
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body, .stApp {
                background-color: #f5f5f5 !important;
                color: #000000 !important;
            }
            .css-18ni7ap, .stMarkdown, .stDataFrame, .css-1cpxqw2, 
            .stTextInput, .stSelectbox, .stMultiSelect {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            .css-1d391kg, .css-qbe2hs {
                color: #000000 !important;
            }
            .block-container {
                background-color: #fefefe !important;
            }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**App Developed for Analytical Exploration of Crime & Well-being Trends in the UK**")