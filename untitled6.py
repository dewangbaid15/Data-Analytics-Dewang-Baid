import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="UK Crime & Well-being Overview", layout="wide")  # ✅ FIRST ST COMMAND

st.title("🚨 UK Crime and Public Well-being Dashboard")
st.markdown("### Overview (2022–2024)")
st.markdown("Gain insights into transport-related crime trends and well-being metrics across the UK.")

# Placeholder data (replace with your actual cleaned dataset)
btp_data = pd.read_excel("ADAinB.xlsx")
ons_data = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
combined_data = pd.read_excel("Combined_BTP_ONS_Quarterly_Data.xlsx")

# KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("📊 Total Reported Crimes", f"{btp_data.shape[0]:,}")
col2.metric("😊 Avg. Life Satisfaction", f"{combined_data['Life_Satisfaction_Mean_Score'].mean():.2f}")
col3.metric("😟 Avg. Anxiety", f"{combined_data['Anxiety_Mean_Score'].mean():.2f}")

st.markdown("---")

# Layout for charts
col4, col5 = st.columns(2)

# Bar Chart: Total Crimes per Quarter
with col4:
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=combined_data, x='Quarter', y='Total_Crimes', ax=ax1, palette='Reds')
    ax1.set_title("Total Crimes per Quarter")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

# Pie Chart: Crime Types (Top 6)
with col5:
    crime_counts = btp_data['Crime type'].value_counts().nlargest(6)
    fig2, ax2 = plt.subplots()
    ax2.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140)
    ax2.set_title("Top 6 Crime Types Distribution")
    st.pyplot(fig2)

st.markdown("---")

# Side-by-side bar charts for well-being
col6, col7 = st.columns(2)

# Bar Chart: Life Satisfaction per Quarter
with col6:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=combined_data, x='Quarter', y='Life_Satisfaction_Mean_Score', ax=ax3, palette='Blues')
    ax3.set_title("Life Satisfaction per Quarter")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

# Bar Chart: Anxiety per Quarter
with col7:
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=combined_data, x='Quarter', y='Anxiety_Mean_Score', ax=ax4, palette='Purples')
    ax4.set_title("Anxiety per Quarter")
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data: British Transport Police & ONS")

