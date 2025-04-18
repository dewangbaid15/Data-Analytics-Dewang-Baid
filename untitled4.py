import pandas as pd



sao = pd.read_excel("ADAinB.xlsx")

print(sao.head())

print("Shape:", sao.shape)

print("\nColumn Info:")
print(sao.dtypes)


print("\nMissing Values:")
print(sao.isnull().sum())


print("Column Names:")
print(sao.columns.tolist())


print("\nSample Data:")
print(sao.head(10))


print("\nUnique Crime Types:")
print(sao['Crime type'].unique())

print("\nUnique Months:")
print(sao['Month'].unique())

print("\nUnique LSOA Codes (non-null):")
print(sao['LSOA code'].dropna().unique()[:10])



df_age_group = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Age Group")
df_area = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Area")
df_gender = pd.read_excel("Life_Satisfaction_Anxiety_All_Quarters.xlsx", sheet_name="Gender")

{
    "Age Group": {
        "shape": df_age_group.shape,
        "columns": df_age_group.columns.tolist(),
        "preview": df_age_group.head(3)
    },
    "Area": {
        "shape": df_area.shape,
        "columns": df_area.columns.tolist(),
        "preview": df_area.head(3)
    },
    "Gender": {
        "shape": df_gender.shape,
        "columns": df_gender.columns.tolist(),
        "preview": df_gender.head(3)
    }
}




sao['Month'] = pd.to_datetime(sao['Month'], format="%Y-%m")
sao['Year'] = sao['Month'].dt.year
sao['Quarter'] = sao['Month'].dt.to_period('Q').astype(str)


print("\nCrime Counts per Quarter:")
print(sao['Quarter'].value_counts().sort_index())


print("\nUpdated BTP DataFrame:")
print(sao[['Month', 'Year', 'Quarter', 'Crime type']].head(10))





print("\nONS Area Columns:")
print(df_area.columns.tolist())


ons_quarterly_avg = df_area.groupby('Quarter').agg({
    'Life_Satisfaction_Mean_Score': 'mean',
    'Anxiety_Mean_Score': 'mean'
}).reset_index()


ons_quarterly_avg = ons_quarterly_avg.round(2)


print("\nQuarterly Averages from ONS (Area data):")
print(ons_quarterly_avg)






ons_file_path = "Life_Satisfaction_Anxiety_All_Quarters.xlsx"
ons_area = pd.read_excel(ons_file_path, sheet_name="Area")

sao['Month'] = pd.to_datetime(sao['Month'], format="%Y-%m")
sao['Quarter'] = sao['Month'].dt.to_period('Q').astype(str)


crime_volume = sao.groupby("Quarter").size().reset_index(name="Total_Crimes")


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


plt.figure(figsize=(10, 5))
sns.lineplot(data=crime_volume, x="Quarter", y="Total_Crimes", marker="o")
plt.xticks(rotation=45)
plt.title("Total Crimes per Quarter (BTP Data)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(data=ons_area['Life_Satisfaction_Mean_Score'])
plt.title("Boxplot of Life Satisfaction Mean Score (ONS)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(data=ons_area['Anxiety_Mean_Score'])
plt.title("Boxplot of Anxiety Mean Score (ONS)")
plt.tight_layout()
plt.show()





btp_quarterly = sao.groupby("Quarter").size().reset_index(name="Total_Crimes")


ons_quarterly = ons_area.groupby("Quarter").agg({
    'Life_Satisfaction_Mean_Score': 'mean',
    'Anxiety_Mean_Score': 'mean'
}).reset_index()


combined_data = pd.merge(btp_quarterly, ons_quarterly, on="Quarter", how="inner")


combined_data = combined_data.round(2)


combined_output_path = "Combined_BTP_ONS_Quarterly_Data.xlsx"
combined_data.to_excel(combined_output_path, index=False)

combined_output_path



btp_quarterly['Quarter'] = btp_quarterly['Quarter'].str.replace(" ", "")
ons_quarterly['Quarter'] = ons_quarterly['Quarter'].str.replace(" ", "")

combined_data = pd.merge(btp_quarterly, ons_quarterly, on="Quarter", how="inner")


print(combined_data.head())

combined_output_path = "Combined_BTP_ONS_Quarterly_Data.xlsx"
combined_data.to_excel(combined_output_path, index=False)


from scipy.stats import pearsonr
combined_data = pd.read_excel("Combined_BTP_ONS_Quarterly_Data.xlsx")


combined_data = combined_data.dropna()


cor_life_sat, p_life_sat = pearsonr(combined_data['Total_Crimes'], combined_data['Life_Satisfaction_Mean_Score'])
cor_anxiety, p_anxiety = pearsonr(combined_data['Total_Crimes'], combined_data['Anxiety_Mean_Score'])


plt.figure(figsize=(6, 4))
sns.regplot(x='Total_Crimes', y='Life_Satisfaction_Mean_Score', data=combined_data)
plt.title('Total Crimes vs Life Satisfaction')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.regplot(x='Total_Crimes', y='Anxiety_Mean_Score', data=combined_data)
plt.title('Total Crimes vs Anxiety')
plt.tight_layout()
plt.show()

{
    "Correlation with Life Satisfaction": round(cor_life_sat, 3),
    "P-value (Life Satisfaction)": round(p_life_sat, 3),
    "Correlation with Anxiety": round(cor_anxiety, 3),
    "P-value (Anxiety)": round(p_anxiety, 3)
}








import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="UK Crime & Well-being Overview", layout="wide")  # ✅ FIRST ST COMMAND

st.title("🚨 UK Crime and Public Well-being Dashboard")
st.markdown("### Overview (2022–2024)")
st.markdown("Gain insights into transport-related crime trends and well-being metrics across the UK.")

# Placeholder data (replace with your actual cleaned dataset)
btp_data = pd.read_excel("Desktop/ADAinB.xlsx")
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

