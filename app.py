import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Config
warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
try:
    df = pd.read_csv("cleaned_titanic.csv")
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'cleaned_titanic.csv' exists.")
    st.stop()

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("Filter Options")
gender_options = ["All"] + df["Sex"].unique().tolist()
gender = st.sidebar.selectbox("Select Gender", options=gender_options)

pclass_options = ["All"] + sorted(df["Pclass"].unique().tolist())
pclass = st.sidebar.selectbox("Select Passenger Class", options=pclass_options)

# Apply Filters
filtered_df = df.copy()
if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]
if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass]

# Preview Filtered Data
st.subheader("🔍 Filtered Data Preview")
st.write(filtered_df.head())

# 1. Survival Count by Gender
st.subheader("📊 Survival Count by Gender")
filtered_df["Survival Status"] = filtered_df["Survived"].map({0: "Did Not Survive", 1: "Survived"})
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.countplot(data=filtered_df, x="Survival Status", hue="Sex", ax=ax1, dodge=True, width=0.5)
ax1.set_title("Survival Count by Gender")
fig1.tight_layout()
st.pyplot(fig1)

# 2. Age Distribution
st.subheader("🎂 Age Distribution")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=30, ax=ax2)
ax2.set_title("Age Distribution of Passengers")
fig2.tight_layout()
st.pyplot(fig2)

# 3. Survival Rate by Class
st.subheader("🏷️ Survival Rate by Passenger Class")
survival_by_class = filtered_df.groupby("Pclass")["Survived"].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.barplot(data=survival_by_class, x="Pclass", y="Survived", ax=ax3, width=0.5)
ax3.set_ylabel("Survival Rate")
ax3.set_ylim(0, 1)
ax3.set_title("Survival Rate by Class")
fig3.tight_layout()
st.pyplot(fig3)

# 4. Average Fare by Class
st.subheader("💰 Average Fare by Class")
fare_by_class = filtered_df.groupby("Pclass")["Fare"].mean().reset_index()
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.barplot(data=fare_by_class, x="Pclass", y="Fare", ax=ax4, width=0.5)
ax4.set_title("Average Fare by Class")
fig4.tight_layout()
st.pyplot(fig4)

# 5. Passenger Count by Embarked Location
st.subheader("🧭 Passenger Count by Embarkation Point")
fig5, ax5 = plt.subplots(figsize=(6, 4))
sns.countplot(data=filtered_df, x="Embarked", ax=ax5, width=0.5)
ax5.set_title("Passengers by Embarkation Port")
fig5.tight_layout()
st.pyplot(fig5)

# 6. Boxplot of Age by Survival
st.subheader("📦 Age Distribution by Survival Status")
fig6, ax6 = plt.subplots(figsize=(6, 4))
sns.boxplot(data=filtered_df, x="Survival Status", y="Age", ax=ax6, width=0.5)
ax6.set_title("Boxplot of Age by Survival")
fig6.tight_layout()
st.pyplot(fig6)

# 7. Correlation Heatmap
st.subheader("📌 Feature Correlation Heatmap")
numeric_df = filtered_df.select_dtypes(include=["number"])
fig7, ax7 = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax7)
ax7.set_title("Correlation Heatmap")
fig7.tight_layout()
st.pyplot(fig7)
