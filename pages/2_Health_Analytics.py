import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Insurance Analytics", page_icon="📊", layout="centered")

st.title("📊 Insurance Dataset Analytics")
# Load dataset
df = pd.read_csv("insurance.csv")
df["bmi"] = df["weight"] / (df["height"] ** 2)

sns.set_theme(
    style="darkgrid",
    rc={
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "grid.color": "#2a2e35",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#9ca3af",
        "ytick.color": "#9ca3af",
        "text.color": "#e5e7eb",
    },
)

FIG_SIZE = (6, 3.5)

# Dataset preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

st.divider()

order = ["Low", "Medium", "High"]
# Chart 1: Age Distribution
st.subheader("📈 Age Distribution")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.histplot(df["age"], bins=15, kde=True, color="#60a5fa", ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

# Chart 2: BMI vs Premium Category
st.subheader("⚖️ BMI vs Insurance Premium Category")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.boxplot(
    x="insurance_premium_category", y="bmi", order=order, data=df, palette="cool", ax=ax
)
ax.set_xlabel("Premium Category")
ax.set_ylabel("BMI")
st.pyplot(fig)
plt.close(fig)

# Chart 3: Smoker vs Premium Category
st.subheader("🚬 Smoker vs Premium Category")
fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.countplot(
    x="insurance_premium_category",
    hue="smoker",
    order=order,
    data=df,
    palette=["#22c55e", "#ef4444"],
    ax=ax,
)
ax.set_xlabel("Premium Category")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

# Chart 4: Income vs Premium Category
st.subheader("💰 Income vs Premium Category")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.boxplot(
    x="insurance_premium_category",
    y="income_lpa",
    order=order,
    data=df,
    palette="viridis",
    ax=ax,
)
ax.set_xlabel("Premium Category")
ax.set_ylabel("Income (LPA)")
st.pyplot(fig)
plt.close(fig)
