import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Page config (SAME AS HEALTH)
# -----------------------
st.set_page_config(
    page_title="Car Insurance Analytics", page_icon="🚗", layout="centered"
)

st.title("🚗 Car Insurance Dataset Analytics")

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("Car_Dataset.csv")

# -----------------------
# EXACT SAME THEME AS HEALTH ANALYTICS
# -----------------------
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

FIG_SIZE = (6, 3.5)  # EXACT SAME SIZE

# -----------------------
# Dataset preview
# -----------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# -----------------------
# Chart 1: Insurance Premium Distribution
# -----------------------
st.subheader("💰 Insurance Premium Distribution")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.histplot(
    df["Insurance Premium"],
    bins=15,
    kde=True,
    color="#60a5fa",
    ax=ax,
)
ax.set_xlabel("Insurance Premium")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

# -----------------------
# Chart 2: Driver Age vs Insurance Premium
# -----------------------
st.subheader("👤 Driver Age vs Insurance Premium")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.scatterplot(
    x="Driver Age",
    y="Insurance Premium",
    data=df,
    color="#34d399",
    ax=ax,
)
ax.set_xlabel("Driver Age")
ax.set_ylabel("Insurance Premium")
st.pyplot(fig)
plt.close(fig)

# -----------------------
# Chart 3: Driver Experience vs Previous Accidents
# -----------------------
st.subheader("🛠 Driver Experience vs Previous Accidents")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.scatterplot(
    x="Driver Experience",
    y="Previous Accidents",
    data=df,
    color="#f472b6",
    ax=ax,
)
ax.set_xlabel("Driver Experience")
ax.set_ylabel("Previous Accidents")
st.pyplot(fig)
plt.close(fig)

# -----------------------
# Chart 4: Annual Mileage vs Insurance Premium
# -----------------------
st.subheader("🛣 Annual Mileage vs Insurance Premium")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.scatterplot(
    x="Annual Mileage (x1000 km)",
    y="Insurance Premium",
    data=df,
    color="#facc15",
    ax=ax,
)
ax.set_xlabel("Annual Mileage (x1000 km)")
ax.set_ylabel("Insurance Premium")
st.pyplot(fig)
plt.close(fig)

# -----------------------
# Chart 5: Previous Accidents vs Insurance Premium
# -----------------------
st.subheader("⚠️ Previous Accidents vs Insurance Premium")

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.boxplot(
    x="Previous Accidents",
    y="Insurance Premium",
    data=df,
    palette="cool",
    ax=ax,
)
ax.set_xlabel("Previous Accidents")
ax.set_ylabel("Insurance Premium")
st.pyplot(fig)
plt.close(fig)
