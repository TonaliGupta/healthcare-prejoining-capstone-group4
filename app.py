import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency
import os
import report_generator
from dotenv import load_dotenv

# Load environment variables from .env file (allow overrides on file updates)
load_dotenv(override=True)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Healthcare Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------

plt.rcParams.update({
    "axes.facecolor":   "#f2f2f2",
    "figure.facecolor": "white",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.9,
    "axes.axisbelow":   True,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        10,
})

PALETTE = {
    "No Diabetes":  "#4C9BE8",
    "Pre-Diabetic": "#F5A623",
    "Diabetic":     "#E84C4C",
}
COLOR_LIST = list(PALETTE.values())

def add_bar_labels(ax, fmt="{:,.0f}", padding=3, fontsize=8):
    for patch in ax.patches:
        h = patch.get_height()
        if h == 0 or np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            xy=(patch.get_x() + patch.get_width() / 2, h),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
        )

def add_hbar_labels(ax, fmt="{:.2f}", padding=2, fontsize=8):
    for patch in ax.patches:
        w = patch.get_width()
        if w == 0 or np.isnan(w):
            continue
        ax.annotate(
            fmt.format(w),
            xy=(w, patch.get_y() + patch.get_height() / 2),
            xytext=(padding, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=fontsize, fontweight="bold",
        )

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Diabetes_data.csv")
    df["Diabetes_Status"] = df["Diabetes_012"].map(
        {0: "No Diabetes", 1: "Pre-Diabetic", 2: "Diabetic"}
    )
    df["Age_Category"] = df["Age"].map({
        1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",
        6:"45-49",7:"50-54",8:"55-59",9:"60-64",10:"65-69",
        11:"70-74",12:"75-79",13:"80+"
    })
    df["BMI_Category"] = pd.cut(
        df["BMI"], bins=[0,18.5,25,30,100],
        labels=["Underweight","Normal","Overweight","Obese"]
    )
    df["GenHlth_Label"] = df["GenHlth"].map(
        {1:"Excellent",2:"Very Good",3:"Good",4:"Fair",5:"Poor"}
    )
    return df

df = load_data()

AGE_ORDER = ["18-24","25-29","30-34","35-39","40-44","45-49",
             "50-54","55-59","60-64","65-69","70-74","75-79","80+"]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏥 Healthcare Risk")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "Diabetes Distribution",
        "BMI Risk Analysis",
        "Age Risk Analysis",
        "Lifestyle Risk Analysis",
        "High Risk Population",
        "Correlation Analysis",
        "Key Findings",
        "AI Health Report",
    ]
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("Healthcare Risk Dashboard")
    st.markdown("**Diabetes Risk Analysis** using demographic, lifestyle, and health-related factors.")
    st.markdown("---")

    total        = len(df)
    pct_diab     = round((df["Diabetes_012"] == 2).mean() * 100, 1)
    pct_prediab  = round((df["Diabetes_012"] == 1).mean() * 100, 1)
    avg_bmi      = round(df["BMI"].mean(), 1)
  

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients",    f"{total:,}")
    c2.metric("% Diabetic",        f"{pct_diab}%")
    c3.metric("% Pre-Diabetic",    f"{pct_prediab}%")
    c4.metric("Avg BMI",           avg_bmi)


    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Variable Distribution")
        counts = df["Diabetes_Status"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(counts.index, counts.values,
                      color=[PALETTE[k] for k in counts.index])
        ax.set_ylabel("Count")
        add_bar_labels(ax)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Risk Factor Snapshot (% Prevalence)")
        risk_map = {
            "High BP": df["HighBP"].mean()*100,
            "High Cholesterol": df["HighChol"].mean()*100,
            "Smoker": df["Smoker"].mean()*100,
            "Inactive": (1-df["PhysActivity"]).mean()*100,
            "Diff Walking": df["DiffWalk"].mean()*100,
            "Heart Disease": df["HeartDiseaseorAttack"].mean()*100,
        }
        labels = list(risk_map.keys())
        values = list(risk_map.values())
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(labels, values, color="#4C9BE8")
        for bar, val in zip(bars, values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")
        ax.set_xlabel("% of Population")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

# --------------------------------------------------
# DIABETES DISTRIBUTION
# --------------------------------------------------

elif page == "Diabetes Distribution":

    st.title("Diabetes Distribution")
    st.markdown("---")

    col1, col2 = st.columns(2)
    counts = df["Diabetes_Status"].value_counts()

    with col1:
        st.subheader("Pie Chart")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        wedge_props = {"linewidth": 2, "edgecolor": "white"}
        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=[PALETTE[k] for k in counts.index],
            wedgeprops=wedge_props,
            startangle=140,
        )
        plt.tight_layout()
        st.pyplot(fig)


    st.markdown("---")
    st.subheader("General Health Rating by Diabetes Status")
    st.caption("1 = Excellent → 5 = Poor")

    genhlth_order = ["Excellent","Very Good","Good","Fair","Poor"]
    genhlth_pct = (
        df.groupby(["GenHlth_Label","Diabetes_Status"])
        .size()
        .unstack(fill_value=0)
        .reindex(genhlth_order)
    )
    genhlth_pct_norm = genhlth_pct.div(genhlth_pct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bottom = np.zeros(len(genhlth_pct_norm))
    for status in ["No Diabetes","Pre-Diabetic","Diabetic"]:
        if status in genhlth_pct_norm.columns:
            vals = genhlth_pct_norm[status].values
            bars = ax.bar(genhlth_pct_norm.index, vals,
                          bottom=bottom, label=status,
                          color=PALETTE[status], edgecolor="white")
            for bar, val, bot in zip(bars, vals, bottom):
                if val > 4:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bot + val/2, f"{val:.0f}%",
                            ha="center", va="center",
                            fontsize=7.5, fontweight="bold", color="white")
            bottom += vals

    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("General Health Rating")
    ax.legend(loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("As general health worsens, diabetes prevalence rises sharply.")

# --------------------------------------------------
# BMI ANALYSIS
# --------------------------------------------------

elif page == "BMI Risk Analysis":

    st.title("BMI Risk Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BMI Distribution by Diabetes Status")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=df, x="Diabetes_Status",
                    y="BMI", ax=ax,
                    order=["No Diabetes","Pre-Diabetic","Diabetic"],
                    palette=PALETTE)
        ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("% Diabetic Across BMI Categories (Stacked 100%)")
        bmi_slab = (
            df.groupby("BMI_Category", observed=True)["Diabetes_012"]
            .value_counts(normalize=True)
            .mul(100)
            .unstack()
            .rename(columns={0:"No Diabetes", 1:"Pre-Diabetic", 2:"Diabetic"})
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bottom = np.zeros(len(bmi_slab))
        for status in ["No Diabetes","Pre-Diabetic","Diabetic"]:
            if status in bmi_slab.columns:
                vals = bmi_slab[status].values
                bars = ax.bar(bmi_slab.index, vals, bottom=bottom,
                              label=status, color=PALETTE[status],
                              edgecolor="white")
                for bar, val, bot in zip(bars, vals, bottom):
                    if val > 3:
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bot + val/2, f"{val:.0f}%",
                                ha="center", va="center",
                                fontsize=8, fontweight="bold", color="white")
                bottom += vals
        ax.set_ylabel("Proportion (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("High BMI + Physical Activity → Diabetes Rate")
    high_bmi = df[df["BMI"] > 30]
    active_rate   = (high_bmi[high_bmi["PhysActivity"]==1]["Diabetes_012"]==2).mean()*100
    inactive_rate = (high_bmi[high_bmi["PhysActivity"]==0]["Diabetes_012"]==2).mean()*100

    c1, c2, c3 = st.columns(3)
    c1.metric("Active (BMI>30) → Diabetes %",   f"{active_rate:.1f}%")
    c2.metric("Inactive (BMI>30) → Diabetes %", f"{inactive_rate:.1f}%")
    c3.metric("Risk Multiplier", f"{inactive_rate/active_rate:.1f}x")

    fig, ax = plt.subplots(figsize=(5, 3))
    labels_  = ["Physically Active\n(BMI > 30)", "Physically Inactive\n(BMI > 30)"]
    vals_    = [active_rate, inactive_rate]
    colors_  = ["#4C9BE8", "#E84C4C"]
    bars = ax.bar(labels_, vals_, color=colors_, width=0.4)
    for bar, val in zip(bars, vals_):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Diabetes Rate (%)")
    ax.set_title("Exercise significantly reduces diabetes risk even with high BMI", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Average BMI by Diabetes Status")
    bmi_avg = df.groupby("Diabetes_Status")["BMI"].mean().round(2).reset_index()
    bmi_avg.columns = ["Diabetes Status", "Average BMI"]
    st.dataframe(bmi_avg, use_container_width=True, hide_index=True)

# --------------------------------------------------
# AGE ANALYSIS
# --------------------------------------------------

elif page == "Age Risk Analysis":

    st.title("Age Risk Analysis")
    st.markdown("---")

    st.subheader("Diabetes Rate (%) by Age Group")
    age_rate = (
        df.groupby("Age_Category")["Diabetes_012"]
        .apply(lambda x: (x == 2).mean() * 100)
        .reindex(AGE_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(AGE_ORDER, age_rate.values, marker="o", color="#E84C4C",
            linewidth=2, markersize=7)
    ax.fill_between(AGE_ORDER, age_rate.values, alpha=0.15, color="#E84C4C")
    for i, (x, y) in enumerate(zip(AGE_ORDER, age_rate.values)):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("% Diabetic in Age Group")
    ax.set_xlabel("Age Group")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Diabetes prevalence rises steeply from age 35 and peaks in the 60–69 group.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Absolute Diabetic Count by Age Group")
        diabetic_counts = (
            df[df["Diabetes_012"] == 2]
            .groupby("Age_Category").size()
            .reindex(AGE_ORDER)
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(AGE_ORDER, diabetic_counts.values, color="#E84C4C", edgecolor="white")
        add_bar_labels(ax, fontsize=7)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Age Distribution — Stacked by Diabetes Status (100%)")
        age_stacked = (
            df.groupby("Age_Category")["Diabetes_012"]
            .value_counts(normalize=True)
            .mul(100)
            .unstack()
            .rename(columns={0:"No Diabetes",1:"Pre-Diabetic",2:"Diabetic"})
            .reindex(AGE_ORDER)
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        bottom = np.zeros(len(age_stacked))
        for status in ["No Diabetes","Pre-Diabetic","Diabetic"]:
            if status in age_stacked.columns:
                vals = age_stacked[status].values
                ax.bar(AGE_ORDER, vals, bottom=bottom,
                       label=status, color=PALETTE[status], edgecolor="white")
                bottom += vals
        ax.set_ylabel("Proportion (%)")
        ax.set_ylim(0,100)
        ax.legend(fontsize=7, loc="upper left")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    pct_55plus = len(df[(df["Diabetes_012"]==2) & (df["Age"]>=8)]) / len(df[df["Diabetes_012"]==2]) * 100
    st.info(f"📌 **{pct_55plus:.1f}%** of all diabetic patients are aged **55 or above**.")

# --------------------------------------------------
# LIFESTYLE ANALYSIS
# --------------------------------------------------

elif page == "Lifestyle Risk Analysis":

    st.title("Lifestyle Risk Analysis")
    st.markdown("---")

    st.subheader("Lifestyle Factor Prevalence: Non-Diabetic vs Diabetic")
    st.caption("Shows the proportion (%) of each group that has the risk factor")

    risk_cols   = ["HighBP","HighChol","Smoker","PhysActivity","HvyAlcoholConsump","Fruits","Veggies"]
    risk_labels = ["High BP","High Chol","Smoker","Phys. Active","Heavy Alcohol","Eats Fruits","Eats Veggies"]

    no_diab  = df[df["Diabetes_012"]==0][risk_cols].mean() * 100
    diabetic = df[df["Diabetes_012"]==2][risk_cols].mean() * 100

    x = np.arange(len(risk_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - width/2, no_diab, width, label="No Diabetes", color="#4C9BE8")
    b2 = ax.bar(x + width/2, diabetic,  width, label="Diabetic",    color="#E84C4C")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f"{bar.get_height():.0f}%",
                ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(risk_labels)
    ax.set_ylabel("% With Factor")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Drill-Down: Single Factor vs Diabetes Status (Proportional)")
    st.caption("100% stacked bars — shows the *diabetes composition* within Yes/No groups")

    feature_names = {
        "PhysActivity":     "Physical Activity",
        "Smoker":           "Smoking History",
        "Fruits":           "Fruit Consumption",
        "Veggies":          "Vegetable Consumption",
        "HvyAlcoholConsump":"Heavy Alcohol Consumption",
        "HighBP":           "High Blood Pressure",
        "HighChol":         "High Cholesterol",
    }

    selected = st.selectbox(
        "Select factor",
        list(feature_names.keys()),
        format_func=lambda x: feature_names[x]
    )
    st.caption("0 = No / Never,  1 = Yes / Present")

    crosstab = pd.crosstab(df[selected], df["Diabetes_Status"])
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**100% Stacked (Proportions)**")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        bottom = np.zeros(len(crosstab_pct))
        for status in ["No Diabetes","Pre-Diabetic","Diabetic"]:
            if status in crosstab_pct.columns:
                vals = crosstab_pct[status].values
                bars = ax.bar(crosstab_pct.index.astype(str), vals,
                              bottom=bottom, label=status,
                              color=PALETTE[status], edgecolor="white")
                for bar, val, bot in zip(bars, vals, bottom):
                    if val > 4:
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bot + val/2, f"{val:.1f}%",
                                ha="center", va="center",
                                fontsize=8, fontweight="bold", color="white")
                bottom += vals
        ax.set_ylabel("Proportion (%)")
        ax.set_ylim(0,100)
        ax.set_xticklabels(["No / Absent","Yes / Present"])
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Raw Counts**")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        crosstab.plot(kind="bar", ax=ax,
                      color=[PALETTE[c] for c in crosstab.columns
                             if c in PALETTE],
                      edgecolor="white")
        add_bar_labels(ax, fontsize=7)
        ax.set_xticklabels(["No / Absent","Yes / Present"], rotation=0)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    st.dataframe(crosstab, use_container_width=True)

# --------------------------------------------------
# HIGH RISK ANALYSIS
# --------------------------------------------------

elif page == "High Risk Population":

    st.title("High Risk Population Analysis")
    st.markdown("---")

    overall_rate = (df["Diabetes_012"] == 2).mean() * 100
    triple = df[(df["HighBP"]==1) & (df["HighChol"]==1) & (df["PhysActivity"]==0)]
    triple_rate = (triple["Diabetes_012"] == 2).mean() * 100
    multiplier  = triple_rate / overall_rate

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High Risk Individuals",           f"{len(triple):,}")
    c2.metric("Diabetic in High Risk",           f"{len(triple[triple['Diabetes_012']==2]):,}")
    c3.metric("Overall Diabetes Rate",           f"{overall_rate:.1f}%")
    c4.metric("High-Risk Diabetes Rate",         f"{triple_rate:.1f}%", delta=f"{multiplier:.1f}x higher")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall vs Triple-Risk Diabetes Rate")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        labels_ = ["Overall\nPopulation", "High BP +\nHigh Chol +\nNo Exercise"]
        vals_   = [overall_rate, triple_rate]
        colors_ = ["#4C9BE8", "#E84C4C"]
        bars = ax.bar(labels_, vals_, color=colors_, width=0.4)
        for bar, val in zip(bars, vals_):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}%", ha="center",
                    fontsize=12, fontweight="bold")
        ax.set_ylabel("Diabetes Prevalence (%)")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Diabetes Status Distribution — High Risk Group")
        dist = triple["Diabetes_Status"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(dist.index, dist.values,
               color=[PALETTE.get(k,"grey") for k in dist.index],
               edgecolor="white")
        add_bar_labels(ax)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Build Your Own Risk Profile")
    st.caption("Select combinations to see the resulting diabetes rate")

    f1 = st.checkbox("High Blood Pressure", value=True)
    f2 = st.checkbox("High Cholesterol",    value=True)
    f3 = st.checkbox("No Physical Activity",value=True)
    f4 = st.checkbox("Smoker")
    f5 = st.checkbox("Difficulty Walking")

    mask = pd.Series([True]*len(df))
    if f1: mask &= (df["HighBP"]==1)
    if f2: mask &= (df["HighChol"]==1)
    if f3: mask &= (df["PhysActivity"]==0)
    if f4: mask &= (df["Smoker"]==1)
    if f5: mask &= (df["DiffWalk"]==1)

    subset = df[mask]
    if len(subset) > 0:
        custom_rate = (subset["Diabetes_012"]==2).mean()*100
        st.metric(
            f"Diabetes rate for selected profile ({len(subset):,} people)",
            f"{custom_rate:.1f}%",
            delta=f"{custom_rate - overall_rate:+.1f}% vs overall"
        )
    else:
        st.warning("No patients match this combination.")

    st.info(
        """
        **High Risk Group Definition (default)**
        - High Blood Pressure = Yes
        - High Cholesterol = Yes
        - No Physical Activity
        """
    )

# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------

elif page == "Correlation Analysis":

    st.title("Correlation & Statistical Analysis")
    st.markdown("---")

    numeric_df = df.drop(columns=["BMI_Category","Age_Category","Diabetes_Status","GenHlth_Label"], errors="ignore")
    corr = numeric_df.corr()

    st.subheader("Feature Correlation with Diabetes (Sorted)")
    corr_target = corr["Diabetes_012"].drop("Diabetes_012").sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_ = ["#E84C4C" if v > 0 else "#4C9BE8" for v in corr_target.values]
    bars = ax.barh(corr_target.index, corr_target.values, color=colors_, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, corr_target.values):
        ax.text(val + (0.004 if val>=0 else -0.004),
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center",
                ha="left" if val>=0 else "right",
                fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    red_patch  = mpatches.Patch(color="#E84C4C", label="Positive (risk factor)")
    blue_patch = mpatches.Patch(color="#4C9BE8", label="Negative (protective)")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "**Top risk factors:** General Health, High BP, BMI, DiffWalk, High Chol  |  "
        "**Protective:** Income, Education, Physical Activity"
    )

    st.markdown("---")
    st.subheader("Full Correlation Heatmap")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="BrBG", linewidths=0.4,
                annot_kws={"size": 6}, ax=ax)
    ax.set_title("Correlation Heatmap — All Features", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)



# --------------------------------------------------
# FINDINGS
# --------------------------------------------------

elif page == "Key Findings":

    st.title("Key Findings & Recommendations")
    st.markdown("---")

    st.subheader("Key Statistical Findings")
    st.success(
        """
        1. **BMI** — Diabetics average BMI ~32 vs ~28 for non-diabetics. Obese individuals have **24.9% diabetes rate** vs 6.2% for normal BMI.

        2. **Age** — Diabetes prevalence rises steeply from age 55. Over **80% of diabetic patients are aged 55+**.

        3. **Triple Risk** — Patients with High BP + High Cholesterol + No Exercise have a **36.5% diabetes rate**, vs 13.9% overall (2.6× higher).

        4. **Exercise** — Among high-BMI individuals (BMI > 30), physically active patients have a **21.7% diabetes rate** vs 31.2% for inactive (1.4× risk).

        5. **General Health** — Patients rating health as "Poor" are overwhelmingly diabetic or pre-diabetic.

        6. **Income & Education** — Negatively correlated with diabetes. Socioeconomic factors are significant protective factors.

        7. **High BP** — Present in 75% of diabetics vs 37% of non-diabetics. Chi-square confirms very strong association.
        """
    )

    st.markdown("---")
    st.subheader("🏥 Recommendations")
    st.info(
        """
        **Clinical & Public Health:**
        - Screen patients aged 45+ for blood glucose annually
        - Prioritise weight management programmes targeting BMI ≥ 30
        - Integrate physical activity counselling into all primary care visits
        - Treat high BP and high cholesterol as combined diabetes risk markers

        **Prevention Targeting:**
        - Focus outreach on the 55–69 age band (highest absolute case count)
        - Address socioeconomic barriers — low income/education populations need tailored support
        - Patients with poor self-rated general health warrant immediate metabolic screening

        """
    )

    st.markdown("---")
    st.subheader("⚗️ Methodology Notes")
    st.warning(
        """
        - Dataset: 253,680 survey respondents (BRFSS)
        - Duplicates retained — no unique patient ID; duplicates may be distinct individuals
        - Outliers in BMI, MentHlth, PhysHlth retained as clinically plausible values
        - All hypothesis tests significant at p < 0.05
        - **Correlation ≠ Causation** — all associations should be interpreted as statistical relationships, not proof of direct causal mechanisms
        """
    )

# --------------------------------------------------
# AI HEALTH REPORT
# --------------------------------------------------

elif page == "AI Health Report":

    st.title("🤖 AI Personalized Health Report")
    st.markdown("Generate a clinical-grade, data-driven health evaluation and recommendation plan based on your risk factors and comparison with the CDC BRFSS cohort (253,680 records).")
    st.markdown("---")

    # API Key Configuration
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_api_key == "your_openai_api_key_here":
        openai_api_key = ""
        
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_api_key == "your_gemini_api_key_here":
        gemini_api_key = ""
        
    if not gemini_api_key and "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]

    # If neither key is found in environment/secrets, prompt in the sidebar
    if not openai_api_key and not gemini_api_key:
        st.sidebar.subheader("🔑 API Key Configuration")
        provider = st.sidebar.selectbox(
            "Select API Provider",
            ["Google Gemini (Recommended - Free Tier)", "OpenAI"]
        )
        if provider == "Google Gemini (Recommended - Free Tier)":
            api_key_input = st.sidebar.text_input(
                "Enter Gemini API Key:",
                type="password",
                help="Get a free API Key from Google AI Studio (aistudio.google.com)."
            )
            if api_key_input:
                gemini_api_key = api_key_input
        else:
            api_key_input = st.sidebar.text_input(
                "Enter OpenAI API Key:",
                type="password"
            )
            if api_key_input:
                openai_api_key = api_key_input

    # Form mapping helpers
    age_label_to_val = {
        "18-24": 1.0, "25-29": 2.0, "30-34": 3.0, "35-39": 4.0, "40-44": 5.0,
        "45-49": 6.0, "50-54": 7.0, "55-59": 8.0, "60-64": 9.0, "65-69": 10.0,
        "70-74": 11.0, "75-79": 12.0, "80+": 13.0
    }
    
    edu_label_to_val = {
        "Never attended school or only kindergarten": 1.0,
        "Grades 1 through 8 (Elementary)": 2.0,
        "Grades 9 through 11 (Some High School)": 3.0,
        "Grade 12 or GED (High School Graduate)": 4.0,
        "College 1 to 3 years (Some College)": 5.0,
        "College 4 years or more (College Graduate)": 6.0
    }
    
    inc_label_to_val = {
        "Less than $10,000": 1.0,
        "$10,000 to $14,999": 2.0,
        "$15,000 to $19,999": 3.0,
        "$20,000 to $24,999": 4.0,
        "$25,000 to $34,999": 5.0,
        "$35,000 to $54,999": 6.0,
        "$55,000 to $74,999": 7.0,
        "$75,000 or more": 8.0
    }
    
    genhlth_label_to_val = {
        "Excellent": 1.0,
        "Very Good": 2.0,
        "Good": 3.0,
        "Fair": 4.0,
        "Poor": 5.0
    }

    # UI Layout split into two sections: input form and report output
    with st.form("health_profile_form"):
        st.subheader("📋 Enter Your Health Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics & Status**")
            sex_choice = st.selectbox("Sex", ["Female", "Male"])
            age_choice = st.selectbox("Age Group", list(age_label_to_val.keys()))
            edu_choice = st.selectbox("Education Level", list(edu_label_to_val.keys()))
            inc_choice = st.selectbox("Annual Income Range", list(inc_label_to_val.keys()))
            
        with col2:
            st.markdown("**Clinical Vitals**")
            bmi_val = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
            high_bp_choice = st.selectbox("High Blood Pressure?", ["No", "Yes"])
            high_chol_choice = st.selectbox("High Cholesterol?", ["No", "Yes"])
            chol_check_choice = st.selectbox("Cholesterol Checked (Last 5 Yrs)?", ["Yes", "No"])
            diff_walk_choice = st.selectbox("Difficulty Walking or Climbing Stairs?", ["No", "Yes"])

        with col3:
            st.markdown("**Lifestyle & History**")
            smoker_choice = st.selectbox("Smoked >= 100 cigarettes in life?", ["No", "Yes"])
            stroke_choice = st.selectbox("Ever had a Stroke?", ["No", "Yes"])
            heart_choice = st.selectbox("Heart Disease / Attack History?", ["No", "Yes"])
            phys_act_choice = st.selectbox("Physical Activity (Last 30 Days)?", ["Yes", "No"])
            fruits_choice = st.selectbox("Consume Fruits Daily?", ["Yes", "No"])
            veggies_choice = st.selectbox("Consume Vegetables Daily?", ["Yes", "No"])
            alcohol_choice = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])

        st.markdown("---")
        col_hlth1, col_hlth2, col_hlth3 = st.columns(3)
        with col_hlth1:
            gen_hlth_choice = st.selectbox("General Health Self-Rating", list(genhlth_label_to_val.keys()), index=2)
        with col_hlth2:
            phys_hlth_days = st.slider("Days of Poor Physical Health (Last 30)", 0.0, 30.0, 0.0, step=1.0)
        with col_hlth3:
            ment_hlth_days = st.slider("Days of Poor Mental Health (Last 30)", 0.0, 30.0, 0.0, step=1.0)
            
        submit_button = st.form_submit_button("📊 Process & Generate AI Health Report")

    # Form Submission Logic
    if submit_button:
        # Determine BMI category
        if bmi_val <= 18.5:
            bmi_cat = "Underweight"
        elif bmi_val <= 25.0:
            bmi_cat = "Normal"
        elif bmi_val <= 30.0:
            bmi_cat = "Overweight"
        else:
            bmi_cat = "Obese"

        user_profile = {
            "Age_Category": age_choice,
            "Age": age_label_to_val[age_choice],
            "Sex": 1.0 if sex_choice == "Male" else 0.0,
            "Education": edu_label_to_val[edu_choice],
            "Income": inc_label_to_val[inc_choice],
            "BMI": bmi_val,
            "BMI_Category": bmi_cat,
            "HighBP": 1.0 if high_bp_choice == "Yes" else 0.0,
            "HighChol": 1.0 if high_chol_choice == "Yes" else 0.0,
            "CholCheck": 1.0 if chol_check_choice == "Yes" else 0.0,
            "DiffWalk": 1.0 if diff_walk_choice == "Yes" else 0.0,
            "Smoker": 1.0 if smoker_choice == "Yes" else 0.0,
            "Stroke": 1.0 if stroke_choice == "Yes" else 0.0,
            "HeartDiseaseorAttack": 1.0 if heart_choice == "Yes" else 0.0,
            "PhysActivity": 1.0 if phys_act_choice == "Yes" else 0.0,
            "Fruits": 1.0 if fruits_choice == "Yes" else 0.0,
            "Veggies": 1.0 if veggies_choice == "Yes" else 0.0,
            "HvyAlcoholConsump": 1.0 if alcohol_choice == "Yes" else 0.0,
            "GenHlth": genhlth_label_to_val[gen_hlth_choice],
            "PhysHlth": phys_hlth_days,
            "MentHlth": ment_hlth_days,
        }
        
        st.session_state["user_profile"] = user_profile
        
        # Calculate stats
        with st.spinner("Analyzing dataset to locate matching cohort..."):
            cohort_stats = report_generator.get_cohort_stats(df, user_profile)
            st.session_state["cohort_stats"] = cohort_stats
            
        # Trigger report generation if an API Key is available
        if gemini_api_key or openai_api_key:
            model_name = "Gemini 2.5 Flash" if gemini_api_key else "gpt-4o-mini"
            with st.spinner(f"Generating clinical AI report with {model_name}..."):
                try:
                    report_text = report_generator.generate_report(
                        user_profile,
                        cohort_stats,
                        openai_api_key=openai_api_key,
                        gemini_api_key=gemini_api_key
                    )
                    st.session_state["generated_report"] = report_text
                    st.success("Report successfully generated!")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    st.session_state["generated_report"] = None
        else:
            st.session_state["generated_report"] = None
            st.warning("⚠️ **API Key Missing**: Calculated cohort stats, but could not generate the AI report. Please configure a Google Gemini or OpenAI API Key and try again.")

    # Render results if available in session state
    if "cohort_stats" in st.session_state and st.session_state["cohort_stats"] is not None:
        stats = st.session_state["cohort_stats"]
        profile = st.session_state["user_profile"]
        
        st.markdown("---")
        st.subheader("📊 CDC Dataset Cohort Insights")
        st.markdown(f"We located a cohort of **{stats['cohort_size']:,}** individuals in the CDC dataset matching your: **{stats['matched_criteria']}**.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matched Cohort Size", f"{stats['cohort_size']:,}")
        c2.metric("Cohort Diabetes Rate", f"{stats['pct_diabetic']}%", delta=f"{stats['relative_risk']}x vs avg")
        c3.metric("Cohort Pre-Diabetes Rate", f"{stats['pct_pre_diabetic']}%")
        c4.metric("Avg Cohort Gen Health", f"{stats['cohort_avg_genhlth']}/5")

        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("Diabetes Prevalence Comparison")
            # Cohort distribution bar chart
            fig, ax = plt.subplots(figsize=(5, 3.5))
            labels = ["No Diabetes", "Pre-Diabetic", "Diabetic"]
            cohort_vals = [stats["pct_no_diabetes"], stats["pct_pre_diabetic"], stats["pct_diabetic"]]
            
            # Overall rates from dataset
            overall_counts = df["Diabetes_Status"].value_counts(normalize=True) * 100
            overall_vals = [
                overall_counts.get("No Diabetes", 84.2), 
                overall_counts.get("Pre-Diabetic", 1.8), 
                overall_counts.get("Diabetic", 13.9)
            ]
            
            x = np.arange(len(labels))
            width = 0.35
            
            rects1 = ax.bar(x - width/2, cohort_vals, width, label='Matched Cohort', color='#E84C4C')
            rects2 = ax.bar(x + width/2, overall_vals, width, label='General Population', color='#4C9BE8')
            
            ax.set_ylabel('Percentage (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(fontsize=8)
            add_bar_labels(ax, fmt="{:.1f}%", fontsize=7.5)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Comparison of diabetes rates in your matched demographic/health cohort vs the full CDC dataset.")

        with col_c2:
            st.subheader("Prevalence of Other Risks in Your Cohort")
            risk_map = {
                "Physically Active": stats["cohort_phys_act_pct"],
                "Smoker": stats["cohort_smoker_pct"],
                "Difficulty Walking": stats["cohort_diff_walk_pct"],
                "Heart Disease": stats["cohort_heart_disease_pct"]
            }
            labels_r = list(risk_map.keys())
            values_r = list(risk_map.values())
            fig_r, ax_r = plt.subplots(figsize=(5, 3.5))
            bars_r = ax_r.barh(labels_r, values_r, color="#F5A623")
            for bar, val in zip(bars_r, values_r):
                ax_r.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")
            ax_r.set_xlabel("% of Matched Cohort")
            ax_r.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_r)
            st.caption("Percentage of people within your matching cohort that possess these health characteristics.")

        # Show report if it is in session state
        if "generated_report" in st.session_state and st.session_state["generated_report"] is not None:
            st.markdown("---")
            st.subheader("🤖 AI Health Risk Report")
            
            # Text download option
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=st.session_state["generated_report"],
                file_name="Personalized_Health_Report.md",
                mime="text/markdown"
            )
            
            st.markdown(st.session_state["generated_report"])
