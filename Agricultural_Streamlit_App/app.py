import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="AI-Powered Sustainable Crop Production & Decision Support",
    layout="wide",
    page_icon="🌾"
)

# ================================================================
# CUSTOM CSS (preserved from original + minor enhancements)
# ================================================================
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(
        to right,
        #e8f5e9,
        #f1f8e9,
        #ffffff
    );
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #dcedc8;
    border-right: 2px solid #aed581;
}

/* Main Title */
h1 {
    color: #1b5e20;
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

/* Subheaders */
h2, h3 {
    color: #2e7d32;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 2px 4px 15px rgba(0,0,0,0.1);
    border-left: 8px solid #43a047;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Buttons */
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.stButton>button:hover {
    background-color: #1b5e20;
    color: white;
}

/* Download Button */
.stDownloadButton>button {
    background-color: #388e3c;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}

/* Info boxes */
.info-box {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 2px 4px 12px rgba(0,0,0,0.08);
    border-left: 6px solid #43a047;
}

/* SDG badge */
.sdg-badge {
    display: inline-block;
    background: #e8f5e9;
    border: 2px solid #43a047;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 4px;
    font-weight: bold;
    color: #1b5e20;
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD DATA (preserved from original)
# ================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("crop_production.csv")

    # Rename columns (preserved from original)
    df.rename(columns={
        "State": "State_Name",
        "District": "District_Name",
        "Area-2024-25": "Area",
        "Production-2024-25": "Production",
        "Yield-2024-25": "Yield"
    }, inplace=True)

    return df

df = load_data()

# ================================================================
# CLEAN DATA (preserved from original)
# ================================================================
df["State_Name"] = df["State_Name"].astype(str).str.strip()
df["District_Name"] = df["District_Name"].astype(str).str.strip()
df["Crop"] = df["Crop"].astype(str).str.strip()
df["Season"] = df["Season"].astype(str).str.strip()

# Convert numeric columns
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

# Fill missing values
df.fillna(0, inplace=True)

# ================================================================
# SIDEBAR — Navigation + Filters
# ================================================================
st.sidebar.title("🌾 Navigation")

# Section navigator
page = st.sidebar.radio(
    "Go to Section",
    [
        "🏠 Project Overview",
        "📊 Agricultural Analytics",
        "🤖 AI Prediction",
        "🌱 Sustainable Decision Support",
        "📈 Model Performance",
        "🧠 AI Insights",
        "⚖️ Responsible AI",
        "📥 Download Results"
    ]
)

st.sidebar.markdown("---")
st.sidebar.title("🔎 Filter Data")

# State Filter (preserved)
state_list = ["All"] + sorted(df["State_Name"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", state_list)

# District Filter (NEW — dynamic based on selected state)
if selected_state != "All":
    district_options = sorted(
        df[df["State_Name"] == selected_state]["District_Name"].unique().tolist()
    )
else:
    district_options = sorted(df["District_Name"].unique().tolist())

district_list = ["All"] + district_options
selected_district = st.sidebar.selectbox("Select District", district_list)

# Season Filter (preserved)
season_list = ["All"] + sorted(df["Season"].unique().tolist())
selected_season = st.sidebar.selectbox("Select Season", season_list)

# Crop Filter (preserved)
crop_list = ["All"] + sorted(df["Crop"].unique().tolist())
selected_crop = st.sidebar.selectbox("Select Crop", crop_list)

# ================================================================
# FILTER DATA (preserved + district filter added)
# ================================================================
filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[filtered_df["State_Name"] == selected_state]

if selected_district != "All":
    filtered_df = filtered_df[filtered_df["District_Name"] == selected_district]

if selected_season != "All":
    filtered_df = filtered_df[filtered_df["Season"] == selected_season]

if selected_crop != "All":
    filtered_df = filtered_df[filtered_df["Crop"] == selected_crop]

# ================================================================
# MAIN TITLE
# ================================================================
st.title("🌾 AI-Powered Sustainable Crop Production & Decision Support")

# ================================================================
# HELPER: Train the ML model (cached so it doesn't retrain on every click)
# ================================================================
@st.cache_resource
def train_model(data_hash):
    """
    Train a Random Forest model using Area, Season, and Crop as features
    to predict Production. Returns the model, encoders, metrics, and
    test-set predictions.
    """
    # Prepare data — only rows with positive Area and Production
    ml_data = df[(df["Area"] > 0) & (df["Production"] > 0)].copy()

    if len(ml_data) < 20:
        return None  # Not enough data

    # Encode categorical features
    season_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()

    ml_data["Season_Encoded"] = season_encoder.fit_transform(ml_data["Season"])
    ml_data["Crop_Encoded"] = crop_encoder.fit_transform(ml_data["Crop"])

    # Features and target
    feature_cols = ["Area", "Season_Encoded", "Crop_Encoded"]
    X = ml_data[feature_cols]
    y = ml_data["Production"]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Feature importances
    importances = dict(zip(
        ["Area", "Season", "Crop"],
        model.feature_importances_
    ))

    return {
        "model": model,
        "season_encoder": season_encoder,
        "crop_encoder": crop_encoder,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
        "importances": importances,
        "feature_cols": feature_cols
    }

# Create a hash of the data for caching
data_hash = hash(df.shape)
model_result = train_model(data_hash)


# ================================================================
# SECTION 1: 🏠 PROJECT OVERVIEW
# ================================================================
if page == "🏠 Project Overview":

    st.header("🏠 Project Overview")

    st.markdown("""
    <div class="info-box">
    <h3>🎯 AI-Powered Sustainable Crop Production & Agricultural Decision Support System</h3>
    <p>This project was developed for the <strong>1M1B AI for Sustainability Virtual Internship</strong>
    in collaboration with <strong>IBM SkillsBuild & AICTE</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Problem Statement
    st.subheader("📋 Problem Statement")
    st.markdown("""
    Farmers and agricultural decision-makers often need to make crop-production decisions
    using historical agricultural data. Understanding production patterns, crop productivity,
    seasonal trends, and expected production can help support better and more sustainable
    agricultural planning.

    This application demonstrates how **AI/ML can convert agricultural data into useful
    decision-support insights**, aligning with the United Nations Sustainable Development Goals.
    """)

    # SDG Alignment
    st.subheader("🌍 SDG Alignment")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="sdg-badge">🎯 SDG 2 — Zero Hunger</div>
        <p><strong>Primary Goal:</strong> Support sustainable food production
        through data-driven agricultural insights.</p>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="sdg-badge">♻️ SDG 12 — Responsible Production</div>
        <p><strong>Secondary:</strong> Promote efficient resource use in agriculture
        through informed crop selection.</p>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="sdg-badge">🌡️ SDG 13 — Climate Action</div>
        <p><strong>Secondary:</strong> Support climate-aware agricultural planning
        through seasonal and production analysis.</p>
        """, unsafe_allow_html=True)

    # Target Users
    st.subheader("👥 Target Users")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.info("🧑‍🌾 Farmers")
    t2.info("👨‍💼 Agricultural Officers")
    t3.info("🔬 Agricultural Researchers")
    t4.info("🎓 Students & Researchers")
    t5.info("📋 Policy & Planning Stakeholders")

    # Why AI
    st.subheader("🤖 Why AI/ML?")
    st.markdown("""
    - **Pattern Recognition:** AI can identify production patterns across thousands of data records
      that would be impractical to analyze manually.
    - **Prediction:** Machine Learning models can estimate expected crop production based on
      historical data, area, season, and crop type.
    - **Decision Support:** AI-generated insights help stakeholders make more informed
      agricultural decisions.
    - **Scalability:** The same model can analyze data across all states, districts, seasons, and crops.
    """)

    # Expected Sustainability Impact
    st.subheader("🌱 Expected Sustainability Impact")
    st.markdown("""
    - Help farmers identify historically productive crop–season combinations
    - Support agricultural officers in data-driven resource allocation
    - Promote responsible crop production planning based on evidence
    - Raise awareness of data limitations and the need for holistic agricultural planning
    """)

    # How AI Works in This Project
    st.subheader("🧠 How AI Works in This Project")
    st.markdown("""
    ```
    📂 Input Agricultural Data (State, District, Crop, Season, Area, Production, Yield)
            ↓
    🧹 Data Preprocessing (cleaning, handling missing values, encoding categories)
            ↓
    🔍 Feature Selection (Area, Season, Crop used as inputs)
            ↓
    🤖 Machine Learning Model (Random Forest Regressor)
            ↓
    📊 Prediction (estimated crop production)
            ↓
    💡 Insights & Decision Support (patterns, recommendations, metrics)
    ```

    **In simple terms:** The model learns from thousands of historical crop production records.
    When you provide an area, season, and crop, it estimates the expected production based on
    similar historical patterns. This is **AI/ML-powered decision support**, not a guaranteed prediction.
    """)


# ================================================================
# SECTION 2: 📊 AGRICULTURAL ANALYTICS (all original charts preserved)
# ================================================================
elif page == "📊 Agricultural Analytics":

    st.header("📊 Agricultural Analytics Dashboard")

    # ---------- KPI SECTION (preserved) ----------
    total_production = filtered_df["Production"].sum()
    total_area = filtered_df["Area"].sum()
    total_crops = filtered_df["Crop"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("🌾 Total Production", f"{total_production:,.2f}")
    col2.metric("🟩 Total Area", f"{total_area:,.2f}")
    col3.metric("🌱 Number of Crops", total_crops)

    # ---------- DATASET PREVIEW (preserved) ----------
    st.subheader("📄 Dataset Preview")

    preview_columns = [
        "State_Name", "District_Name", "Crop",
        "Season", "Area", "Production", "Yield"
    ]
    display_df = filtered_df[preview_columns].drop_duplicates()
    st.dataframe(display_df, use_container_width=True, height=400)

    # ---------- TOP CROPS (preserved) ----------
    st.subheader("🌱 Top Producing Crops")

    crop_prod = (
        filtered_df
        .groupby("Crop")["Production"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig_crop = px.bar(
        x=crop_prod.index,
        y=crop_prod.values,
        labels={"x": "Crop", "y": "Production"},
        title="Top 10 Producing Crops"
    )
    st.plotly_chart(fig_crop, use_container_width=True)

    # ---------- SEASONAL ANALYSIS (preserved) ----------
    st.subheader("🍂 Seasonal Analysis")

    season_prod = (
        filtered_df
        .groupby("Season")["Production"]
        .sum()
        .reset_index()
    )

    fig_season = px.pie(
        season_prod,
        names="Season",
        values="Production",
        title="Season-wise Production"
    )
    st.plotly_chart(fig_season, use_container_width=True)

    # ---------- TOP STATES (preserved) ----------
    st.subheader("🏆 Top Producing States")

    state_prod = (
        filtered_df
        .groupby("State_Name")["Production"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig_state = px.bar(
        x=state_prod.index,
        y=state_prod.values,
        color=state_prod.values,
        labels={"x": "State", "y": "Production"},
        title="Top Producing States"
    )
    st.plotly_chart(fig_state, use_container_width=True)

    # ---------- DISTRICT ANALYSIS (preserved) ----------
    st.subheader("🏘️ Top Districts")

    district_prod = (
        filtered_df
        .groupby("District_Name", as_index=False)["Production"]
        .sum()
        .sort_values(by="Production", ascending=False)
        .head(10)
    )

    fig_district = px.bar(
        district_prod,
        x="District_Name",
        y="Production",
        title="Top Districts"
    )
    st.plotly_chart(fig_district, use_container_width=True)

    # ---------- YIELD VS PRODUCTION (preserved) ----------
    st.subheader("📈 Yield vs Production")

    fig_scatter = px.scatter(
        filtered_df,
        x="Yield",
        y="Production",
        color="Season",
        hover_data=["Crop"],
        title="Yield vs Production"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------- AREA VS PRODUCTION (preserved) ----------
    st.subheader("🌧️ Area vs Production")

    fig_area = px.scatter(
        filtered_df,
        x="Area",
        y="Production",
        color="Season",
        hover_data=["Crop"],
        title="Area vs Production"
    )
    st.plotly_chart(fig_area, use_container_width=True)

    # ---------- CROP DISTRIBUTION (preserved) ----------
    st.subheader("🥬 Crop Distribution")

    crop_count = filtered_df["Crop"].value_counts().head(10)

    fig_distribution = px.pie(
        names=crop_count.index,
        values=crop_count.values,
        title="Crop Distribution"
    )
    st.plotly_chart(fig_distribution, use_container_width=True)


# ================================================================
# SECTION 3: 🤖 AI PREDICTION (upgraded from original)
# ================================================================
elif page == "🤖 AI Prediction":

    st.header("🤖 AI-Powered Crop Production Prediction")

    st.info(
        "💡 **Note:** This model provides **decision-support estimates** based on "
        "historical agricultural data. Predictions are not guaranteed future outcomes. "
        "Always consider local soil, water, climate, and expert guidance."
    )

    if model_result is None:
        st.warning("⚠️ Not enough data to train the prediction model (need at least 20 records with positive Area and Production).")
    else:
        model = model_result["model"]
        season_enc = model_result["season_encoder"]
        crop_enc = model_result["crop_encoder"]

        st.subheader("📝 Enter Details for Production Prediction")

        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            area_input = st.number_input(
                "Enter Cultivated Area (hectares)",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

        with p_col2:
            # Show only seasons the model was trained on
            available_seasons = list(season_enc.classes_)
            season_input = st.selectbox("Select Season", available_seasons)

        with p_col3:
            # Show only crops the model was trained on
            available_crops = list(crop_enc.classes_)
            crop_input = st.selectbox("Select Crop", available_crops)

        # Make prediction
        if st.button("🔮 Predict Production"):
            try:
                season_encoded = season_enc.transform([season_input])[0]
                crop_encoded = crop_enc.transform([crop_input])[0]

                input_features = pd.DataFrame(
                    [[area_input, season_encoded, crop_encoded]],
                    columns=model_result["feature_cols"]
                )
                prediction = model.predict(input_features)[0]

                # Ensure prediction is not negative
                prediction = max(0, prediction)

                st.success(f"📊 **Estimated Production: {prediction:,.2f}**")

                st.markdown(f"""
                <div class="info-box">
                <strong>Prediction Details:</strong>
                <ul>
                    <li><strong>Area:</strong> {area_input:,.2f} hectares</li>
                    <li><strong>Season:</strong> {season_input}</li>
                    <li><strong>Crop:</strong> {crop_input}</li>
                    <li><strong>Model:</strong> Random Forest Regressor</li>
                </ul>
                <p><em>Based on historical data, the model estimates this production value.
                Actual production depends on weather, soil quality, irrigation, and many
                other factors not captured in this dataset.</em></p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

        # Feature Importance
        st.subheader("📊 Feature Importance")
        st.markdown("*How much each input feature contributes to the prediction:*")

        imp = model_result["importances"]
        fig_imp = px.bar(
            x=list(imp.keys()),
            y=list(imp.values()),
            labels={"x": "Feature", "y": "Importance"},
            title="Feature Importance in Production Prediction",
            color=list(imp.values()),
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ================================================================
# SECTION 4: 🌱 SUSTAINABLE DECISION SUPPORT
# ================================================================
elif page == "🌱 Sustainable Decision Support":

    st.header("🌱 Sustainable Agricultural Decision Support")

    st.info(
        "⚠️ **Disclaimer:** The following recommendations are **decision-support suggestions** "
        "based solely on historical agricultural data. They are **not professional agricultural "
        "advice**. Always consult local agricultural experts, consider soil conditions, water "
        "availability, climate forecasts, and regional guidance before making final decisions."
    )

    # Check if user has selected specific filters
    has_selection = (
        selected_state != "All" or selected_crop != "All" or selected_season != "All"
    )

    if not has_selection:
        st.warning("🔎 Please select at least a **State**, **Crop**, or **Season** from the sidebar filters to receive tailored decision-support insights.")
    else:
        st.subheader("📋 Analysis for Your Selection")

        # Show current selection
        sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)
        sel_col1.metric("State", selected_state)
        sel_col2.metric("District", selected_district)
        sel_col3.metric("Season", selected_season)
        sel_col4.metric("Crop", selected_crop)

        if len(filtered_df) == 0:
            st.warning("No data available for the selected combination.")
        else:
            # Basic statistics for the selection
            avg_production = filtered_df["Production"].mean()
            avg_yield = filtered_df["Yield"].mean()
            avg_area = filtered_df["Area"].mean()
            total_records = len(filtered_df)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Production", f"{avg_production:,.2f}")
            m2.metric("Avg Yield", f"{avg_yield:,.2f}")
            m3.metric("Avg Area", f"{avg_area:,.2f}")
            m4.metric("Records", total_records)

            # Compare to overall averages
            overall_avg_prod = df["Production"].mean()
            overall_avg_yield = df["Yield"].mean()

            st.subheader("📊 Comparison with Overall Averages")

            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                prod_ratio = (avg_production / overall_avg_prod * 100) if overall_avg_prod > 0 else 0
                if prod_ratio > 100:
                    st.success(
                        f"✅ Based on historical data, this selection shows **{prod_ratio:.0f}%** "
                        f"of the overall average production — **above average**."
                    )
                elif prod_ratio > 0:
                    st.warning(
                        f"📉 Based on historical data, this selection shows **{prod_ratio:.0f}%** "
                        f"of the overall average production — **below average**."
                    )

            with comp_col2:
                yield_ratio = (avg_yield / overall_avg_yield * 100) if overall_avg_yield > 0 else 0
                if yield_ratio > 100:
                    st.success(
                        f"✅ Based on historical data, this selection shows **{yield_ratio:.0f}%** "
                        f"of the overall average yield — **above average**."
                    )
                elif yield_ratio > 0:
                    st.warning(
                        f"📉 Based on historical data, this selection shows **{yield_ratio:.0f}%** "
                        f"of the overall average yield — **below average**."
                    )

            # Recommendation
            st.subheader("💡 Decision-Support Recommendation")

            # Build a contextual recommendation
            crop_text = selected_crop if selected_crop != "All" else "the selected crops"
            season_text = selected_season if selected_season != "All" else "the selected seasons"
            state_text = selected_state if selected_state != "All" else "the selected states"

            if avg_production > overall_avg_prod and avg_yield > overall_avg_yield:
                st.markdown(f"""
                <div class="info-box">
                <h4>🟢 Favorable Historical Performance</h4>
                <p>Based on historical agricultural records, <strong>{crop_text}</strong> during
                <strong>{season_text}</strong> in <strong>{state_text}</strong> shows comparatively
                strong production and yield performance compared to overall averages.</p>
                <p><strong>Suggestion:</strong> Consider evaluating soil quality, water availability,
                current climate conditions, and local agricultural guidance before making a final
                decision. Historical performance does not guarantee future results.</p>
                </div>
                """, unsafe_allow_html=True)
            elif avg_production > 0:
                st.markdown(f"""
                <div class="info-box">
                <h4>🟡 Moderate Historical Performance</h4>
                <p>Based on historical agricultural records, <strong>{crop_text}</strong> during
                <strong>{season_text}</strong> in <strong>{state_text}</strong> shows moderate
                production performance. The yield or production may be below the overall average.</p>
                <p><strong>Suggestion:</strong> Consider exploring alternative crop–season combinations
                that may offer better historical productivity for this region. Consult local agricultural
                experts and extension services for region-specific recommendations.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                <h4>🔴 Limited Historical Data</h4>
                <p>Based on available records, there is very limited or zero production data for this
                combination. This does not necessarily mean the crop will not perform — it may simply
                indicate a data gap.</p>
                <p><strong>Suggestion:</strong> Consult local agricultural authorities and conduct
                small-scale trials before committing significant resources.</p>
                </div>
                """, unsafe_allow_html=True)

            # Top crops for the selected region/season
            if selected_state != "All":
                st.subheader(f"🏆 Top 5 Crops in {selected_state}")
                region_df = df[df["State_Name"] == selected_state]
                if selected_season != "All":
                    region_df = region_df[region_df["Season"] == selected_season]
                    st.caption(f"Filtered for season: {selected_season}")

                top_crops = (
                    region_df.groupby("Crop")["Production"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )

                if len(top_crops) > 0:
                    fig_top = px.bar(
                        x=top_crops.index,
                        y=top_crops.values,
                        labels={"x": "Crop", "y": "Total Production"},
                        title=f"Top Producing Crops — {selected_state}",
                        color=top_crops.values,
                        color_continuous_scale="Greens"
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
                    st.caption("*Based on historical production data. Past performance does not guarantee future results.*")


# ================================================================
# SECTION 5: 📈 MODEL PERFORMANCE
# ================================================================
elif page == "📈 Model Performance":

    st.header("📈 Model Performance Metrics")

    if model_result is None:
        st.warning("⚠️ Not enough data to train the model.")
    else:
        st.markdown("""
        The model was evaluated on a **held-out test set** (20% of the data) to measure
        how well it generalizes to unseen records. Below are the key performance metrics.
        """)

        # Metrics cards
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "📏 MAE (Mean Absolute Error)",
            f"{model_result['mae']:,.2f}",
            help="Average absolute difference between predicted and actual production. Lower is better."
        )
        m2.metric(
            "📐 RMSE (Root Mean Squared Error)",
            f"{model_result['rmse']:,.2f}",
            help="Square root of the average squared errors. Penalizes large errors more. Lower is better."
        )
        m3.metric(
            "📊 R² Score",
            f"{model_result['r2']:.4f}",
            help="Proportion of variance explained by the model. 1.0 = perfect, 0.0 = no better than average. Higher is better."
        )

        # Explain metrics in simple terms
        st.subheader("📖 What Do These Metrics Mean?")

        st.markdown(f"""
        | Metric | Value | Meaning |
        |--------|-------|---------|
        | **MAE** | {model_result['mae']:,.2f} | On average, predictions differ from actual values by this amount |
        | **RMSE** | {model_result['rmse']:,.2f} | Similar to MAE but penalizes large errors more heavily |
        | **R² Score** | {model_result['r2']:.4f} | {('The model explains a good portion of the variation in production' if model_result['r2'] > 0.7 else 'The model captures some patterns but has room for improvement')} |
        """)

        # Actual vs Predicted scatter
        st.subheader("📊 Actual vs Predicted Production")

        y_test = model_result["y_test"]
        y_pred = model_result["y_pred"]

        fig_avp = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": "Actual Production", "y": "Predicted Production"},
            title="Actual vs Predicted Production (Test Set)",
            opacity=0.5
        )

        # Add perfect-prediction reference line
        max_val = max(y_test.max(), y_pred.max())
        fig_avp.add_shape(
            type="line",
            x0=0, y0=0,
            x1=max_val, y1=max_val,
            line=dict(color="green", width=2, dash="dash")
        )
        fig_avp.update_layout(
            annotations=[
                dict(
                    x=max_val * 0.7, y=max_val * 0.75,
                    text="Perfect Prediction Line",
                    showarrow=False,
                    font=dict(color="green", size=12)
                )
            ]
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        # Residual distribution
        st.subheader("📉 Prediction Error Distribution")

        residuals = y_test.values - y_pred
        fig_res = px.histogram(
            x=residuals,
            nbins=50,
            labels={"x": "Prediction Error (Actual - Predicted)", "y": "Count"},
            title="Distribution of Prediction Errors"
        )
        fig_res.update_layout(showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)

        st.caption(
            "*A distribution centered near zero indicates the model is not systematically "
            "over- or under-predicting. Wider spread indicates higher uncertainty.*"
        )


# ================================================================
# SECTION 6: 🧠 AI INSIGHTS
# ================================================================
elif page == "🧠 AI Insights":

    st.header("🧠 AI-Driven Agricultural Insights")

    st.markdown("""
    > All insights below are derived from **historical agricultural data** in the dataset.
    > They reflect past patterns and should not be treated as guaranteed future outcomes.
    """)

    # Use the filtered data for context, but full data for overall insights
    analysis_df = df[(df["Area"] > 0) & (df["Production"] > 0)].copy()

    # --- Insight 1: Highest-Producing Crops ---
    st.subheader("🌾 Highest-Producing Crops")

    top_prod_crops = (
        analysis_df.groupby("Crop")["Production"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig_tp = px.bar(
        x=top_prod_crops.index,
        y=top_prod_crops.values,
        labels={"x": "Crop", "y": "Total Production"},
        title="Top 10 Crops by Total Historical Production",
        color=top_prod_crops.values,
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig_tp, use_container_width=True)
    st.caption("*Based on historical data, these crops have the highest cumulative production across all states and seasons.*")

    # --- Insight 2: Highest-Yield Crops ---
    st.subheader("📈 Highest-Yield Crops")

    top_yield_crops = (
        analysis_df.groupby("Crop")["Yield"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig_ty = px.bar(
        x=top_yield_crops.index,
        y=top_yield_crops.values,
        labels={"x": "Crop", "y": "Average Yield"},
        title="Top 10 Crops by Average Historical Yield",
        color=top_yield_crops.values,
        color_continuous_scale="YlGn"
    )
    st.plotly_chart(fig_ty, use_container_width=True)
    st.caption("*Based on historical data, these crops show the highest average yield per unit area.*")

    # --- Insight 3: Seasonal Production Patterns ---
    st.subheader("🍂 Seasonal Production Patterns")

    season_stats = (
        analysis_df.groupby("Season")
        .agg(
            Total_Production=("Production", "sum"),
            Avg_Production=("Production", "mean"),
            Avg_Yield=("Yield", "mean"),
            Record_Count=("Production", "count")
        )
        .sort_values("Total_Production", ascending=False)
        .reset_index()
    )

    st.dataframe(season_stats, use_container_width=True)

    fig_sp = px.bar(
        season_stats,
        x="Season",
        y="Total_Production",
        color="Avg_Yield",
        labels={"Total_Production": "Total Production", "Avg_Yield": "Avg Yield"},
        title="Production and Yield by Season",
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig_sp, use_container_width=True)
    st.caption("*Based on historical data, seasonal patterns show variation in both total production and average yield.*")

    # --- Insight 4: Area vs Production Relationship ---
    st.subheader("🔗 Area vs Production Relationship")

    # Calculate correlation
    correlation = analysis_df[["Area", "Production"]].corr().iloc[0, 1]

    st.markdown(f"""
    **Correlation between Area and Production:** `{correlation:.4f}`

    {"Based on historical data, there is a **strong positive relationship** between cultivated area and production. Larger cultivated areas tend to produce more." if correlation > 0.7 else "Based on historical data, the relationship between area and production is **moderate**, suggesting that other factors beyond area significantly influence production." if correlation > 0.4 else "Based on historical data, the relationship between area and production is **weak**, indicating that factors other than area (such as crop type, season, and regional conditions) play a major role in determining production."}
    """)

    # --- Insight 5: Crops with Best Productivity ---
    st.subheader("⭐ Crops with Comparatively Better Historical Productivity")

    # Calculate a simple productivity score (avg yield * avg production)
    crop_perf = (
        analysis_df.groupby("Crop")
        .agg(
            Avg_Yield=("Yield", "mean"),
            Avg_Production=("Production", "mean"),
            Total_Production=("Production", "sum"),
            Districts_Grown=("District_Name", "nunique")
        )
        .sort_values("Avg_Yield", ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(crop_perf, use_container_width=True)
    st.caption(
        "*Based on historical data, these crops show comparatively better productivity metrics. "
        "Actual performance varies by region, soil, climate, and management practices.*"
    )


# ================================================================
# SECTION 7: ⚖️ RESPONSIBLE AI
# ================================================================
elif page == "⚖️ Responsible AI":

    st.header("⚖️ Responsible AI Considerations")

    st.markdown("""
    This section outlines the ethical and responsible use of AI in this agricultural
    decision-support system. It is important that users understand the capabilities
    **and limitations** of this tool.
    """)

    # Fairness
    st.subheader("⚖️ Fairness")
    st.markdown("""
    - The model may reflect **biases or limitations** present in historical agricultural data.
    - If certain regions, crops, or seasons are underrepresented in the dataset, predictions
      for those categories may be less accurate.
    - The system does not intentionally favour any particular region, crop, or community.
    """)

    # Transparency
    st.subheader("🔍 Transparency")
    st.markdown("""
    - All predictions are generated from **historical agricultural data** using a
      **Random Forest Regressor** machine learning model.
    - The model uses **Area, Season, and Crop** as input features to predict Production.
    - Model performance metrics (MAE, RMSE, R²) are openly displayed so users can assess
      the model's reliability.
    - This project uses **AI/ML-powered agricultural decision support**. It does **not**
      use generative AI, IBM Granite, RAG, or AI agents.
    """)

    # Privacy
    st.subheader("🔒 Privacy")
    st.markdown("""
    - This application does **not** collect any personal information from users.
    - No login, registration, or personal data is required to use this tool.
    - The dataset contains only aggregated agricultural statistics at the
      State/District level — no individual farmer data is used.
    """)

    # Accuracy
    st.subheader("🎯 Accuracy")
    st.markdown("""
    - Predictions should **not** be treated as guaranteed outcomes.
    - The model provides **estimates** based on patterns in historical data.
    - Actual crop production depends on many factors beyond what is captured in this dataset.
    - Users should view predictions as one input among many in their decision-making process.
    """)

    # Human Oversight
    st.subheader("👤 Human Oversight")
    st.markdown("""
    - Final agricultural decisions should always involve:
      - **Farmers** with local knowledge
      - **Agricultural experts** and extension officers
      - Consideration of **current local conditions** (soil, weather, water)
    - This tool is designed to **support** human decision-making, not replace it.
    - AI-generated insights should be **validated** with domain expertise before action.
    """)

    # Data Limitations
    st.subheader("📊 Data Limitations")
    st.warning("""
    **Important:** The dataset used in this project contains only the following variables:
    State, District, Crop, Season, Area, Production, and Yield.

    The dataset does **NOT** contain critical agricultural factors such as:
    - 🌧️ **Weather / Rainfall** data
    - 🌍 **Soil conditions** and quality
    - 💧 **Irrigation** availability
    - 🧪 **Fertilizer** usage
    - 🐛 **Pest and disease** information
    - 🌡️ **Temperature** and climate data
    - 💰 **Market prices** and economic factors

    The model does **not** consider these variables because they are not present in the data.
    Predictions and insights are therefore based on a **limited set of factors** and should
    be interpreted cautiously.
    """)

    # Model Limitations
    st.subheader("⚠️ Model Limitations")
    st.markdown("""
    1. **Historical Data Only:** The model learns from past records and cannot account for
       unprecedented events (climate change, new pests, policy changes).
    2. **Aggregated Data:** The dataset contains State/District-level aggregates, not
       individual farm-level data. Predictions may not reflect conditions at a specific farm.
    3. **Prediction Uncertainty:** All predictions carry inherent uncertainty. The model
       may perform poorly for rare crop–region combinations with few historical records.
    4. **No Causal Claims:** The model identifies statistical patterns (correlations),
       not causal relationships. A high correlation between area and production does not
       mean increasing area will always increase production.
    5. **Single Time Period:** The current dataset covers 2024-25 data. Trends over
       multiple years are not captured.
    6. **Expert Validation Required:** Any insight or recommendation from this system
       should be validated by agricultural domain experts before implementation.
    """)


# ================================================================
# SECTION 8: 📥 DOWNLOAD RESULTS (preserved from original)
# ================================================================
elif page == "📥 Download Results":

    st.header("📥 Download Results")

    st.subheader("⬇️ Download Filtered Dataset")

    st.markdown(f"""
    **Current Filters:**
    - State: `{selected_state}` | District: `{selected_district}`
    - Season: `{selected_season}` | Crop: `{selected_crop}`
    - **Records:** {len(filtered_df):,}
    """)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name="filtered_crop_report.csv",
        mime="text/csv"
    )

    # Preview what will be downloaded
    st.subheader("📄 Preview")
    st.dataframe(
        filtered_df.head(20),
        use_container_width=True
    )


# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    """
    <center>
    <h4 style='color:green;'>
    🌾 AI-Powered Sustainable Crop Production & Decision Support System
    </h4>
    <p style='color:gray;'>
    Developed for the 1M1B AI for Sustainability Virtual Internship |
    IBM SkillsBuild & AICTE
    </p>
    </center>
    """,
    unsafe_allow_html=True
)