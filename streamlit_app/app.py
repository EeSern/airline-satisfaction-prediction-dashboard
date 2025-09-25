# STREAMLIT DASHBOARD
import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import datetime

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD ASSETS
@st.cache_resource
def load_assets():
    """
    Loads all necessary assets from the disk. This function runs only once.
    """
    try:
        model = joblib.load('final_xgboost_model.joblib')
        with open('final_features.json', 'r') as f:
            feature_info = json.load(f)
        template_data = pd.read_csv('airline_data_final_10_features.csv')
        # Load the new, un-encoded data for charting
        charts_data = pd.read_csv('airline_data_for_charts.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading assets: {e}. Make sure all required files (including airline_data_for_charts.csv) are in the same folder as app.py.")
        return None, None, None, None
    
    return model, feature_info, template_data, charts_data

model, feature_info, template_data, charts_data = load_assets()

# HELPER FUNCTIONS
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# MAIN APP STRUCTURE
if model is None:
    st.error("Application cannot start due to missing asset files.")
    st.stop()

st.title("✈️ Airline Passenger Satisfaction Dashboard")
st.markdown("An interactive dashboard to predict passenger satisfaction and explore key business insights.")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "tab1"

features = feature_info['features_for_model']
tab1, tab2, tab3 = st.tabs(["📊 **Descriptive Analysis & Insights**", "👤 **Single Passenger Prediction**", "📁 **Batch Prediction**"])

# TAB 1: DESCRIPTIVE ANALYSIS
with tab1:
    st.header("Visualizing the Key Drivers of Satisfaction")
    st.markdown("Explore the relationships that our predictive model found to be most important.")
    
    st.subheader("Top 10 Most Impactful Features for Prediction")
    
    # Two columns to make a numbered list.
    col_a, col_b = st.columns(2)
    midpoint = len(features) // 2
    
    with col_a:
        for i, feature in enumerate(features[:midpoint]):
            st.markdown(f"**{i+1}.** {feature.replace('_', ' ')}")
            
    with col_b:
        for i, feature in enumerate(features[midpoint:]):
            st.markdown(f"**{i+midpoint+1}.** {feature.replace('_', ' ')}")

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        # Interactive Chart 1: Service Ratings
        st.subheader("How Do Service Ratings Affect Satisfaction?")
        rating_options = [
            'Online Boarding', 'In-flight Wifi Service', 'Leg Room Service', 'Cleanliness',
            'In-flight Service', 'Baggage Handling', 'On-board Service', 'Ease of Online Booking',
            'Check-in Service', 'In-flight Entertainment', 'Seat Comfort'
        ]
        final_rating_options = [feature for feature in rating_options if feature in features]
        selected_rating = st.selectbox("Choose a service to analyze:", final_rating_options)
        
        if selected_rating and charts_data is not None:
            rating_satisfaction = charts_data.groupby(selected_rating)['Satisfaction_Numeric'].apply(lambda x: (x == 0).mean()).mul(100).reset_index()
            rating_satisfaction.columns = [selected_rating, 'percent']
            
            fig1 = px.bar(
                rating_satisfaction,
                x=selected_rating,
                y='percent',
                title=f"Satisfaction Rate vs. '{selected_rating}' Rating",
                labels={selected_rating: "Service Rating (1-5)", 'percent': "Percentage Satisfied (%)"},
                text=rating_satisfaction['percent'].apply(lambda x: f'{x:.1f}%')
            )
            fig1.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Interactive Chart 2: Passenger Segments
        st.subheader("Satisfaction Across Different Passenger Segments")
        segment_options = {
            "Travel Type": 'Type of Travel',
            "Customer Type": 'Customer Type',
            "Class": 'Class'
        }
        selected_segment = st.selectbox("Choose a segment to compare:", list(segment_options.keys()))
        
        if charts_data is not None:
            segment_col = segment_options[selected_segment]
            segment_satisfaction = charts_data.groupby(segment_col)['Satisfaction_Numeric'].apply(lambda x: (x == 0).mean()).mul(100).reset_index(name='Satisfied_pct')
            
            fig2 = px.pie(
                segment_satisfaction,
                names=segment_col,
                values='Satisfied_pct',
                title=f"Satisfaction Rate by {selected_segment}",
                hole=0.4
            )
            fig2.update_traces(textinfo='percent+label', pull=[0.05] * len(segment_satisfaction))
            st.plotly_chart(fig2, use_container_width=True)

    # DETAILED BUSINESS RECOMMENDATIONS
    st.markdown("---")
    st.subheader("Actionable Business Recommendations")
    
    st.markdown("Based on the key drivers identified by our predictive model, we propose the following strategic recommendations:")

    # --- Recommendation 1 ---
    st.markdown("#### 1. Invest in the Digital Experience")
    st.markdown("""
    *   **Optimize the Online Boarding Process:** Launch a project to refine the mobile app and website's boarding process. The primary goal should be a fast, intuitive, and user-friendly interface.
    *   **Upgrade In-flight Wi-Fi:** Invest in upgrading the Wi-Fi infrastructure to support modern internet usage. Consider offering a free basic tier and a paid high-performance tier to meet different passenger needs.
    """)

    # --- Recommendation 2 ---
    st.markdown("#### 2. Develop Segment-Specific Service Strategies")
    st.markdown("""
    *   **Address the 'Personal & Economy' Experience:** Conduct targeted surveys for personal and economy class travelers to better understand their specific expectations and pain points. This may uncover valuable insights related to baggage policies, in-flight comfort, or service perceptions.
    """)

    # --- Recommendation 3 ---
    st.markdown("#### 3. Maintain and Elevate Core Service Quality")
    st.markdown("""
    *   **Enforce Strict Cleanliness Protocols:** *Cleanliness* was identified as a key feature. Regular audits and strict protocols are essential to maintain high standards.
    *   **Uphold Service Excellence:** *On-board Service* and *In-flight Service* are consistently important. Implement routine staff training programs to ensure a high and consistent quality of passenger interaction.
    """)

    # --- Recommendation 4 ---
    st.markdown("#### 4. Enhance the Loyalty Program")
    st.markdown("""
    *   **Personalize Rewards:** Leverage passenger data to offer personalized promotions, upgrades, and exclusive experiences that are relevant to their travel history and preferences.
    *   **Gamify Engagement:** Introduce clear tiered membership levels with visible point tracking to encourage engagement and tangibly reward customer loyalty.
    """)

# TAB 2: SINGLE PASSENGER PREDICTION
with tab2:
    st.header("Predict Satisfaction for an Individual Passenger")

    # This session state variable will control when the prediction is shown
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False

    # Function to be called when the form is submitted
    def handle_submit():
        st.session_state.show_prediction = True

    with st.form("single_prediction_form"):
        st.markdown("Please provide the passenger's details and ratings:")
        user_input = {}
        categories = {
            "Passenger & Journey Details": ["Type of Travel_Personal", "Customer Type_Returning", "Class_Economy"],
            "In-Flight Experience Ratings": ["In-flight Wifi Service", "Leg Room Service", "On-board Service", "Cleanliness", "In-flight Service"],
            "Pre-Flight & Ground Experience Ratings": ["Online Boarding", "Baggage Handling"]
        }
        def create_widget(feature):
            if any(kw in feature for kw in ["Service", "Cleanliness", "Boarding", "Room", "Handling"]):
                return st.slider(f"{feature.replace('_', ' ')} (1-5)", 1, 5, 3, key=feature)
            elif "Type of Travel_Personal" == feature:
                val = st.selectbox("Type of Travel", ["Business", "Personal"], key=feature)
                return 1 if val == "Personal" else 0
            elif "Customer Type_Returning" == feature:
                val = st.selectbox("Customer Type", ["First-time", "Returning"], key=feature)
                return 1 if val == "Returning" else 0
            elif "Class_Economy" == feature:
                val = st.selectbox("Class", ["Business", "Economy", "Economy Plus"], key=feature)
                return 1 if val == "Economy" else 0
            else:
                return st.number_input(f"Value for {feature}", value=0, key=feature)
        for category, cat_features in categories.items():
            features_in_cat = [f for f in cat_features if f in features]
            if features_in_cat:
                st.subheader(category)
                for feature in features_in_cat:
                    user_input[feature] = create_widget(feature)
        st.markdown("---")

        # Handle_submit function is called
        submit_button = st.form_submit_button(label='✨ Get Prediction', on_click=handle_submit)

        if submit_button:
            # When submitted, save the input data to the session state
            st.session_state['user_input'] = user_input

    if st.session_state.show_prediction:
        user_input_data = st.session_state['user_input']
        
        # Perform the prediction
        input_df = pd.DataFrame([user_input_data], columns=features)
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display the prediction result
        st.subheader("🔮 Prediction Result")
        if prediction[0] == 0:
            st.success("This passenger is likely to be **SATISFIED** ✅")
            st.metric("Confidence Score (Satisfied)", f"{prediction_proba[0][0]:.2%}")
        else: # If prediction is Dissatisfied (1)
            st.error("This passenger is likely to be **NEUTRAL OR DISSATISFIED** ❌")
            st.metric("Confidence Score (Dissatisfied)", f"{prediction_proba[0][1]:.2%}")

            st.markdown("---")
            st.subheader("💬 Help Us Improve!")
            st.write("To help us understand, could you please provide a brief reason for your ratings?")
            
            with st.form("feedback_form"):
                feedback_text = st.text_area("Reason for dissatisfaction (optional):", height=100)
                feedback_submit_button = st.form_submit_button("Submit Feedback")

                if feedback_submit_button:
                    # Log the feedback to a CSV
                    feedback_record = user_input_data.copy()
                    feedback_record['Timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feedback_record['Predicted_Satisfaction'] = prediction[0]
                    feedback_record['Feedback_Text'] = feedback_text
                    feedback_df = pd.DataFrame([feedback_record])
                    try:
                        header = not pd.io.common.file_exists('feedback_log.csv')
                        feedback_df.to_csv('feedback_log.csv', mode='a', header=header, index=False)
                        st.success("Thank you! Your feedback has been recorded.")
                    except Exception as e:
                        st.error(f"Could not save feedback: {e}")

# TAB 3: BATCH PREDICTION
with tab3:
    st.header("Get Predictions for a File")
    st.download_button(
        label="📥 Download Template CSV",
        data=convert_df_to_csv(template_data),
        file_name="prediction_template.csv",
        mime="text/csv"
    )
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload a CSV file with the same columns as the template", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())
            
            batch_df_processed = batch_df[features]
            predictions = model.predict(batch_df_processed)
            predictions_proba = model.predict_proba(batch_df_processed)
            
            results_df = batch_df.copy()

            results_df['Predicted_Satisfaction'] = ["Satisfied" if p == 0 else "Neutral or Dissatisfied" for p in predictions]
            results_df['Probability_Satisfied'] = [f"{p[0]:.2%}" for p in predictions_proba]
            
            st.subheader("✅ Prediction Results")
            st.dataframe(results_df)
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=convert_df_to_csv(results_df),
                file_name="prediction_results.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please ensure your uploaded file has the exact same columns as the template file.")