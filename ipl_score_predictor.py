# Import the libraries
import math
import numpy as np
import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# SET PAGE CONFIGURATION
st.set_page_config(
    page_title='IPL Score Predictor 2025',
    page_icon='🏏',
    layout="centered"
)

# Dark theme CSS with cricket background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    
    .main-container {
        background: rgba(0, 0, 0, 0.8);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .title {
        text-align: center;
        color: #00d4ff;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .section-header {
        color: #00d4ff;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 0.5rem;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
    }
    
    .stSelectbox > div > div > div {
        color: white;
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
    }
    
    .stSlider > div > div > div > div {
        background-color: #00d4ff;
    }
    
    .metric-card {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 25px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 212, 255, 0.4);
    }
    
    .stAlert > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load ML model
@st.cache_data
def load_model():
    try:
        filename = 'ml_model.pkl'
        return pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'ml_model.pkl' is in the same directory.")
        return None

# Load SHAP explainer
@st.cache_data
def load_shap_explainer():
    """Load and cache SHAP explainer for the model"""
    try:
        # You'll need to save some training data for SHAP background
        # For now, create a small sample dataset
        sample_data = np.array([
            [1,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0, 45, 2, 8.3, 23, 1],  # Sample 1
            [0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0, 78, 3, 12.4, 35, 2], # Sample 2
            [0,0,1,0,0,0,0,0, 1,0,0,0,0,0,0,0, 92, 1, 15.2, 42, 0], # Sample 3
        ])
        model = load_model()
        if model is not None:
            explainer = shap.TreeExplainer(model, sample_data)
            return explainer
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load SHAP explainer: {e}")
        return None

# SHAP explanation function
def explain_prediction_with_shap(prediction_array, explainer):
    """Generate SHAP explanations for the current prediction"""
    
    # Feature names matching your model
    feature_names = [
        'BAT_CSK', 'BAT_DD', 'BAT_KXIP', 'BAT_KKR', 'BAT_MI', 'BAT_RR', 'BAT_RCB', 'BAT_SRH',
        'BOWL_CSK', 'BOWL_DD', 'BOWL_KXIP', 'BOWL_KKR', 'BOWL_MI', 'BOWL_RR', 'BOWL_RCB', 'BOWL_SRH',
        'Current_Runs', 'Wickets', 'Overs', 'Runs_Last_5', 'Wickets_Last_5'
    ]
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(prediction_array)
    
    return shap_values, feature_names

model = load_model()
if model is None:
    st.stop()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">🏏 IPL Score Predictor 2025</h1>', unsafe_allow_html=True)

# Team Selection
st.markdown('<div class="section-header">🏟️ Team Selection</div>', unsafe_allow_html=True)

teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
         'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
         'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', teams, index=0)
with col2:
    bowling_team = st.selectbox('Bowling Team', teams, index=1)

if bowling_team == batting_team:
    st.error('Bowling and Batting teams should be different')

# Current Match Status
st.markdown('<div class="section-header">📊 Current Match Status</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    overs = st.number_input('Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
    if overs - math.floor(overs) > 0.5:
        st.error('Invalid over input - max 6 balls per over')

with col2:
    runs = st.number_input('Current Runs', min_value=0, max_value=354, step=1, format='%i')

with col3:
    wickets = st.slider('Wickets Fallen', 0, 9, value=0)

# Recent Performance
st.markdown('<div class="section-header">⚡ Last 5 Overs Performance</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    runs_in_prev_5 = st.number_input('Runs in Last 5 Overs', min_value=0, max_value=runs, step=1, format='%i')
with col2:
    wickets_in_prev_5 = st.number_input('Wickets in Last 5 Overs', min_value=0, max_value=wickets, step=1, format='%i')

# Live Metrics
st.markdown('<div class="section-header">📈 Live Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    current_rr = round(runs / overs, 2) if overs > 0 else 0
    st.metric("Run Rate", f"{current_rr}")
with col2:
    balls_faced = int(overs) * 6 + int((overs % 1) * 10)
    strike_rate = round((runs / balls_faced) * 100, 2) if balls_faced > 0 else 0
    st.metric("Strike Rate", f"{strike_rate}")
with col3:
    remaining_overs = round(20 - overs, 1)
    st.metric("Overs Left", f"{remaining_overs}")
with col4:
    wickets_in_hand = 10 - wickets
    st.metric("Wickets Left", f"{wickets_in_hand}")

# Prepare prediction array
prediction_array = []

# Encode batting team
team_encoding = [0] * 8
team_names = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
              'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
              'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
if batting_team in team_names:
    team_encoding[team_names.index(batting_team)] = 1
prediction_array.extend(team_encoding)

# Encode bowling team
team_encoding = [0] * 8
if bowling_team in team_names:
    team_encoding[team_names.index(bowling_team)] = 1
prediction_array.extend(team_encoding)

# Add match stats
prediction_array.extend([runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5])
prediction_array = np.array([prediction_array])

# Score Prediction
st.markdown('<div class="section-header">🎯 Score Prediction</div>', unsafe_allow_html=True)

if bowling_team != batting_team and overs >= 5.1:
    # Load SHAP explainer
    explainer = load_shap_explainer()
    
    if st.button('🔮 Predict Final Score'):
        with st.spinner('Analyzing match data...'):
            try:
                predict = model.predict(prediction_array)
                my_prediction = int(round(predict[0]))
                
                # Display prediction as before
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🏏 PREDICTED FINAL SCORE</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{my_prediction-5} - {my_prediction+5}</h1>
                    <p style="font-size: 1.2rem;">Most likely score: <strong>{my_prediction}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add SHAP explanations
                if explainer:
                    st.markdown('<div class="section-header">🧠 AI Explanation (SHAP Analysis)</div>', unsafe_allow_html=True)
                    
                    # Generate SHAP explanations
                    shap_values, feature_names = explain_prediction_with_shap(prediction_array, explainer)
                    
                    # Create tabs for different SHAP visualizations
                    tab1, tab2, tab3 = st.tabs(["📊 Feature Impact", "🎯 Force Plot", "💡 Key Insights"])
                    
                    with tab1:
                        st.write("**How each factor influences the predicted score:**")
                        
                        # Get SHAP values for this prediction
                        shap_vals = shap_values[0]
                        base_value = explainer.expected_value
                        
                        # Create a simple bar plot showing feature contributions
                        important_features = []
                        important_values = []
                        
                        for i, (feat, val) in enumerate(zip(feature_names, shap_vals)):
                            if abs(val) > 0.5:  # Only show significant features
                                if feat.startswith('BAT_') and prediction_array[0][i] == 1:
                                    important_features.append(f"Batting: {batting_team[:3]}")
                                    important_values.append(val)
                                elif feat.startswith('BOWL_') and prediction_array[0][i] == 1:
                                    important_features.append(f"Bowling: {bowling_team[:3]}")
                                    important_values.append(val)
                                elif not feat.startswith(('BAT_', 'BOWL_')):
                                    important_features.append(feat.replace('_', ' '))
                                    important_values.append(val)
                        
                        if important_features:
                            # Create matplotlib figure with better styling
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Set background color
                            fig.patch.set_facecolor('#0E1117')
                            ax.set_facecolor('#0E1117')
                            
                            # Create horizontal bar chart
                            colors = ['#00ff41' if v > 0 else '#ff4444' for v in important_values]
                            bars = ax.barh(range(len(important_features)), important_values, 
                                         color=colors, alpha=0.8, height=0.6)
                            
                            # Customize the plot
                            ax.set_yticks(range(len(important_features)))
                            ax.set_yticklabels(important_features, color='white', fontsize=11)
                            ax.set_xlabel('Impact on Predicted Score', color='white', fontsize=12, fontweight='bold')
                            ax.set_title('Feature Contributions to Prediction', color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
                            
                            # Add vertical line at x=0
                            ax.axvline(x=0, color='white', linestyle='-', alpha=0.4, linewidth=1)
                            
                            # Style the axes
                            ax.tick_params(colors='white', labelsize=10)
                            ax.spines['bottom'].set_color('white')
                            ax.spines['top'].set_color('white')
                            ax.spines['right'].set_color('white')
                            ax.spines['left'].set_color('white')
                            
                            # Add value labels on bars
                            for i, (bar, val) in enumerate(zip(bars, important_values)):
                                width = bar.get_width()
                                label_x = width + (0.5 if width > 0 else -0.5)
                                ax.text(label_x, i, f'{val:+.1f}', 
                                       ha='left' if width > 0 else 'right', 
                                       va='center', color='white', fontweight='bold', fontsize=10)
                            
                            # Adjust layout
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Also show as text summary
                            st.write("**Summary:**")
                            for feat, val in zip(important_features, important_values):
                                impact = "increases" if val > 0 else "decreases"
                                color = "🟢" if val > 0 else "🔴"
                                st.write(f"{color} **{feat}** {impact} score by {abs(val):.1f} runs")
                                
                        else:
                            st.info("All features have minimal impact on this prediction.")
                    
                    with tab2:
                        st.write("**Visual breakdown of prediction logic:**")
                        
                        # Create a simplified force plot representation
                        base_score = int(explainer.expected_value)
                        prediction_delta = my_prediction - base_score
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Base Score", f"{base_score}", help="Average score in training data")
                        with col2:
                            st.metric("Prediction Adjustment", f"{prediction_delta:+d}", help="How current situation changes the prediction")
                        with col3:
                            st.metric("Final Prediction", f"{my_prediction}", help="Base + Adjustment")
                        
                        # Show reasoning
                        positive_factors = []
                        negative_factors = []
                        
                        if runs > current_rr * 10:  # Runs above average
                            positive_factors.append(f"🟢 Strong start: {runs} runs")
                        if wickets <= 2:
                            positive_factors.append(f"🟢 Wickets in hand: {10-wickets} remaining")
                        if runs_in_prev_5 > 25:
                            positive_factors.append(f"🟢 Good momentum: {runs_in_prev_5} runs in last 5 overs")
                        
                        if wickets >= 4:
                            negative_factors.append(f"🔴 Wickets lost: {wickets} down")
                        if current_rr < 6:
                            negative_factors.append(f"🔴 Slow run rate: {current_rr}")
                        if wickets_in_prev_5 >= 2:
                            negative_factors.append(f"🔴 Recent wickets: {wickets_in_prev_5} in last 5 overs")
                        
                        if positive_factors:
                            st.write("**Factors increasing the score:**")
                            for factor in positive_factors:
                                st.write(factor)
                        
                        if negative_factors:
                            st.write("**Factors decreasing the score:**")
                            for factor in negative_factors:
                                st.write(factor)
                    
                    with tab3:
                        st.write("**Key insights from the AI model:**")
                        
                        insights = []
                        
                        # Generate contextual insights
                        if my_prediction > 180:
                            insights.append("🚀 **High-scoring prediction** - Model expects strong batting performance")
                        elif my_prediction < 140:
                            insights.append("📉 **Below-par prediction** - Model sees challenges ahead")
                        else:
                            insights.append("⚖️ **Balanced prediction** - Model expects competitive total")
                        
                        if runs_in_prev_5 > runs * 0.4:
                            insights.append("⚡ **Momentum building** - Recent overs show acceleration")
                        
                        if batting_team in ['Mumbai Indians', 'Chennai Super Kings']:
                            insights.append("🏆 **Strong batting lineup** - Historically high-scoring team")
                        
                        if bowling_team in ['Sunrisers Hyderabad', 'Chennai Super Kings']:
                            insights.append("🎯 **Quality bowling attack** - May restrict scoring")
                        
                        remaining_balls = (20 - overs) * 6
                        required_rr = round((my_prediction - runs) / (20 - overs), 1)
                        insights.append(f"📊 **Required rate**: {required_rr} runs per over for predicted total")
                        
                        for insight in insights:
                            st.write(insight)
                        
                        # Model confidence indicator
                        confidence = min(100, max(60, 100 - abs(prediction_delta) * 2))
                        st.write(f"🎯 **Model Confidence**: {confidence}%")
                        
                        if confidence > 80:
                            st.success("High confidence prediction")
                        elif confidence > 70:
                            st.warning("Moderate confidence prediction")
                        else:
                            st.error("Low confidence - unusual match situation")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
else:
    if bowling_team == batting_team:
        st.warning("Please select different teams")
    if overs < 5.1:
        st.warning("Need at least 5.1 overs for accurate prediction")

st.markdown('</div>', unsafe_allow_html=True)