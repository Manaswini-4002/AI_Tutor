# AI Tutor Personalized Learning Experience Web App - ENHANCED
# -------------------------------------------------
# Run this with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="AI Tutor - Personalized Learning", 
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom CSS with light background colors
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background-color: #397fc4
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .success-prediction {
        background-color: #d4edda;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .failure-prediction {
        background-color: #f8d7da;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .progress-bar {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #ced4da;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #28a745, #20c997);
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 20px;
    }
    .metric-card {
        background-color: #844dab;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-light {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .sidebar-light {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .tab-light {
        background-color: #63ab88;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# App header with animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🎓 AI Tutor Pro</h1>', unsafe_allow_html=True)
    st.markdown("### 🤖 Intelligent Personalized Learning Platform")

# Initialize session state
if 'student_history' not in st.session_state:
    st.session_state.student_history = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# Generate enhanced synthetic dataset
def generate_enhanced_data(n=500):
    np.random.seed(42)
    student_id = np.arange(n)
    ability = np.random.beta(2, 2, n)  # More realistic ability distribution
    engagement = np.random.beta(3, 1.5, n)
    difficulty = np.random.beta(1.5, 2, n)
    prior_attempts = np.random.poisson(3, n)
    time_since_last = np.random.exponential(2, n)
    session_count = np.random.poisson(5, n)
    study_duration = np.random.normal(45, 15, n)  # minutes
    
    # More sophisticated success calculation
    success_prob = (
        ability * 0.4 + 
        engagement * 0.3 + 
        (1 - difficulty) * 0.2 +
        np.log1p(session_count) * 0.1
    )
    success = (success_prob > np.random.rand(n)).astype(int)
    
    df = pd.DataFrame({
        'student_id': student_id,
        'ability': ability,
        'engagement': engagement,
        'difficulty': difficulty,
        'prior_attempts': prior_attempts,
        'time_since_last': time_since_last,
        'session_count': session_count,
        'study_duration': study_duration,
        'success': success
    })
    return df

@st.cache_data
def train_enhanced_model():
    df = generate_enhanced_data(1000)  # Larger dataset
    X = df.drop(['student_id', 'success'], axis=1)
    y = df['success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    return model, acc, df, y_pred_proba

# Load model and data
model, acc, df, y_pred_proba = train_enhanced_model()

# Sidebar with enhanced inputs
st.sidebar.markdown('<div class="sidebar-light">', unsafe_allow_html=True)
st.sidebar.markdown("## 🎯 Student Profile")

# Student basic info
st.sidebar.subheader("📝 Basic Information")
student_name = st.sidebar.text_input("Student Name", "John Doe")
student_age = st.sidebar.slider("Age", 10, 25, 16)
learning_style = st.sidebar.selectbox(
    "Preferred Learning Style",
    ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"]
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-light">', unsafe_allow_html=True)
st.sidebar.subheader("📊 Learning Metrics")

ability = st.sidebar.slider("Student Ability (0-10)", 0.0, 1.0, 0.7, 0.05)
engagement = st.sidebar.slider("Engagement Level (0-10)", 0.0, 1.0, 0.6, 0.05)
difficulty = st.sidebar.slider("Content Difficulty (0-10)", 0.0, 1.0, 0.5, 0.05)
prior_attempts = st.sidebar.number_input("Prior Attempts", min_value=0, max_value=20, value=3)
time_since_last = st.sidebar.slider("Days Since Last Study", 0.0, 2.0, 4.0, 6.0)
session_count = st.sidebar.number_input("Weekly Study Sessions", min_value=1, max_value=20, value=5)
study_duration = st.sidebar.slider("Average Study Duration (min)", 15, 120, 45)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📈 Analytics", "💡 Recommendations", "📚 Learning Path"])

with tab1:
    st.markdown('<div class="section-light">', unsafe_allow_html=True)
    st.header("Learning Success Prediction")
    
    # Create input data
    input_data = pd.DataFrame({
        'ability': [ability],
        'engagement': [engagement],
        'difficulty': [difficulty],
        'prior_attempts': [prior_attempts],
        'time_since_last': [time_since_last],
        'session_count': [session_count],
        'study_duration': [study_duration]
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Predict Learning Success", use_container_width=True):
            with st.spinner('Analyzing learning patterns...'):
                time.sleep(1.5)
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Store prediction in history
                prediction_record = {
                    'timestamp': datetime.now(),
                    'student_name': student_name,
                    'prediction': prediction,
                    'confidence': max(prediction_proba),
                    'inputs': input_data.iloc[0].to_dict()
                }
                st.session_state.student_history.append(prediction_record)
                
                # Display result with enhanced UI
                if prediction == 1:
                    st.markdown('<div class="success-prediction">', unsafe_allow_html=True)
                    st.markdown("## 🎉 EXCELLENT! Student is likely to SUCCEED!")
                    st.metric("Confidence Level", f"{max(prediction_proba)*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Success recommendations
                    st.session_state.recommendations = [
                        "🎯 Challenge with advanced topics",
                        "📚 Provide extension materials",
                        "👥 Peer teaching opportunities",
                        "🚀 Accelerated learning path"
                    ]
                else:
                    st.markdown('<div class="failure-prediction">', unsafe_allow_html=True)
                    st.markdown("## 📚 NEEDS SUPPORT! Student may struggle")
                    st.metric("Risk Level", f"{(1-max(prediction_proba))*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Support recommendations
                    st.session_state.recommendations = [
                        "🔄 Review foundational concepts",
                        "🎯 Targeted practice exercises",
                        "👨‍🏫 One-on-one tutoring sessions",
                        "📊 Additional formative assessments"
                    ]
                
                # Confidence meter
                st.subheader("Confidence Level")
                confidence = max(prediction_proba) * 100
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {confidence}%">{confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="tab-light">', unsafe_allow_html=True)
        st.subheader("📊 Quick Stats")
        st.metric("Model Accuracy", f"{acc*100:.1f}%")
        st.metric("Students Analyzed", len(df))
        st.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-light">', unsafe_allow_html=True)
    st.header("📈 Learning Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="tab-light">', unsafe_allow_html=True)
        # Feature importance chart
        st.subheader("Feature Importance")
        importances = pd.Series(
            model.feature_importances_, 
            index=['Ability', 'Engagement', 'Difficulty', 'Prior Attempts', 'Time Gap', 'Sessions', 'Duration']
        ).sort_values(ascending=True)
        
        # Use Streamlit's native bar chart
        st.bar_chart(importances)
        st.write("**What Factors Most Influence Success?**")
        
        # Display importance values
        for feature, importance in importances.items():
            st.markdown(f"""
            <div class="metric-card">
                <strong>{feature}:</strong> {importance:.3f}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="tab-light">', unsafe_allow_html=True)
        # Success distribution
        st.subheader("Success Distribution")
        success_counts = df['success'].value_counts()
        success_labels = {0: 'Needs Support', 1: 'On Track'}
        
        col21, col22 = st.columns(2)
        with col21:
            st.metric("On Track Students", f"{success_counts.get(1, 0)}", 
                     f"{success_counts.get(1, 0)/len(df)*100:.1f}%")
        with col22:
            st.metric("Needs Support", f"{success_counts.get(0, 0)}", 
                     f"{success_counts.get(0, 0)/len(df)*100:.1f}%")
        
        # Simple pie chart using bar chart
        st.bar_chart(success_counts.rename(success_labels))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fixed correlation matrix without matplotlib
    st.markdown('<div class="tab-light">', unsafe_allow_html=True)
    st.subheader("Feature Correlations")
    correlation_matrix = df[['ability', 'engagement', 'difficulty', 'prior_attempts', 
                           'session_count', 'study_duration', 'success']].corr()
    
    # Display correlation matrix with custom styling
    def style_correlation(val):
        if abs(val) > 0.7:
            color = 'background-color: #4258c7; color: #155724;'  # Light green
        elif abs(val) > 0.3:
            color = 'background-color:#9c4f6b; color: #856404;'  # Light yellow
        else:
            color = 'background-color: #807ceb; color: #721c24;'  # Light red
        return color
    
    styled_corr = correlation_matrix.style.applymap(style_correlation)
    st.dataframe(styled_corr)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-light">', unsafe_allow_html=True)
    st.header("💡 Personalized Recommendations")
    
    if st.session_state.recommendations:
        st.subheader("🎯 Recommended Actions")
        for i, recommendation in enumerate(st.session_state.recommendations, 1):
            st.markdown(f"""
            <div class="feature-card">
                <h4>Step {i}: {recommendation}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Learning resources based on learning style
        st.subheader("📚 Learning Resources")
        resources = {
            "Visual": ["📊 Interactive diagrams", "🎥 Video explanations", "📈 Graph-based learning"],
            "Auditory": ["🎧 Podcast lessons", "🔊 Audio explanations", "👥 Group discussions"],
            "Kinesthetic": ["🛠️ Hands-on activities", "🎯 Interactive simulations", "✍️ Practice exercises"],
            "Reading/Writing": ["📖 Detailed textbooks", "📝 Writing assignments", "🔍 Research projects"]
        }
        
        selected_resources = resources.get(learning_style, [])
        for resource in selected_resources:
            st.markdown(f"""
            <div class="feature-card">
                <strong>{learning_style} Learner:</strong> {resource}
            </div>
            """, unsafe_allow_html=True)
            
        # Study tips based on metrics
        st.subheader("🎓 Study Tips")
        tips = []
        if ability < 0.4:
            tips.append("💡 Focus on building foundational knowledge with step-by-step tutorials")
        if engagement < 0.5:
            tips.append("💡 Try gamified learning apps to increase engagement")
        if difficulty > 0.7:
            tips.append("💡 Break down complex topics into smaller, manageable chunks")
        
        for tip in tips:
            st.markdown(f"""
            <div class="feature-card">
                {tip}
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("👆 Click 'Predict Learning Success' to get personalized recommendations")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-light">', unsafe_allow_html=True)
    st.header("📚 Personalized Learning Path")
    
    # Generate dynamic learning path based on inputs
    if st.button("Generate Learning Path", key="generate_path"):
        st.subheader("🎯 Your 4-Week Learning Plan")
        
        weeks = [
            {"week": 1, "focus": "Foundation Building", "activities": ["Basic concepts review", "Formative assessment", "Guided practice"]},
            {"week": 2, "focus": "Skill Development", "activities": ["Applied exercises", "Group learning", "Progress check"]},
            {"week": 3, "focus": "Advanced Application", "activities": ["Complex problems", "Real-world projects", "Peer teaching"]},
            {"week": 4, "focus": "Mastery & Assessment", "activities": ["Final assessment", "Knowledge demonstration", "Next steps planning"]}
        ]
        
        for week in weeks:
            with st.expander(f"📅 Week {week['week']}: {week['focus']}"):
                for activity in week['activities']:
                    st.markdown(f"""
                    <div class="feature-card">
                        ✅ {activity}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Progress tracker
        st.subheader("📊 Progress Tracker")
        progress = st.slider("Current Progress", 0, 100, 25)
        st.progress(progress)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Student History
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-light">', unsafe_allow_html=True)
st.sidebar.subheader("📋 Prediction History")
if st.session_state.student_history:
    for i, record in enumerate(st.session_state.student_history[-3:]):  # Show last 3
        status = "✅ Success" if record['prediction'] == 1 else "❌ Needs Support"
        bg_color = "#d4edda" if record['prediction'] == 1 else "#f8d7da"
        st.sidebar.markdown(f"""
        <div style="background-color: {bg_color}; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid #dee2e6;">
            <strong>{i+1}. {status}</strong><br>
            <small>Confidence: {record['confidence']*100:.1f}%</small>
        </div>
        """, unsafe_allow_html=True)
else:
    st.sidebar.info("No predictions yet")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Add some fun elements
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-light">', unsafe_allow_html=True)
st.sidebar.markdown("### 🏆 Achievement Badges")
if ability > 0.8:
    st.sidebar.markdown('<div style="background-color: #d4edda; padding: 0.5rem; border-radius: 8px; margin: 0.3rem 0; border: 1px solid #c3e6cb;">🌟 High Ability Star</div>', unsafe_allow_html=True)
if engagement > 0.7:
    st.sidebar.markdown('<div style="background-color: #d1ecf1; padding: 0.5rem; border-radius: 8px; margin: 0.3rem 0; border: 1px solid #bee5eb;">🔥 Engagement Champion</div>', unsafe_allow_html=True)
if session_count >= 5:
    st.sidebar.markdown('<div style="background-color: #dbad42; padding: 0.5rem; border-radius: 8px; margin: 0.3rem 0; border: 1px solid #ffeaa7;">📚 Consistent Learner</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### 🎓 AI Tutor Pro v2.0")

st.markdown("*Personalized learning powered by machine intelligence*")
