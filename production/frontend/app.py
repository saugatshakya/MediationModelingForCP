"""
Interactive Streamlit Frontend for Exam Score Prediction

Features:
- Drink selector with caffeine calculation
- Interactive timetable/activity planner (24-hour day)
- Feature inputs for all relevant variables
- Exam score prediction
- Optimization for target scores
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib
from scipy.optimize import minimize
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pipeline import ExamScorePipeline


# Drink data with caffeine content
DRINKS_DATA = {
    "Espresso": {"serving_ml": 30, "caffeine_mg": 63},
    "Americano": {"serving_ml": 240, "caffeine_mg": 95},
    "Cappuccino": {"serving_ml": 240, "caffeine_mg": 75},
    "Latte": {"serving_ml": 240, "caffeine_mg": 75},
    "Drip/Filter Coffee": {"serving_ml": 240, "caffeine_mg": 130},
    "Instant Coffee": {"serving_ml": 240, "caffeine_mg": 100},
    "Thai Iced Coffee": {"serving_ml": 240, "caffeine_mg": 70},
    "Black Tea": {"serving_ml": 240, "caffeine_mg": 55},
    "Green Tea": {"serving_ml": 240, "caffeine_mg": 32},
    "Oolong Tea": {"serving_ml": 240, "caffeine_mg": 40},
    "White Tea": {"serving_ml": 240, "caffeine_mg": 22},
    "Matcha": {"serving_ml": 240, "caffeine_mg": 70},
    "Thai Iced Tea": {"serving_ml": 240, "caffeine_mg": 45},
    "Red Bull": {"serving_ml": 250, "caffeine_mg": 80},
    "Monster": {"serving_ml": 500, "caffeine_mg": 160},
    "M-150": {"serving_ml": 250, "caffeine_mg": 80},
    "Carabao": {"serving_ml": 330, "caffeine_mg": 106},
    "5-hour Energy": {"serving_ml": 57, "caffeine_mg": 200},
}


def load_model(model_path: str = "artifacts/model_linear_model.joblib"):
    """Load the trained model."""
    try:
        pipeline = ExamScorePipeline.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"Model not found at {model_path}. Please run experiments first.")
        return None


def calculate_caffeine_intake(drinks_selection):
    """Calculate total caffeine from selected drinks."""
    total_caffeine = 0
    for drink, count in drinks_selection.items():
        if count > 0:
            total_caffeine += DRINKS_DATA[drink]["caffeine_mg"] * count
    return total_caffeine


def caffeine_to_scale(caffeine_mg):
    """Convert caffeine mg to 1-10 scale (50-500mg → 1-10)."""
    return np.clip((caffeine_mg - 50) / 50, 1, 10)


def create_timetable_ui():
    """Create interactive timetable for 24-hour activity planning."""
    st.subheader("📅 Daily Activity Timetable")
    st.write("Allocate your 24 hours across activities (must sum to 24):")
    
    activities = {
        "Sleep": {"icon": "😴", "default": 7.0},
        "Study": {"icon": "📚", "default": 5.0},
        "Social Media": {"icon": "📱", "default": 2.0},
        "Netflix/Entertainment": {"icon": "📺", "default": 2.0},
        "Exercise": {"icon": "🏃", "default": 1.0},
        "Part-time Job": {"icon": "💼", "default": 0.0},
        "Other": {"icon": "🎯", "default": 7.0},
    }
    
    cols = st.columns(2)
    activity_hours = {}
    
    for i, (activity, info) in enumerate(activities.items()):
        with cols[i % 2]:
            hours = st.slider(
                f"{info['icon']} {activity}",
                min_value=0.0,
                max_value=24.0,
                value=info['default'],
                step=0.5,
                key=f"timetable_{activity}"
            )
            activity_hours[activity] = hours
    
    total_hours = sum(activity_hours.values())
    
    # Show total and validation
    if total_hours != 24.0:
        st.warning(f"⚠️ Total hours: {total_hours:.1f}/24. Please adjust to equal 24 hours.")
    else:
        st.success(f"✅ Total hours: {total_hours:.1f}/24")
    
    # Visualize timetable
    fig = go.Figure(data=[go.Pie(
        labels=list(activity_hours.keys()),
        values=list(activity_hours.values()),
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    fig.update_layout(
        title="Daily Time Distribution",
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return activity_hours


def create_feature_inputs(activity_hours):
    """Create input widgets for all features."""
    st.subheader("📊 Additional Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Derived from timetable
        sleep_hours = activity_hours.get("Sleep", 7.0)
        study_hours = activity_hours.get("Study", 5.0)
        social_hours = activity_hours.get("Social Media", 2.0)
        netflix_hours = activity_hours.get("Netflix/Entertainment", 2.0)
        
        st.write(f"**Sleep Hours:** {sleep_hours}")
        st.write(f"**Study Hours:** {study_hours}")
        st.write(f"**Social Media:** {social_hours}")
        st.write(f"**Netflix:** {netflix_hours}")
        
        # Calculate exercise frequency from hours (0-10 scale)
        exercise_hours = activity_hours.get("Exercise", 1.0)
        exercise_frequency = min(10, int(exercise_hours * 2))  # 5 hours = 10/10
        st.write(f"**Exercise Frequency:** {exercise_frequency}/10")
        
        # Additional inputs
        sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7, key="sleep_quality")
        attendance = st.slider("Attendance %", 0, 100, 90, key="attendance")
        
    with col2:
        mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 7, key="mental_health")
        
        diet_quality = st.selectbox(
            "Diet Quality",
            options=["Poor", "Fair", "Good"],
            index=2,
            key="diet_quality"
        )
        
        internet_quality = st.selectbox(
            "Internet Quality",
            options=["Poor", "Average", "Good"],
            index=2,
            key="internet_quality"
        )
        
        part_time_job = "Yes" if activity_hours.get("Part-time Job", 0) > 0 else "No"
        st.write(f"**Part-time Job:** {part_time_job}")
        
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0, key="gender")
        age = st.number_input("Age", min_value=15, max_value=30, value=20, key="age")
    
    # Map categorical to numeric for model
    diet_map = {'Poor': -1, 'Fair': 0, 'Good': 1}
    internet_map = {'Poor': -1, 'Average': 0, 'Good': 1}
    
    features = {
        'sleep_hours': sleep_hours,
        'study_hours_per_day': study_hours,
        'social_media_hours': social_hours,
        'netflix_hours': netflix_hours,
        'exercise_frequency': exercise_frequency,
        'sleep_quality': sleep_quality,
        'attendance_percentage': attendance,
        'mental_health_rating': mental_health,
        'diet_quality': diet_quality,
        'diet_score': diet_map[diet_quality],
        'internet_quality': internet_quality,
        'internet_score': internet_map[internet_quality],
        'part_time_job': part_time_job,
        'part_job_score': 1 if part_time_job == "Yes" else 0,
        'gender': gender,
        'age': age
    }
    
    return features


def calculate_stress_and_focus(features, caffeine_intake_scale):
    """Calculate stress and focus proxies using the formulas from main.ipynb."""
    # Normalize features
    Sd = features['sleep_hours'] / 12.0  # Assuming max 12 hours
    Sq = features['sleep_quality'] / 10.0
    
    # Caffeine effects
    C_mg = 50 + 50 * caffeine_intake_scale
    fC = np.exp(-((C_mg - 200) ** 2) / (2 * (100 ** 2)))
    gC = max(0, (C_mg - 300) / 200)
    
    # Normalize inputs
    study = features['study_hours_per_day'] / 10.0
    social = features['social_media_hours'] / 10.0
    netflix = features['netflix_hours'] / 10.0
    exercise = features['exercise_frequency'] / 10.0
    mental = features['mental_health_rating'] / 10.0
    
    # Stress calculation
    stress = (
        gC - Sq +
        0.4 * social +
        0.3 * netflix +
        0.2 * study +
        0.3 * features['part_job_score'] -
        0.3 * exercise -
        0.4 * mental -
        0.2 * features['diet_score'] -
        0.2 * features['internet_score']
    )
    
    # Focus calculation
    focus = (
        Sd + Sq - stress + fC +
        0.4 * exercise +
        0.4 * mental +
        0.3 * (features['attendance_percentage'] / 100) +
        0.2 * features['diet_score'] +
        0.2 * features['internet_score'] -
        0.3 * social -
        0.2 * netflix -
        0.2 * features['part_job_score']
    )
    
    # Rescale to 1-10
    stress_scaled = np.clip((stress + 2) * 2.5, 1, 10)  # Rough rescaling
    focus_scaled = np.clip((focus + 1) * 1.8, 1, 10)
    
    return stress_scaled, focus_scaled


def predict_exam_score(features, caffeine_intake_scale, stress, focus, model=None):
    """Predict exam score using the trained model or fallback to formula."""
    
    if model is not None:
        try:
            # Prepare features DataFrame for model prediction
            # Only include features that were used during training (from main.ipynb)
            feature_dict = {
                'study_hours_per_day': features['study_hours_per_day'],
                'sleep_hours': features['sleep_hours'],
                'social_media_hours': features['social_media_hours'],
                'netflix_hours': features['netflix_hours'],
                'exercise_frequency': features['exercise_frequency'],
                'sleep_quality': features['sleep_quality'],
                'attendance_percentage': features['attendance_percentage'],
                'caffeine_intake': caffeine_intake_scale,
                'stress_proxy': stress,
                'focus_proxy': focus,
            }
            
            # Create DataFrame with single row
            X = pd.DataFrame([feature_dict])
            
            # Predict using trained model
            exam_score = model.predict(X)[0]
            
            return np.clip(exam_score, 0, 100)
            
        except Exception as e:
            st.warning(f"⚠️ Model prediction failed: {str(e)}. Using fallback formula.")
    
    # Fallback: simplified prediction based on correlations from main.ipynb
    score = (
        0.25 * features['study_hours_per_day'] * 10 +  # ~25% importance
        0.20 * focus * 10 +  # ~20% importance
        0.15 * (10 - stress) * 10 +  # ~15% importance (inverse)
        0.12 * features['attendance_percentage'] +  # ~12% importance
        0.10 * features['mental_health_rating'] * 10 +  # ~10% importance
        0.08 * features['sleep_quality'] * 10 +  # ~8% importance
        0.05 * features['sleep_hours'] * 10 +  # ~5% importance
        0.05 * (10 - caffeine_intake_scale) * 5  # ~5% (inverse for high caffeine)
    )
    
    # Add noise and clip to 0-100
    exam_score = np.clip(score + np.random.normal(0, 2), 0, 100)
    
    return exam_score


def optimize_for_target_score(current_features, current_caffeine, target_score, model=None):
    """Optimize activity allocation to achieve target exam score."""
    st.subheader("🎯 Optimization for Target Score")
    
    # Simple optimization: suggest adjustments
    current_score_estimate = predict_exam_score(
        current_features,
        caffeine_to_scale(current_caffeine),
        *calculate_stress_and_focus(current_features, caffeine_to_scale(current_caffeine)),
        model
    )
    
    gap = target_score - current_score_estimate
    
    if abs(gap) < 2:
        st.success(f"✅ You're already close to your target! (Current: {current_score_estimate:.1f})")
        return
    
    recommendations = []
    
    if gap > 0:  # Need to increase score
        st.info(f"📈 To reach {target_score}, you need to improve by {gap:.1f} points.")
        
        # Suggest increasing study hours
        study_increase = min(gap / 5, 3.0)  # Each study hour ~5 points
        recommendations.append(f"📚 Increase study time by {study_increase:.1f} hours")
        
        # Suggest reducing distractions
        social_decrease = min(gap / 10, current_features['social_media_hours'] - 0.5)
        if social_decrease > 0:
            recommendations.append(f"📱 Reduce social media by {social_decrease:.1f} hours")
        
        # Suggest improving sleep
        if current_features['sleep_quality'] < 8:
            recommendations.append(f"😴 Improve sleep quality (currently {current_features['sleep_quality']}/10)")
        
        # Suggest attendance
        if current_features['attendance_percentage'] < 95:
            recommendations.append(f"🎓 Increase attendance to 95%+ (currently {current_features['attendance_percentage']}%)")
        
    else:  # Can reduce effort
        st.info(f"📉 You're exceeding the target by {abs(gap):.1f} points - you can ease up!")
        recommendations.append(f"😊 You have room to balance your schedule better")
    
    st.write("**Recommendations:**")
    for rec in recommendations:
        st.write(f"- {rec}")


def main():
    st.set_page_config(
        page_title="Exam Score Predictor",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Exam Score Prediction System")
    st.write("Predict your exam score based on lifestyle, study habits, and caffeine intake!")
    
    # Load model once and cache it
    @st.cache_resource
    def get_model():
        return load_model()
    
    model = get_model()
    
    # Show model status
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.warning("⚠️ Using fallback prediction formula")
    
    # Sidebar for drink selection
    st.sidebar.header("☕ Caffeine Intake Calculator")
    st.sidebar.write("Select your daily drinks:")
    
    drinks_selection = {}
    total_caffeine = 0
    
    for drink, info in DRINKS_DATA.items():
        count = st.sidebar.number_input(
            f"{drink} ({info['caffeine_mg']}mg / {info['serving_ml']}ml)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key=f"drink_{drink}"
        )
        drinks_selection[drink] = count
        total_caffeine += info['caffeine_mg'] * count
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Caffeine", f"{total_caffeine} mg")
    caffeine_scale = caffeine_to_scale(total_caffeine)
    st.sidebar.metric("Caffeine Scale (1-10)", f"{caffeine_scale:.1f}")
    
    # Health recommendations based on caffeine
    if total_caffeine > 400:
        st.sidebar.warning("⚠️ High caffeine intake! Consider reducing.")
    elif total_caffeine > 200:
        st.sidebar.info("ℹ️ Moderate caffeine intake.")
    else:
        st.sidebar.success("✅ Low to moderate caffeine.")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📅 Timetable & Prediction", "🎯 Optimize", "📊 Insights"])
    
    with tab1:
        # Timetable
        activity_hours = create_timetable_ui()
        
        # Only proceed if timetable is valid
        if sum(activity_hours.values()) == 24.0:
            # Feature inputs
            features = create_feature_inputs(activity_hours)
            
            # Add caffeine to features
            features['caffeine_intake'] = caffeine_scale
            
            # Calculate stress and focus
            stress, focus = calculate_stress_and_focus(features, caffeine_scale)
            features['stress_proxy'] = stress
            features['focus_proxy'] = focus
            
            # Display calculated metrics
            st.subheader("📈 Calculated Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Caffeine Intake", f"{caffeine_scale:.1f}/10", 
                         delta=f"{total_caffeine} mg")
            with col2:
                st.metric("Stress Level", f"{stress:.1f}/10",
                         delta="Lower is better" if stress > 5 else "Good")
            with col3:
                st.metric("Focus Level", f"{focus:.1f}/10",
                         delta="Higher is better" if focus < 7 else "Good")
            
            # Predict exam score
            if st.button("🎯 Predict Exam Score", type="primary"):
                with st.spinner("Calculating prediction..."):
                    exam_score = predict_exam_score(features, caffeine_scale, stress, focus, model)
                    
                    st.success(f"### Predicted Exam Score: {exam_score:.1f}/100")
                    
                    # Score interpretation
                    if exam_score >= 80:
                        st.balloons()
                        st.success("🌟 Excellent! Keep up the great work!")
                    elif exam_score >= 70:
                        st.info("👍 Good performance! Room for improvement.")
                    elif exam_score >= 60:
                        st.warning("⚠️ Passing, but consider optimizing your schedule.")
                    else:
                        st.error("❌ Below average. Check optimization suggestions!")
                    
                    # Store in session state for optimization
                    st.session_state['current_score'] = exam_score
                    st.session_state['current_features'] = features
                    st.session_state['current_caffeine'] = total_caffeine
    
    with tab2:
        st.header("🎯 Score Optimization")
        
        if 'current_score' in st.session_state:
            st.write(f"**Current Predicted Score:** {st.session_state['current_score']:.1f}/100")
            
            target_score = st.slider(
                "Target Exam Score",
                min_value=0,
                max_value=100,
                value=int(st.session_state['current_score']) + 10,
                step=1
            )
            
            if st.button("Generate Recommendations"):
                optimize_for_target_score(
                    st.session_state['current_features'],
                    st.session_state['current_caffeine'],
                    target_score,
                    model
                )
        else:
            st.info("👈 Please make a prediction first in the 'Timetable & Prediction' tab.")
    
    with tab3:
        st.header("📊 Insights & Tips")
        
        st.subheader("Key Factors for Academic Success")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Most Important Factors:**
            1. 📚 **Study Hours** (25% impact)
            2. 🎯 **Focus Level** (20% impact)
            3. 😌 **Low Stress** (15% impact)
            4. 🎓 **Attendance** (12% impact)
            5. 🧠 **Mental Health** (10% impact)
            """)
        
        with col2:
            st.markdown("""
            **Caffeine Guidelines:**
            - ☕ Optimal: 100-200mg/day
            - ⚠️ Moderate: 200-400mg/day
            - 🚫 Excessive: >400mg/day
            
            **Sleep Recommendations:**
            - 😴 7-9 hours per night
            - 🌙 Consistent sleep schedule
            - ✨ Quality matters more than quantity
            """)
        
        # Show sample data visualization
        st.subheader("Impact of Study Hours on Exam Scores")
        
        # Generate sample data for visualization
        study_range = np.linspace(0, 10, 50)
        scores = []
        for study in study_range:
            sample_features = {
                'study_hours_per_day': study,
                'sleep_hours': 7,
                'social_media_hours': 2,
                'netflix_hours': 2,
                'exercise_frequency': 5,
                'sleep_quality': 7,
                'attendance_percentage': 85,
                'mental_health_rating': 7,
                'diet_score': 1,
                'internet_score': 1,
                'part_job_score': 0
            }
            s, f = calculate_stress_and_focus(sample_features, 5)
            score = predict_exam_score(sample_features, 5, s, f, model)
            scores.append(score)
        
        fig = px.line(
            x=study_range,
            y=scores,
            labels={'x': 'Study Hours per Day', 'y': 'Predicted Exam Score'},
            title='Study Hours vs Exam Score'
        )
        fig.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
