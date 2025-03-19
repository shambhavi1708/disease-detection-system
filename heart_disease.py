import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def heart_disease_app():
    

    df = pd.read_csv(r"C:\Users\KIIT\OneDrive\Desktop\FILES\GIT\PROJECTS\heart_disease_data.csv")

    df = pd.DataFrame({
         'age': np.random.randint(29, 80, 100),
         'sex': np.random.randint(0, 2, 100),
         'cp': np.random.randint(0, 4, 100),
         'trestbps': np.random.randint(94, 200, 100),
         'chol': np.random.randint(126, 564, 100),
         'fbs': np.random.randint(0, 2, 100),
         'restecg': np.random.randint(0, 3, 100),
         'thalach': np.random.randint(71, 202, 100),
         'exang': np.random.randint(0, 2, 100),
         'oldpeak': np.random.uniform(0, 6.2, 100),
         'slope': np.random.randint(0, 3, 100),
         'ca': np.random.randint(0, 5, 100),
         'thal': np.random.randint(0, 4, 100),
         'target': np.random.randint(0, 2, 100)
    })

    x = df.drop(['target'], axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    st.title('Cardiovascular Disease Checkup')
    st.sidebar.header('Patient Data')

    normal_ranges = {
       'Age': "Age in years. Risk increases with age, particularly after 45 for men and 55 for women.",
       'Trestbps': "Normal: <120 mmHg\n\nElevated: 120-129 mmHg\n\nStage 1: 130-139 mmHg\n\nStage 2: ≥140 mmHg\n\nCrisis: >180 mmHg",
       'Chol': "Normal: <200 mg/dL\n\nBorderline: 200-239 mg/dL\n\nHigh: ≥240 mg/dL\n\nVery High: ≥300 mg/dL",
       'Thalach': "Maximum heart rate varies by age.\n\nEstimated max = 220 - age.",
       'Oldpeak': "Normal: <0.5 mm\n\nBorderline: 0.5-1.0 mm\n\nAbnormal: >1.0 mm\n\nSeverely Abnormal: >2.0 mm",
       'Ca': "Normal: 0 vessels\n\nMild-Moderate: 1-2 vessels\n\nSevere: 3-4 vessels"
    }

    def user_report():
       age = st.sidebar.number_input('Age (years)', min_value=20, max_value=100, value=45, step=1)
    
       sex = st.sidebar.radio('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    
       cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3], 
                              format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
    
       trestbps = st.sidebar.number_input('Resting Blood Pressure (mmHg)', min_value=90, max_value=240, value=120, step=1)
       if trestbps > 180:
            st.sidebar.warning("Blood pressure above 180 mmHg indicates hypertensive crisis")
    
       chol = st.sidebar.number_input('Cholesterol (mg/dL)', min_value=120, max_value=600, value=200, step=1)
       if chol > 300:
            st.sidebar.warning("Cholesterol above 300 mg/dL indicates very high risk")
    
       fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dL', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    
       restecg = st.sidebar.selectbox('Resting ECG', [0, 1, 2], 
             format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x])
    
       thalach = st.sidebar.number_input('Maximum Heart Rate', min_value=60, max_value=220, value=150, step=1)
    
       exang = st.sidebar.radio('Exercise Induced Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    
       oldpeak = st.sidebar.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
       if oldpeak > 4.0:
             st.sidebar.warning("ST depression above 4.0 mm indicates severe ischemia")
    
       slope = st.sidebar.selectbox('Slope of ST Segment', [0, 1, 2], 
                                format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
    
       ca = st.sidebar.number_input('Number of Major Vessels', min_value=0, max_value=4, value=0, step=1)
    
       thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3], 
                               format_func=lambda x: ['NULL', 'Fixed Defect', 'Normal', 'Reversible Defect'][x])
    
       user_report = {
          'age': age,
          'sex': sex,
          'cp': cp,
          'trestbps': trestbps,
          'chol': chol,
          'fbs': fbs,
          'restecg': restecg,
          'thalach': thalach,
          'exang': exang,
          'oldpeak': oldpeak,
          'slope': slope,
          'ca': ca,
          'thal': thal
        }
       return pd.DataFrame(user_report, index=[0])

    user_data = user_report()

    st.subheader('Patient Data')
    st.write(user_data)

    st.subheader('Patient Data Visualization')
    user_data_transposed = user_data.T
    st.bar_chart(user_data_transposed)

    def clinical_assessment(data):
      assessment = {}
    
      bp_value = data['trestbps'][0]
      bp_assessment = []
      if bp_value < 120:
        bp_assessment.append("Normal")
      if 120 <= bp_value < 130:
        bp_assessment.append("Elevated")
      if 130 <= bp_value < 140:
        bp_assessment.append("Hypertension Stage 1")
      if 140 <= bp_value < 180:
        bp_assessment.append("Hypertension Stage 2")
      if bp_value >= 180:
        bp_assessment.append("Hypertensive Crisis - Medical emergency")
      assessment['Blood Pressure'] = "\n".join(bp_assessment)
    
      chol_value = data['chol'][0]
      chol_assessment = []
      if chol_value < 200:
        chol_assessment.append("Desirable level")
      if 200 <= chol_value < 240:
        chol_assessment.append("Borderline high")
      if 240 <= chol_value < 300:
        chol_assessment.append("High - Increased cardiovascular risk")
      if chol_value >= 300:
        chol_assessment.append("Very high - Significant cardiovascular risk")
      assessment['Cholesterol'] = "\n".join(chol_assessment)
    
      st_value = data['oldpeak'][0]
      st_assessment = []
      if st_value < 0.5:
        st_assessment.append("Normal")
      if 0.5 <= st_value < 1.0:
        st_assessment.append("Borderline")
      if 1.0 <= st_value < 2.0:
        st_assessment.append("Abnormal - Indicative of ischemia")
      if st_value >= 2.0:
        st_assessment.append("Severely abnormal - High likelihood of coronary disease")
      assessment['ST Depression'] = "\n".join(st_assessment)
    
      ca_value = data['ca'][0]
      ca_assessment = []
      if ca_value == 0:
        ca_assessment.append("Normal - No significant vessel disease")
      if 1 <= ca_value <= 2:
        ca_assessment.append("Mild to moderate coronary artery disease")
      if ca_value >= 3:
        ca_assessment.append("Severe coronary artery disease")
      assessment['Major Vessels'] = "\n".join(ca_assessment)
    
      return assessment

    st.subheader('Clinical Assessment')
    assessment = clinical_assessment(user_data)
    for param, status in assessment.items():
       st.write(f"**{param}**:\n{status}")

    user_result = rf.predict(user_data)

    st.subheader('Report: ')
    output = 'Heart Disease Detected' if user_result[0] == 1 else 'No Heart Disease Detected'
    st.title(output)

    st.subheader('Accuracy: ')
    st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")

    st.markdown("---")
    st.markdown("## Clinical Reference Ranges")
    for param, range_value in normal_ranges.items():
       st.markdown(f"#### {param}")
       st.write(range_value)

    st.markdown("---")
    st.warning("**Medical Disclaimer**: This tool provides an estimate based on statistical models and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.")
