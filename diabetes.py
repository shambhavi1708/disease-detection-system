# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# def diabetes_app():
#     df = pd.read_csv(r'C:\Users\KIIT\OneDrive\Desktop\PYTHON\CODES\diabetes.csv')
#     x = df.drop(['Outcome'], axis=1)
#     y = df.iloc[:, -1]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#     rf = RandomForestClassifier()
#     rf.fit(x_train, y_train)

#     st.title('Diabetes Checkup')
#     st.sidebar.header('Patient Data')

#     def user_report():
#         pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=3, step=1)
#         glucose = st.sidebar.number_input('Glucose (mg/dL)', min_value=0, max_value=600, value=120, step=1)
#         if glucose > 400:
#             st.sidebar.warning("Glucose values above 400 mg/dL indicate a medical emergency")
    
#         bp = st.sidebar.number_input('Blood Pressure (mmHg)', min_value=0, max_value=250, value=70, step=1)
#         if bp > 180:
#             st.sidebar.warning("Blood pressure above 180 mmHg indicates hypertensive crisis")
    
#         skinthickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
#         insulin = st.sidebar.number_input('Insulin (μU/mL)', min_value=0, max_value=1200, value=79, step=1)
    
#         bmi = st.sidebar.number_input('BMI (kg/m²)', min_value=0, max_value=80, value=20, step=1)
#         if bmi > 50:
#             st.sidebar.warning("BMI values above 50 indicate extreme obesity with severe health risks")
    
#         dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.47, step=0.01)
#         age = st.sidebar.number_input('Age (years)', min_value=0, max_value=100, value=33, step=1)
    
#         user_report = {
#           'Pregnancies': pregnancies,
#           'Glucose': glucose,
#           'BloodPressure': bp,
#           'SkinThickness': skinthickness,
#           'Insulin': insulin,
#           'BMI': bmi,
#           'DiabetesPedigreeFunction': dpf,
#           'Age': age
#         }
#         return pd.DataFrame(user_report, index=[0])

#     user_data = user_report()

#     st.subheader('Patient Data')
#     st.write(user_data)

#     st.subheader('Patient Data Visualization')
#     user_data_transposed = user_data.T
#     st.bar_chart(user_data_transposed)

#     normal_ranges = {
#       'Glucose': "Normal: 70-99 mg/dL\n\nPrediabetes: 100-125 mg/dL\n\nDiabetes: ≥126 mg/dL\n\nSevere: >400 mg/dL",
#       'BloodPressure': "Normal: <120 mmHg\n\nElevated: 120-129 mmHg\n\nStage 1: 130-139 mmHg\n\nStage 2: ≥140 mmHg\n\nCrisis: >180 mmHg",
#       'SkinThickness': "Males: 8-14 mm\n\nFemales: 11-22 mm",
#       'Insulin': "Normal: 3-25 μU/mL\n\nInsulin Resistance: >30 μU/mL\n\nExtreme: >300 μU/mL",
#       'BMI': "Normal: 18.5-24.9 kg/m²\n\nOverweight: 25-29.9 kg/m²\n\nObesity: 30-39.9 kg/m²\n\nSevere: ≥40 kg/m²",
#       'DiabetesPedigreeFunction': "Low Risk: <0.5\n\nModerate Risk: 0.5-1.0\n\nHigh Risk: >1.0",
#       'Age': "Risk increases after age 45"
#     }

#     def clinical_assessment(data):
#        assessment = {}
    
#        glucose_value = data['Glucose'][0]
#        glucose_assessment = []
#        if glucose_value < 100:
#          glucose_assessment.append("Normal: 70-99 mg/dL")
#        if 100 <= glucose_value < 126:
#          glucose_assessment.append("Prediabetes: 100-125 mg/dL")
#        if glucose_value >= 126:
#          glucose_assessment.append("Diabetes: ≥126 mg/dL")
#        if glucose_value > 400:
#          glucose_assessment.append("Severe hyperglycemia - Medical emergency")
#        assessment['Glucose'] = "\n".join(glucose_assessment)
    
#        bp_value = data['BloodPressure'][0]
#        bp_assessment = []
#        if bp_value < 120:
#           bp_assessment.append("Normal")
#        if 120 <= bp_value < 130:
#           bp_assessment.append("Elevated")
#        if 130 <= bp_value < 140:
#           bp_assessment.append("Hypertension Stage 1")
#        if 140 <= bp_value < 180:
#           bp_assessment.append("Hypertension Stage 2")
#        if bp_value >= 180:
#           bp_assessment.append("Hypertensive Crisis - Medical emergency")
#        assessment['BloodPressure'] = "\n".join(bp_assessment)
    
#        bmi_value = data['BMI'][0]
#        bmi_assessment = []
#        if bmi_value < 18.5:
#           bmi_assessment.append("Underweight")
#        if 18.5 <= bmi_value < 25:
#           bmi_assessment.append("Normal weight")
#        if 25 <= bmi_value < 30:
#           bmi_assessment.append("Overweight")
#        if 30 <= bmi_value < 35:
#           bmi_assessment.append("Obesity Class I")
#        if 35 <= bmi_value < 40:
#           bmi_assessment.append("Obesity Class II")
#        if 40 <= bmi_value < 50:
#           bmi_assessment.append("Obesity Class III (Severe)")
#        if bmi_value >= 50:
#           bmi_assessment.append("Super obesity - Extreme health risk")
#        assessment['BMI'] = "\n".join(bmi_assessment)
    
#        return assessment

#     st.subheader('Clinical Assessment')
#     assessment = clinical_assessment(user_data)
#     for param, status in assessment.items():
#        st.write(f"**{param}**:\n{status}")

#     user_result = rf.predict(user_data)

#     st.subheader('Report: ')
#     output = 'Diabetic' if user_result[0] == 1 else 'Not Diabetic'
#     st.title(output)

#     st.subheader('Accuracy: ')
#     st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")

#     st.markdown("---")
#     st.markdown("## Clinical Reference Ranges")
#     for param, range_value in normal_ranges.items():
#       st.markdown(f"#### {param}")
#       st.write(range_value)

#     st.markdown("---")
#     st.warning("**Medical Disclaimer**: This tool provides an estimate based on statistical models and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.")

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def diabetes_app():
    df = pd.read_csv(r'C:\Users\KIIT\OneDrive\Desktop\PYTHON\CODES\diabetes.csv')
    x = df.drop(['Outcome'], axis=1)
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=0)
    svm_model.fit(x_train, y_train)

    log_reg = LogisticRegression(max_iter=500, random_state=0)
    log_reg.fit(x_train, y_train)

    st.title('Diabetes Checkup')
    st.sidebar.header('Patient Data')

    def user_report():
        pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=3, step=1)
        glucose = st.sidebar.number_input('Glucose (mg/dL)', min_value=0, max_value=600, value=120, step=1)
        if glucose > 400:
            st.sidebar.warning("Glucose values above 400 mg/dL indicate a medical emergency")
        bp = st.sidebar.number_input('Blood Pressure (mmHg)', min_value=0, max_value=250, value=70, step=1)
        if bp > 180:
            st.sidebar.warning("Blood pressure above 180 mmHg indicates hypertensive crisis")
        skinthickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
        insulin = st.sidebar.number_input('Insulin (μU/mL)', min_value=0, max_value=1200, value=79, step=1)
        bmi = st.sidebar.number_input('BMI (kg/m²)', min_value=0, max_value=80, value=20, step=1)
        if bmi > 50:
            st.sidebar.warning("BMI values above 50 indicate extreme obesity with severe health risks")
        dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.47, step=0.01)
        age = st.sidebar.number_input('Age (years)', min_value=0, max_value=100, value=33, step=1)
        user_report = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skinthickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        return pd.DataFrame(user_report, index=[0])

    user_data = user_report()

    st.subheader('Patient Data')
    st.write(user_data)

    st.subheader('Patient Data Visualization')
    user_data_transposed = user_data.T
    st.bar_chart(user_data_transposed)

    normal_ranges = {
        'Glucose': "Normal: 70-99 mg/dL\n\nPrediabetes: 100-125 mg/dL\n\nDiabetes: ≥126 mg/dL\n\nSevere: >400 mg/dL",
        'BloodPressure': "Normal: <120 mmHg\n\nElevated: 120-129 mmHg\n\nStage 1: 130-139 mmHg\n\nStage 2: ≥140 mmHg\n\nCrisis: >180 mmHg",
        'SkinThickness': "Males: 8-14 mm\n\nFemales: 11-22 mm",
        'Insulin': "Normal: 3-25 μU/mL\n\nInsulin Resistance: >30 μU/mL\n\nExtreme: >300 μU/mL",
        'BMI': "Normal: 18.5-24.9 kg/m²\n\nOverweight: 25-29.9 kg/m²\n\nObesity: 30-39.9 kg/m²\n\nSevere: ≥40 kg/m²",
        'DiabetesPedigreeFunction': "Low Risk: <0.5\n\nModerate Risk: 0.5-1.0\n\nHigh Risk: >1.0",
        'Age': "Risk increases after age 45"
    }

    def clinical_assessment(data):
        assessment = {}
        glucose_value = data['Glucose'][0]
        glucose_assessment = []
        if glucose_value < 100:
            glucose_assessment.append("Normal: 70-99 mg/dL")
        if 100 <= glucose_value < 126:
            glucose_assessment.append("Prediabetes: 100-125 mg/dL")
        if glucose_value >= 126:
            glucose_assessment.append("Diabetes: ≥126 mg/dL")
        if glucose_value > 400:
            glucose_assessment.append("Severe hyperglycemia - Medical emergency")
        assessment['Glucose'] = "\n".join(glucose_assessment)
        bp_value = data['BloodPressure'][0]
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
        assessment['BloodPressure'] = "\n".join(bp_assessment)
        bmi_value = data['BMI'][0]
        bmi_assessment = []
        if bmi_value < 18.5:
            bmi_assessment.append("Underweight")
        if 18.5 <= bmi_value < 25:
            bmi_assessment.append("Normal weight")
        if 25 <= bmi_value < 30:
            bmi_assessment.append("Overweight")
        if 30 <= bmi_value < 35:
            bmi_assessment.append("Obesity Class I")
        if 35 <= bmi_value < 40:
            bmi_assessment.append("Obesity Class II")
        if 40 <= bmi_value < 50:
            bmi_assessment.append("Obesity Class III (Severe)")
        if bmi_value >= 50:
            bmi_assessment.append("Super obesity - Extreme health risk")
        assessment['BMI'] = "\n".join(bmi_assessment)
        return assessment

    st.subheader('Clinical Assessment')
    assessment = clinical_assessment(user_data)
    for param, status in assessment.items():
        st.write(f"**{param}**:\n{status}")

    user_result_rf = rf.predict(user_data)
    user_result_svm = svm_model.predict(user_data)
    user_result_log = log_reg.predict(user_data)

    st.subheader('Prediction Results:')
    st.write(f"Random Forest: {'Diabetic' if user_result_rf[0] == 1 else 'Not Diabetic'}")
    st.write(f"SVM: {'Diabetic' if user_result_svm[0] == 1 else 'Not Diabetic'}")
    st.write(f"Logistic Regression: {'Diabetic' if user_result_log[0] == 1 else 'Not Diabetic'}")

    train_accuracy_rf = accuracy_score(y_train, rf.predict(x_train)) * 100
    test_accuracy_rf = accuracy_score(y_test, rf.predict(x_test)) * 100
    train_accuracy_svm = accuracy_score(y_train, svm_model.predict(x_train)) * 100
    test_accuracy_svm = accuracy_score(y_test, svm_model.predict(x_test)) * 100
    train_accuracy_log = accuracy_score(y_train, log_reg.predict(x_train)) * 100
    test_accuracy_log = accuracy_score(y_test, log_reg.predict(x_test)) * 100

    st.subheader('Model Accuracy:')
    st.write(f"Random Forest - Training Accuracy: {train_accuracy_rf:.2f}%, Test Accuracy: {test_accuracy_rf:.2f}%")
    st.write(f"SVM - Training Accuracy: {train_accuracy_svm:.2f}%, Test Accuracy: {test_accuracy_svm:.2f}%")
    st.write(f"Logistic Regression - Training Accuracy: {train_accuracy_log:.2f}%, Test Accuracy: {test_accuracy_log:.2f}%")

    st.markdown("---")
    st.markdown("## Clinical Reference Ranges")
    for param, range_value in normal_ranges.items():
        st.markdown(f"#### {param}")
        st.write(range_value)

    st.markdown("---")
    st.warning("**Medical Disclaimer**: This tool provides an estimate based on statistical models and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.")
