# import pandas as pd
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import svm
# from sklearn.metrics import accuracy_score

# def parkinsons_app():
    

#     parkinsons_data = pd.read_csv(r'C:\Users\KIIT\OneDrive\Desktop\FILES\GIT\PROJECTS\parkinsons.csv')

#     X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
#     Y = parkinsons_data['status']
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = svm.SVC(kernel='linear')
#     model.fit(X_train, Y_train)

#     st.title("Parkinson's Disease Checkup")
#     st.sidebar.header('Patient Data')

#     def user_report():
#       features = {}
    
#       st.sidebar.subheader('Voice Characteristics')
#       features['MDVP:Fo(Hz)'] = st.sidebar.number_input('Average Vocal Fundamental Frequency (Hz)', min_value=80.0, max_value=300.0, value=162.568, step=0.001)
#       features['MDVP:Fhi(Hz)'] = st.sidebar.number_input('Maximum Vocal Fundamental Frequency (Hz)', min_value=100.0, max_value=600.0, value=198.346, step=0.001)
#       features['MDVP:Flo(Hz)'] = st.sidebar.number_input('Minimum Vocal Fundamental Frequency (Hz)', min_value=50.0, max_value=200.0, value=77.63, step=0.001)
    
#       st.sidebar.subheader('Frequency Variation Measures')
#       features['MDVP:Jitter(%)'] = st.sidebar.number_input('MDVP Jitter in percentage', min_value=0.0, max_value=0.1, value=0.00502, step=0.00001, format="%.5f")
#       features['MDVP:Jitter(Abs)'] = st.sidebar.number_input('MDVP Absolute Jitter', min_value=0.0, max_value=0.001, value=0.00003, step=0.000001, format="%.6f")
#       features['MDVP:RAP'] = st.sidebar.number_input('MDVP Relative Amplitude Perturbation', min_value=0.0, max_value=0.1, value=0.00280, step=0.00001, format="%.5f")
#       features['MDVP:PPQ'] = st.sidebar.number_input('MDVP Five-point Period Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.00253, step=0.00001, format="%.5f")
#       features['Jitter:DDP'] = st.sidebar.number_input('Average Absolute Difference of Differences', min_value=0.0, max_value=0.1, value=0.00841, step=0.00001, format="%.5f")
    
#       st.sidebar.subheader('Amplitude Variation Measures')
#       features['MDVP:Shimmer'] = st.sidebar.number_input('MDVP Local Shimmer', min_value=0.0, max_value=0.2, value=0.01791, step=0.00001, format="%.5f")
#       features['MDVP:Shimmer(dB)'] = st.sidebar.number_input('MDVP Local Shimmer in dB', min_value=0.0, max_value=2.0, value=0.16800, step=0.00001, format="%.5f")
#       features['Shimmer:APQ3'] = st.sidebar.number_input('Three-point Amplitude Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.00793, step=0.00001, format="%.5f")
#       features['Shimmer:APQ5'] = st.sidebar.number_input('Five-point Amplitude Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.01057, step=0.00001, format="%.5f")
#       features['MDVP:APQ'] = st.sidebar.number_input('MDVP 11-point Amplitude Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.01799, step=0.00001, format="%.5f")
#       features['Shimmer:DDA'] = st.sidebar.number_input('Average Absolute Differences of Amplitude', min_value=0.0, max_value=0.1, value=0.02380, step=0.00001, format="%.5f")
    
#       st.sidebar.subheader('Harmonicity Measures')
#       features['NHR'] = st.sidebar.number_input('Noise-to-Harmonics Ratio', min_value=0.0, max_value=0.5, value=0.01170, step=0.00001, format="%.5f")
#       features['HNR'] = st.sidebar.number_input('Harmonics-to-Noise Ratio', min_value=0.0, max_value=50.0, value=25.6780, step=0.0001)
    
#       st.sidebar.subheader('Nonlinear Measures')
#       features['RPDE'] = st.sidebar.number_input('Recurrence Period Density Entropy', min_value=0.0, max_value=1.0, value=0.427785, step=0.000001)
#       features['DFA'] = st.sidebar.number_input('Detrended Fluctuation Analysis', min_value=0.0, max_value=1.0, value=0.723797, step=0.000001)
#       features['spread1'] = st.sidebar.number_input('Spread1', min_value=-10.0, max_value=10.0, value=-6.635729, step=0.000001)
#       features['spread2'] = st.sidebar.number_input('Spread2', min_value=-10.0, max_value=10.0, value=0.209866, step=0.000001)
#       features['D2'] = st.sidebar.number_input('Correlation Dimension', min_value=0.0, max_value=10.0, value=1.957961, step=0.000001)
#       features['PPE'] = st.sidebar.number_input('Pitch Period Entropy', min_value=0.0, max_value=1.0, value=0.135242, step=0.000001)
    
#       return pd.DataFrame(features, index=[0])

#     user_data = user_report()

#     st.subheader('Patient Data')
#     st.write(user_data)
#     st.subheader('Patient Data Visualization')

#     key_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 'DFA']
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.bar(key_features, user_data[key_features].values[0])
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     st.pyplot(fig)

#     normal_ranges = {
#       'MDVP:Fo(Hz)': "Males: 85-180 Hz\n\nFemales: 165-255 Hz",
#       'MDVP:Jitter(%)': "Normal: <1.040%\n\nParkinson's: Often >1.040%",
#       'MDVP:Shimmer': "Normal: <3.810%\n\nParkinson's: Often >3.810%",
#       'NHR': "Normal: <0.190\n\nParkinson's: Often >0.190",
#       'HNR': "Normal: >21 dB\n\nParkinson's: Often <21 dB",
#       'DFA': "Normal: ~0.5-0.7\n\nParkinson's: Often >0.7",
#       'RPDE': "Normal: Lower values\n\nParkinson's: Higher values",
#       'PPE': "Normal: Lower values\n\nParkinson's: Higher values"
# }

#     def clinical_assessment(data):
#        assessment = {}
    
#        fo_value = data['MDVP:Fo(Hz)'][0]
#        fo_assessment = []
#        if 85 <= fo_value <= 180:
#           fo_assessment.append("Within typical male range")
#        if 165 <= fo_value <= 255:
#           fo_assessment.append("Within typical female range")
#        if fo_value < 85 or fo_value > 255:
#           fo_assessment.append("Outside typical ranges")
#        assessment['MDVP:Fo(Hz)'] = "\n".join(fo_assessment)
    
#        jitter_value = data['MDVP:Jitter(%)'][0]
#        if jitter_value < 0.01040:
#            assessment['MDVP:Jitter(%)'] = "Normal range"
#        else:
#            assessment['MDVP:Jitter(%)'] = "Elevated - may indicate vocal pathology"
    
#        shimmer_value = data['MDVP:Shimmer'][0]
#        if shimmer_value < 0.03810:
#            assessment['MDVP:Shimmer'] = "Normal range"
#        else:
#            assessment['MDVP:Shimmer'] = "Elevated - may indicate vocal pathology"
    
#        nhr_value = data['NHR'][0]
#        if nhr_value < 0.0190:
#            assessment['NHR'] = "Normal range"
#        else:
#            assessment['NHR'] = "Elevated - may indicate vocal pathology"
    
#        hnr_value = data['HNR'][0]
#        if hnr_value > 21:
#            assessment['HNR'] = "Normal range"
#        else:
#            assessment['HNR'] = "Reduced - may indicate vocal pathology"
    
#        return assessment

#     st.subheader('Clinical Assessment')
#     assessment = clinical_assessment(user_data)
#     for param, status in assessment.items():
#       st.write(f"**{param}**:\n{status}")

#     input_data_scaled = scaler.transform(user_data)
#     prediction = model.predict(input_data_scaled)

#     st.subheader('Prediction Result:')
#     output = 'Parkinson\'s Disease Detected' if prediction[0] == 1 else 'No Parkinson\'s Disease Detected'
#     st.title(output)

#     train_accuracy = accuracy_score(Y_train, model.predict(X_train))
#     test_accuracy = accuracy_score(Y_test, model.predict(X_test))

#     st.subheader('Model Accuracy:')
#     st.write(f"Training data accuracy: {train_accuracy * 100:.2f}%")
#     st.write(f"Test data accuracy: {test_accuracy * 100:.2f}%")

#     st.markdown("---")
#     st.markdown("## Clinical Reference Ranges")
#     for param, range_value in normal_ranges.items():
#        st.markdown(f"#### {param}")
#        st.write(range_value)

#     st.markdown("---")
#     st.warning("**Medical Disclaimer**: This tool provides an estimate based on statistical models and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def parkinsons_app():
    parkinsons_data = pd.read_csv(r'C:\Users\KIIT\OneDrive\Desktop\FILES\GIT\PROJECTS\parkinsons.csv')

    X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
    Y = parkinsons_data['status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, Y_train)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, Y_train)

    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, Y_train)

    st.title("Parkinson's Disease Checkup")
    st.sidebar.header('Patient Data')

    def user_report():
        features = {}
        st.sidebar.subheader('Voice Characteristics')
        features['MDVP:Fo(Hz)'] = st.sidebar.number_input('Average Vocal Fundamental Frequency (Hz)', min_value=80.0, max_value=300.0, value=162.568, step=0.001)
        features['MDVP:Fhi(Hz)'] = st.sidebar.number_input('Maximum Vocal Fundamental Frequency (Hz)', min_value=100.0, max_value=600.0, value=198.346, step=0.001)
        features['MDVP:Flo(Hz)'] = st.sidebar.number_input('Minimum Vocal Fundamental Frequency (Hz)', min_value=50.0, max_value=200.0, value=77.63, step=0.001)
        st.sidebar.subheader('Frequency Variation Measures')
        features['MDVP:Jitter(%)'] = st.sidebar.number_input('MDVP Jitter in percentage', min_value=0.0, max_value=0.1, value=0.00502, step=0.00001, format="%.5f")
        features['MDVP:Jitter(Abs)'] = st.sidebar.number_input('MDVP Absolute Jitter', min_value=0.0, max_value=0.001, value=0.00003, step=0.000001, format="%.6f")
        features['MDVP:RAP'] = st.sidebar.number_input('MDVP Relative Amplitude Perturbation', min_value=0.0, max_value=0.1, value=0.00280, step=0.00001, format="%.5f")
        features['MDVP:PPQ'] = st.sidebar.number_input('MDVP Five-point Period Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.00253, step=0.00001, format="%.5f")
        features['Jitter:DDP'] = st.sidebar.number_input('Average Absolute Difference of Differences', min_value=0.0, max_value=0.1, value=0.00841, step=0.00001, format="%.5f")
        st.sidebar.subheader('Amplitude Variation Measures')
        features['MDVP:Shimmer'] = st.sidebar.number_input('MDVP Local Shimmer', min_value=0.0, max_value=0.2, value=0.01791, step=0.00001, format="%.5f")
        features['MDVP:Shimmer(dB)'] = st.sidebar.number_input('MDVP Local Shimmer in dB', min_value=0.0, max_value=2.0, value=0.16800, step=0.00001, format="%.5f")
        features['Shimmer:APQ3'] = st.sidebar.number_input('Three-point Amplitude Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.00793, step=0.00001, format="%.5f")
        features['Shimmer:APQ5'] = st.sidebar.number_input('Five-point Amplitude Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.01057, step=0.00001, format="%.5f")
        features['MDVP:APQ'] = st.sidebar.number_input('MDVP 11-point Amplitude Perturbation Quotient', min_value=0.0, max_value=0.1, value=0.01799, step=0.00001, format="%.5f")
        features['Shimmer:DDA'] = st.sidebar.number_input('Average Absolute Differences of Amplitude', min_value=0.0, max_value=0.1, value=0.02380, step=0.00001, format="%.5f")
        st.sidebar.subheader('Harmonicity Measures')
        features['NHR'] = st.sidebar.number_input('Noise-to-Harmonics Ratio', min_value=0.0, max_value=0.5, value=0.01170, step=0.00001, format="%.5f")
        features['HNR'] = st.sidebar.number_input('Harmonics-to-Noise Ratio', min_value=0.0, max_value=50.0, value=25.6780, step=0.0001)
        st.sidebar.subheader('Nonlinear Measures')
        features['RPDE'] = st.sidebar.number_input('Recurrence Period Density Entropy', min_value=0.0, max_value=1.0, value=0.427785, step=0.000001)
        features['DFA'] = st.sidebar.number_input('Detrended Fluctuation Analysis', min_value=0.0, max_value=1.0, value=0.723797, step=0.000001)
        features['spread1'] = st.sidebar.number_input('Spread1', min_value=-10.0, max_value=10.0, value=-6.635729, step=0.000001)
        features['spread2'] = st.sidebar.number_input('Spread2', min_value=-10.0, max_value=10.0, value=0.209866, step=0.000001)
        features['D2'] = st.sidebar.number_input('Correlation Dimension', min_value=0.0, max_value=10.0, value=1.957961, step=0.000001)
        features['PPE'] = st.sidebar.number_input('Pitch Period Entropy', min_value=0.0, max_value=1.0, value=0.135242, step=0.000001)
        return pd.DataFrame(features, index=[0])

    user_data = user_report()

    st.subheader('Patient Data')
    st.write(user_data)
    st.subheader('Patient Data Visualization')

    key_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 'DFA']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(key_features, user_data[key_features].values[0])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    normal_ranges = {
        'MDVP:Fo(Hz)': "Males: 85-180 Hz\n\nFemales: 165-255 Hz",
        'MDVP:Jitter(%)': "Normal: <1.040%\n\nParkinson's: Often >1.040%",
        'MDVP:Shimmer': "Normal: <3.810%\n\nParkinson's: Often >3.810%",
        'NHR': "Normal: <0.190\n\nParkinson's: Often >0.190",
        'HNR': "Normal: >21 dB\n\nParkinson's: Often <21 dB",
        'DFA': "Normal: ~0.5-0.7\n\nParkinson's: Often >0.7",
        'RPDE': "Normal: Lower values\n\nParkinson's: Higher values",
        'PPE': "Normal: Lower values\n\nParkinson's: Higher values"
    }

    def clinical_assessment(data):
        assessment = {}
        fo_value = data['MDVP:Fo(Hz)'][0]
        fo_assessment = []
        if 85 <= fo_value <= 180:
            fo_assessment.append("Within typical male range")
        if 165 <= fo_value <= 255:
            fo_assessment.append("Within typical female range")
        if fo_value < 85 or fo_value > 255:
            fo_assessment.append("Outside typical ranges")
        assessment['MDVP:Fo(Hz)'] = "\n".join(fo_assessment)
        jitter_value = data['MDVP:Jitter(%)'][0]
        if jitter_value < 0.01040:
            assessment['MDVP:Jitter(%)'] = "Normal range"
        else:
            assessment['MDVP:Jitter(%)'] = "Elevated - may indicate vocal pathology"
        shimmer_value = data['MDVP:Shimmer'][0]
        if shimmer_value < 0.03810:
            assessment['MDVP:Shimmer'] = "Normal range"
        else:
            assessment['MDVP:Shimmer'] = "Elevated - may indicate vocal pathology"
        nhr_value = data['NHR'][0]
        if nhr_value < 0.0190:
            assessment['NHR'] = "Normal range"
        else:
            assessment['NHR'] = "Elevated - may indicate vocal pathology"
        hnr_value = data['HNR'][0]
        if hnr_value > 21:
            assessment['HNR'] = "Normal range"
        else:
            assessment['HNR'] = "Reduced - may indicate vocal pathology"
        return assessment

    st.subheader('Clinical Assessment')
    assessment = clinical_assessment(user_data)
    for param, status in assessment.items():
        st.write(f"**{param}**:\n{status}")

    input_data_scaled = scaler.transform(user_data)
    svm_prediction = svm_model.predict(input_data_scaled)
    rf_prediction = rf_model.predict(input_data_scaled)
    log_reg_prediction = log_reg_model.predict(input_data_scaled)

    st.subheader('Prediction Results:')
    st.write(f"SVM: {'Parkinson\'s Disease Detected' if svm_prediction[0] == 1 else 'No Parkinson\'s Disease Detected'}")
    st.write(f"Random Forest: {'Parkinson\'s Disease Detected' if rf_prediction[0] == 1 else 'No Parkinson\'s Disease Detected'}")
    st.write(f"Logistic Regression: {'Parkinson\'s Disease Detected' if log_reg_prediction[0] == 1 else 'No Parkinson\'s Disease Detected'}")

    train_accuracy_svm = accuracy_score(Y_train, svm_model.predict(X_train))
    test_accuracy_svm = accuracy_score(Y_test, svm_model.predict(X_test))
    train_accuracy_rf = accuracy_score(Y_train, rf_model.predict(X_train))
    test_accuracy_rf = accuracy_score(Y_test, rf_model.predict(X_test))
    train_accuracy_log = accuracy_score(Y_train, log_reg_model.predict(X_train))
    test_accuracy_log = accuracy_score(Y_test, log_reg_model.predict(X_test))

    st.subheader('Model Accuracy:')
    st.write(f"SVM - Training Accuracy: {train_accuracy_svm * 100:.2f}%, Test Accuracy: {test_accuracy_svm * 100:.2f}%")
    st.write(f"Random Forest - Training Accuracy: {train_accuracy_rf * 100:.2f}%, Test Accuracy: {test_accuracy_rf * 100:.2f}%")
    st.write(f"Logistic Regression - Training Accuracy: {train_accuracy_log * 100:.2f}%, Test Accuracy: {test_accuracy_log * 100:.2f}%")

    st.markdown("---")
    st.markdown("## Clinical Reference Ranges")
    for param, range_value in normal_ranges.items():
        st.markdown(f"#### {param}")
        st.write(range_value)

    st.markdown("---")
    st.warning("**Medical Disclaimer**: This tool provides an estimate based on statistical models and should not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.")
