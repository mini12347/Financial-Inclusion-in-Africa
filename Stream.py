import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from PIL import Image
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
df = pd.read_csv('Financial_inclusion_dataset.csv')
st.set_page_config(page_title="Bank Account Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Bank Account", "Dashboard"])
if page == "Predict Bank Account":
    st.title("Predict Bank Account Ownership")
    st.image('UNUZ4zR - Imgur.jpg')
    st.markdown("Enter the information below to predict if a person is likely to have a bank account.")
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("Country", df['country'].unique())
            year = st.number_input("Year", min_value=2000, max_value=2025, value=2025)
            location_type = st.radio("Location Type", ['Rural', 'Urban'])
            cellphone_access = st.radio("Cellphone Access", ['Yes', 'No'])
            household_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)

        with col2:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.radio("Gender", ['Male', 'Female'])
            relationship = st.selectbox("Relationship with Head", df['relationship_with_head'].unique())
            marital_status = st.selectbox("Marital Status", df['marital_status'].unique())
            education_level = st.selectbox("Education Level", df['education_level'].unique())

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        input_df = pd.DataFrame({
            'country': [country],
            'year': [year],
            'location_type': [location_type],
            'cellphone_access': [cellphone_access],
            'household_size': [household_size],
            'age': [age],
            'gender': [gender],
            'relationship_with_head': [relationship],
            'marital_status': [marital_status],
            'education_level': [education_level]
        })
        input_df['cellphone_access'] = input_df['cellphone_access'].map({'Yes':1, 'No':0})
        input_df['location_type'] = input_df['location_type'].map({'Urban':1, 'Rural':0})
        input_df = pd.get_dummies(input_df, columns=['country', 'gender', 'relationship_with_head', 'marital_status', 'education_level'])

        model_features = model.get_booster().feature_names if hasattr(model, 'get_booster') else model.feature_names_in_
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        st.markdown(f"### Prediction: **{'Has Bank Account' if prediction==1 else 'No Bank Account'}**")
        st.progress(int(prediction_proba*100))
        st.write(f"Probability of having a bank account: {prediction_proba:.2f}")
if page == "Dashboard":
    st.title("🌍 Bank Account Ownership Dashboard")
    st.markdown("Explore interactive insights from the financial inclusion dataset.")

    selected_country = st.selectbox("Select Country", ["All"] + list(df['country'].unique()))
    if selected_country != "All":
        data = df[df['country'] == selected_country].copy()
    else:
        data = df.copy()

    data['bank_account'] = data['bank_account'].map({'Yes': 1, 'No': 0})
    data['bank_account_label'] = data['bank_account'].map({1: 'Has Account', 0: 'No Account'})

    total = len(data)
    has_account = data['bank_account'].sum()
    pct_has_account = has_account / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Respondents", f"{total:,}")
    col2.metric("Have Bank Accounts", f"{has_account:,}")
    col3.metric("Percentage with Bank Accounts", f"{pct_has_account:.2f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Bank Account Distribution")
        pie_data = data['bank_account_label'].value_counts().reset_index()
        pie_data.columns = ['Account Status', 'Count']
        fig_pie = px.pie(
            pie_data, 
            names='Account Status', 
            values='Count', 
            color='Account Status',
            color_discrete_map={'Has Account': '#2ecc71', 'No Account': '#e74c3c'},
            hole=0.4,
            template='plotly_dark'
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Bank Account Ownership by Location Type")
        fig1 = px.bar(
            data.groupby('location_type')['bank_account'].mean().reset_index(),
            x='location_type', y='bank_account',
            labels={'bank_account':'% with Bank Account'},
            color='location_type', template='plotly_dark'
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ownership by Gender")
        fig2 = px.bar(
            data.groupby('gender_of_respondent')['bank_account'].mean().reset_index(),
            x='gender_of_respondent', y='bank_account',
            labels={'bank_account':'% with Bank Account', 'gender_of_respondent':'Gender'},
            color='gender_of_respondent', template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Ownership by Education Level")
        fig_edu = px.bar(
            data.groupby('education_level')['bank_account'].mean().reset_index(),
            x='education_level', y='bank_account',
            labels={'bank_account':'% with Bank Account', 'education_level':'Education Level'},
            color='education_level', template='plotly_dark'
        )
        fig_edu.update_xaxes(tickangle=45)
        st.plotly_chart(fig_edu, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution by Bank Account")
        fig3 = px.box(
            data, 
            x='bank_account_label', y='age_of_respondent',
            color='bank_account_label',
            labels={'bank_account_label':'Account Status', 'age_of_respondent':'Age'},
            template='plotly_dark'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
       st.subheader("Age vs. Probability of Bank Account")
       fig3 = px.histogram(data, x='age_of_respondent', color='bank_account', barmode='overlay', template='plotly_dark')
       st.plotly_chart(fig3, use_container_width=True)
