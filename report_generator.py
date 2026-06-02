import pandas as pd
import numpy as np
from openai import OpenAI

def get_cohort_stats(df, user_profile):
    """
    Finds a cohort of similar individuals in the dataset based on key risk factors
    and computes statistical summaries to provide as context to the AI.
    """
    # Define filtering stages for fallback if cohort is too small
    stages = [
        # Stage 1: Full match on demographic and primary vitals
        ["Age_Category", "BMI_Category", "HighBP", "HighChol"],
        # Stage 2: Relax cholesterol
        ["Age_Category", "BMI_Category", "HighBP"],
        # Stage 3: Relax blood pressure
        ["Age_Category", "BMI_Category"],
        # Stage 4: Relax age, match just BMI
        ["BMI_Category"]
    ]
    
    cohort = df
    matched_criteria = []
    
    for criteria in stages:
        temp_cohort = df
        for col in criteria:
            val = user_profile.get(col)
            if val is not None:
                temp_cohort = temp_cohort[temp_cohort[col] == val]
        
        # Check if we have enough samples for statistical significance
        if len(temp_cohort) >= 50:
            cohort = temp_cohort
            matched_criteria = criteria
            break
            
    # Calculate statistics of this cohort
    cohort_size = len(cohort)
    
    # Diabetes distribution in cohort
    counts = cohort["Diabetes_Status"].value_counts(normalize=True) * 100
    pct_no = round(counts.get("No Diabetes", 0.0), 1)
    pct_pre = round(counts.get("Pre-Diabetic", 0.0), 1)
    pct_diab = round(counts.get("Diabetic", 0.0), 1)
    
    # Overall average rates in the entire dataset for comparison
    overall_counts = df["Diabetes_Status"].value_counts(normalize=True) * 100
    overall_pct_diab = round(overall_counts.get("Diabetic", 0.0), 1)
    
    # Relative risk multiplier
    if overall_pct_diab > 0:
        relative_risk = round(pct_diab / overall_pct_diab, 2)
    else:
        relative_risk = 1.0
        
    # Other lifestyle statistics in this cohort
    cohort_smoker = round(cohort["Smoker"].mean() * 100, 1)
    cohort_phys_act = round(cohort["PhysActivity"].mean() * 100, 1)
    cohort_diff_walk = round(cohort["DiffWalk"].mean() * 100, 1)
    cohort_heart_disease = round(cohort["HeartDiseaseorAttack"].mean() * 100, 1)
    
    # Average general health (1=Excellent, 5=Poor)
    cohort_avg_genhlth = round(cohort["GenHlth"].mean(), 2)
    
    # Format criteria names for display
    display_mapping = {
        "Age_Category": "Age Group",
        "BMI_Category": "BMI Category",
        "HighBP": "High Blood Pressure",
        "HighChol": "High Cholesterol"
    }
    display_criteria = [display_mapping[col] for col in matched_criteria]
    
    stats = {
        "cohort_size": cohort_size,
        "matched_criteria": ", ".join(display_criteria),
        "pct_no_diabetes": pct_no,
        "pct_pre_diabetic": pct_pre,
        "pct_diabetic": pct_diab,
        "overall_pct_diabetic": overall_pct_diab,
        "relative_risk": relative_risk,
        "cohort_smoker_pct": cohort_smoker,
        "cohort_phys_act_pct": cohort_phys_act,
        "cohort_diff_walk_pct": cohort_diff_walk,
        "cohort_heart_disease_pct": cohort_heart_disease,
        "cohort_avg_genhlth": cohort_avg_genhlth,
        "distribution": cohort["Diabetes_Status"].value_counts().to_dict()
    }
    return stats

def generate_report(user_profile, cohort_stats, openai_api_key=None, gemini_api_key=None):
    """
    Generates a personalized health report using Google Gemini (recommended free tier) or OpenAI.
    """
    # Format general health rating text
    gen_hlth_labels = {1.0: "Excellent", 2.0: "Very Good", 3.0: "Good", 4.0: "Fair", 5.0: "Poor"}
    user_gen_hlth = gen_hlth_labels.get(user_profile["GenHlth"], "Unknown")
    
    # Format sex text
    user_sex = "Male" if user_profile["Sex"] == 1.0 else "Female"
    
    # Create prompt
    prompt = f"""
    A user has submitted their personal health profile on a healthcare risk dashboard.
    We have also queried a CDC BRFSS dataset containing 253,680 records to find a cohort of individuals with similar key risk factors and compared the user's profile against them.
    
    ### USER PROFILE DATA
    - Age Group: {user_profile['Age_Category']}
    - Sex: {user_sex}
    - BMI: {user_profile['BMI']} ({user_profile['BMI_Category']})
    - High Blood Pressure: {'Yes' if user_profile['HighBP'] == 1.0 else 'No'}
    - High Cholesterol: {'Yes' if user_profile['HighChol'] == 1.0 else 'No'}
    - Checked Cholesterol in last 5 years: {'Yes' if user_profile['CholCheck'] == 1.0 else 'No'}
    - History of Smoking (>=100 cigarettes): {'Yes' if user_profile['Smoker'] == 1.0 else 'No'}
    - History of Stroke: {'Yes' if user_profile['Stroke'] == 1.0 else 'No'}
    - History of Heart Disease or Heart Attack: {'Yes' if user_profile['HeartDiseaseorAttack'] == 1.0 else 'No'}
    - Physical Activity (past 30 days): {'Yes' if user_profile['PhysActivity'] == 1.0 else 'No'}
    - Daily Fruit Consumption: {'Yes' if user_profile['Fruits'] == 1.0 else 'No'}
    - Daily Vegetable Consumption: {'Yes' if user_profile['Veggies'] == 1.0 else 'No'}
    - Heavy Alcohol Consumption: {'Yes' if user_profile['HvyAlcoholConsump'] == 1.0 else 'No'}
    - Self-rated General Health: {user_gen_hlth} (on a scale of 1-5, where 1 is Excellent and 5 is Poor)
    - Days of Poor Physical Health in past 30 days: {user_profile['PhysHlth']}/30 days
    - Days of Poor Mental Health in past 30 days: {user_profile['MentHlth']}/30 days
    - Difficulty Walking or Climbing Stairs: {'Yes' if user_profile['DiffWalk'] == 1.0 else 'No'}
    
    ### DATA-DRIVEN COHORT INSIGHTS (From 253,680 records)
    - Matched Cohort Size: {cohort_stats['cohort_size']} individuals who also match the following factors: {cohort_stats['matched_criteria']}
    - Diabetes Prevalence in Matched Cohort:
      - Diabetic: {cohort_stats['pct_diabetic']}%
      - Pre-Diabetic: {cohort_stats['pct_pre_diabetic']}%
      - No Diabetes: {cohort_stats['pct_no_diabetes']}%
    - General Population Diabetes Prevalence: {cohort_stats['overall_pct_diabetic']}%
    - Relative Risk Multiplier: {cohort_stats['relative_risk']}x compared to the general population in the dataset.
    - Within this matched cohort:
      - Smoker Prevalence: {cohort_stats['cohort_smoker_pct']}%
      - Physical Activity Rate: {cohort_stats['cohort_phys_act_pct']}%
      - Difficulty Walking Rate: {cohort_stats['cohort_diff_walk_pct']}%
      - Heart Disease Prevalence: {cohort_stats['cohort_heart_disease_pct']}%
      - Average Self-Rated Health: {cohort_stats['cohort_avg_genhlth']}/5
    
    ### TASK
    Write a concise, direct, personalized, and highly actionable health evaluation report addressing the user directly ("you", "your").
    Avoid generic introductory sentences, verbose boilerplate explanations, and unnecessary medical jargon. 
    Keep the report short, to-the-point, and easily scannable using bullet points and short paragraphs (2-3 sentences max).
    
    Structure the report using these exact Markdown headings:
    
    # MY PERSONALIZED DIABETES RISK & PREVENTION REPORT
    
    ## My Health Summary
    Provide a concise (2-3 sentences) overview of your health profile, overall diabetes risk level, and matched cohort results.
    
    ## My Risk Compared to Similar Cohorts
    Explain in a few bullet points what your risk means compared to the {cohort_stats['cohort_size']} similar individuals in the CDC BRFSS dataset, including your relative risk multiplier ({cohort_stats['relative_risk']}x) and the prevalence of diabetes in your cohort.
    
    ## My Risk Factors & Strengths
    - **My Primary Risk Factors**: Directly list and briefly explain your key risk factors (such as BMI, high BP, cholesterol, lack of exercise, smoking, etc.) and how they impact your body.
    - **My Strengths & Protective Factors**: Directly list your positive habits (such as exercise, fruit/vegetable intake, not smoking, etc.) and how they help protect your health.
    
    ## My Step-by-Step Prevention Plan
    Provide highly specific, short bullet points:
    - **Dietary Action**: Actionable food/nutrition changes tailored to your BMI and vitals.
    - **Physical Activity**: Concrete, low-impact or active exercise recommendations matching your physical activity level and walking ability.
    - **Daily Habits**: Stress, sleep, or mental/physical health management points based on your reported poor health days.
    
    ## Recommended Health Screenings & Discussions
    Provide a checklist of 3-4 specific clinical tests or targets to bring to your next doctor's appointment (e.g., HbA1c screening, target blood pressure, or lipid panel frequency).
    
    ## Important Medical Disclaimer
    State clearly in 1-2 sentences that this is an educational assessment based on statistical survey data and not a formal diagnosis or medical advice.
    
    Keep the tone direct, supportive, and evidence-based.
    """
    
    system_instruction = "You are a clinical AI health advisor. You analyze survey risk factors and statistical medical data to provide patient-facing educational reports on metabolic and chronic disease risks. Keep the tone professional, empathetic, objective, and clear."
    
    if gemini_api_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction
        )
        response = model.generate_content(prompt)
        return response.text
    elif openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        return response.choices[0].message.content
    else:
        raise ValueError("No API Key configured for report generation. Please configure Gemini or OpenAI key.")
