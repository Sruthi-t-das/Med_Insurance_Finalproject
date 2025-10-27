🟨 Day 1 — Data Familiarity

Goal:
Understand dataset structure, quality, and early cost drivers before deeper EDA.

Tasks Completed

Created virtual environment and installed dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn).

Loaded dataset (medical_insurance.csv) and inspected shape → (100 000 × 54).

Checked data types, missing values, and duplicates — minimal issues.

Cleaned columns (standardized names, handled missing categorical values).

Generated numeric distributions and categorical counts to explore value ranges.

Created derived features:

bmi_band (Underweight → Obese II+)

age_band (<25 → 65+)

Analyzed early cost signals by smoker, BMI, age, and region.

Confirmed smokers and high-BMI groups drive higher medical expenses.

🟨 Day 2 – Exploratory Data Analysis (EDA) & Hypothesis Testing

Objective:
Identify the major factors influencing annual medical cost through exploratory data analysis and statistical testing.

Steps Completed:

Loaded the cleaned dataset from Day 1 and verified data integrity.

Explored variable distributions (age, BMI, smoker status, chronic count, hospitalizations).

Created grouped summaries (“early cost signals”) by key drivers.

Visualized relationships using Seaborn and Plotly (boxplots, scatterplots, bar charts).

Formulated and statistically tested five hypotheses using ANOVA and correlation.

Hypotheses & Results:

| ID | Hypothesis                                  | Test Used       | Result                               |
| -- | ------------------------------------------- | --------------- | ------------------------------------ |
| H1 | Smokers have higher annual medical costs    | ANOVA           | ✅ Significant (p < 0.0001)           |
| H2 | Higher BMI → higher medical cost            | ANOVA           | ✅ Significant (p < 0.0001)           |
| H3 | Age is positively related to medical cost   | ANOVA           | ✅ Significant (p < 0.0001)           |
| H4 | More chronic diseases → higher medical cost | **Correlation** | ✅ Significant (p < 0.0001, r ≈ 0.30) |
| H5 | More hospitalizations → higher medical cost | **ANOVA**       | ✅ Significant (p < 0.0001)           |


Insights:

Smoking, higher BMI, and older age strongly increase average medical expenses.

Chronic disease count and hospitalization frequency are key cost amplifiers.

All tested hypotheses were statistically supported, confirming primary cost drivers for modeling.