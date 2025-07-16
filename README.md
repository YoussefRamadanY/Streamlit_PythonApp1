---

# 📱 Social Media Addiction Score Predictor

A machine learning project that predicts a student's social media addiction score based on their personal and behavioral attributes. Built with Python, trained on real survey data, and deployed via a Streamlit web interface.

---

## 🧠 Overview

This project uses a **Linear Regression model** to predict a numerical **Addiction Score** from factors like age, gender, platform usage, sleep patterns, and relationship status. The goal is to gain insight into how different habits and demographics influence potential social media addiction.

---

## 📊 Dataset

- **Source**: [Kaggle - Social Media Addiction vs Relationships](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)
- **Size**: ~300 entries
- **Features**:
  - Age, Gender, Relationship Status
  - Average daily social media usage
  - Sleep hours per night
  - Most used social media platform
  - Self-reported addiction score

---

## 🔍 Features & Flow

1. **Data Cleaning**:
   - Dropped duplicates and irrelevant columns
   - Standardized column names
   - One-hot encoding for categorical variables

2. **Feature Engineering**:
   - Grouped age, sleep hours, and usage into categories
   - Consolidated less common platforms under "Other"

3. **Modeling**:
   - Trained a Linear Regression model
   - Evaluated using R² Score and Mean Squared Error
   - Visualized feature importance

4. **Deployment**:
   - Built an interactive UI using Streamlit
   - Accepts input fields and provides a score + qualitative feedback



---

🖥️ Web Interface

Input: Age, Gender, Sleep hours, Daily usage, Platform, Relationship status

Output: Addiction Score (with color-coded risk level and interpretation)


Example:

> Addiction Score: 9.55
🟠 High risk: Noticeable impact from social media usage.




---

📈 Model Performance

R² Score: 0.959

Top Features:

Daily usage hours

Sleep patterns

Most used platforms

Relationship status




---

💡 Future Enhancements

Switch to more advanced models (e.g., XGBoost or Random Forest)

Add interactive visualizations to the web app

Track trends over time or allow batch predictions

Explore classification thresholds for labeling addiction levels



---

🤝 Acknowledgements

Dataset by Adil Shamim on Kaggle

Built with Python, Scikit-learn, and Streamlit



---

📜 License

This project is for academic and educational use only.
You are free to fork and build upon it with credit. 
