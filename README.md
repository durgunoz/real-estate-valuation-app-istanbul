# real-estate-valuation-app-istanbul
A Streamlit app that predicts house prices based on location and property features using a trained Random Forest model.

# House Price Prediction App

This project predicts estimated house prices based on district, neighborhood, and property features such as square meters, number of rooms, age, and floor with 0.92 R2 rate.

Technologies Used:
- Python
- Streamlit (Web UI)
- Scikit-learn (Machine Learning)
- Random Forest Regressor (Model)
- Joblib (Model saving/loading)
- Selenium WebDriver (for optional data scraping/automation)

Model: Trained using Random Forest and saved with joblib.

To Run the App:
1. Install required libraries: `pip install -r requirements.txt`
2. Launch the app: `streamlit run app.py`
