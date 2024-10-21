# ADVANCED COMPARATIVE ANALYSIS OF CUTTING-EDGE MACHINE LEARNING ALGORITHMS FOR PREDICTIVE ENGINE HEALTH AND FAULT DIAGNOSIS
This study investigates the performance of various machine learning algorithms in predicting engine conditions based on sensor data, including engine RPM, oil pressure, fuel pressure, and temperatures. The dataset contains over 21,200 records, and six models were evaluated: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, XGBoost, Long Short-Term Memory (LSTM), and Gradient Boosting. Data preprocessing involved scaling the features and splitting the data into training and testing sets. The accuracy of each model was calculated, with XGBoost achieving the highest accuracy at 68.50%, followed by Gradient Boosting (67.39%) and LSTM (66.76%). SVM had the lowest accuracy (64.57%), suggesting it may not be the best option for this problem. 
The results highlight that boosting algorithms ( XGBoost, Gradient Boosting) outperformed other models, while LSTM, despite its ability to handle sequential data, did not surpass these traditional methods. The comparison indicates that simpler machine learning models can often yield better results than complex deep learning techniques in certain contexts. Further research could involve hyperparameter tuning and leveraging time-series data for improved prediction.

Results: 
![image](https://github.com/user-attachments/assets/3b8a6b87-4415-4d15-9334-d15e93039d5d)


Stream Version 
Good Health
![image](https://github.com/user-attachments/assets/a258c225-ce46-49f3-9184-b4bb81955955)

Not Good Health
![image](https://github.com/user-attachments/assets/cb8194d8-6064-42cc-8bc1-0622d49e3241)

Requirements
scikit-learn
streamlit

Installation

Clone the repository

  git clone https://github.com/Kabilduke/EngineHealth.care.git
  
  cd
  
Create a virtual environment and activate it:

   python -m venv venv
   
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   
Install the required packages:

   pip install requirements.txt
   
Run the streamlit app:

   streamlit run app.py


   
Contribution


Contributions are welcome! Feel free to open an issue or submit a pull request for any changes or improvements.
