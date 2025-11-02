# AI-Powered Credit Risk Assessment

## 📌 Project Overview  

This project leverages machine learning to assess credit risk, predicting whether a customer is at low risk or high risk of loan default based on financial and demographic factors. The model is deployed using Flask for API-based predictions and Streamlit for an interactive user interface, making risk evaluation seamless and efficient. 🚀
____________________________________________________________________________________________________________________________________________________________


## 📊 Dataset Details                                                                                                                                                                                                          
The credit risk model is trained on a well-structured dataset that includes financial and demographic attributes of loan applicants. This dataset provides insights into factors such as credit history, employment status, loan amount, and more, helping to assess an applicant’s likelihood of defaulting.                                                                                                                                           
🔹 Source: UCI Machine Learning Repository - Statlog (German Credit Data)                                                                                                                                                  
🔹 Size: 1,000 instances                                                                                                                                                                                                   
🔹 Features: 20 attributes (a mix of numerical and categorical variables)                                                                                                                                                 
🔹 Target Variable:                                                                                                                                                                                                        
       ✔ 1 (Good Credit) – Low risk of default                                                                                                                                                                                 
       ❌ 0 (Bad Credit) – High risk of default                                                                                                                                                                                
   
🔢 Feature Encodings in the Dataset                                                                                                                                                                                        
This project uses the German Credit Risk Dataset, where categorical features are encoded as follows:

1️⃣ Status of Existing Checking Account (Status)                                                                                                                                                                            
Indicates the applicant's checking account status:                                                                                                                                                                          
| Code | Description                        | Risk Level  |
|------|------------------------------------|-------------|
| A11  | No checking account               | High Risk   |
| A12  | Balance < 0 DM                     | High Risk   |
| A13  | 0 ≤ Balance < 200 DM               | Medium Risk |
| A14  | Balance ≥ 200 DM or salary assigned to account | Low Risk |                                                                                                                                                      

2️⃣ Duration in Months (Duration)
- Shorter loans (~12 months) → **Lower risk**  
- Longer loans (~48 months) → **Higher risk**


3️⃣ Credit History (CreditHistory)  
Describes past credit behavior:
| Code | Description                          | Risk Level  |
|------|--------------------------------------|-------------|
| A30  | No credit history (New customer)    | Medium Risk |
| A31  | Critical credit history (Defaults)  | High Risk   |
| A32  | Delayed payments in the past        | High Risk   |
| A33  | No known issues (Paid loans duly)   | Low Risk    |
| A34  | Existing credits paid back on time  | Low Risk    |


4️⃣ Purpose of Loan (Purpose)  
Purpose for which the loan is taken:
| Code  | Description            | Risk Level  |
|-------|------------------------|-------------|
| A40   | Car (new)              | Low Risk    |
| A41   | Car (used)             | Medium Risk |
| A42   | Furniture/equipment    | Low Risk    |
| A43   | Radio/television       | Medium Risk |
| A44   | Domestic appliances    | Medium Risk |
| A45   | Repairs                | Medium Risk |
| A46   | Education              | Medium Risk |
| A47   | Vacation               | High Risk   |
| A48   | Retraining             | High Risk   |
| A49   | Business               | Medium Risk |
| A410  | Others                 | Medium Risk |


5️⃣ Credit Amount (CreditAmount)
-	Higher loan amounts → **Higher risk**
-	Lower loan amounts (~500-2000 DM) → **Lower risk**

6️⃣ Savings Account/Bonds (Savings)
Indicates savings held by the applicant:
| Code  | Description                     | Risk Level  |
|-------|---------------------------------|-------------|
| A61   | No savings account              | High Risk   |
| A62   | < 100 DM savings                | High Risk   |
| A63   | 100 ≤ Savings < 500 DM          | Medium Risk |
| A64   | 500 ≤ Savings < 1000 DM         | Medium Risk |
| A65   | ≥ 1000 DM or life insurance     | Low Risk    |


7️⃣ Years Employed (Employment)
Indicates job stability:
| Code  | Description                   | Risk Level  |
|-------|-------------------------------|-------------|
| A71   | Unemployed                    | High Risk   |
| A72   | < 1 year                       | High Risk   |
| A73   | 1 ≤ Employment < 4 years       | Medium Risk |
 

8️⃣ Installment Rate as % of Disposable Income (InstallmentRate)  
•	Lower values (~1.5) → **Lower risk**  
•	Higher values (~3.5+) → **Higher risk**

9️⃣ Personal Status & Sex (PersonalStatus)
| Code  | Description                    | Risk Level  |
|-------|--------------------------------|-------------|
| A91   | Male, divorced/separated       | Medium Risk |
| A92   | Female, divorced/married       | Medium Risk |
| A93   | Male, single                   | High Risk   |
| A94   | Male, married/widowed          | Medium Risk |


🔟 Other Debtors or Guarantors (Debtors)
| Code  | Description                     | Risk Level  |
|-------|---------------------------------|-------------|
| A101  | No other debtors/guarantors     | High Risk   |
| A102  | Co-applicant                    | Medium Risk |
| A103  | Guarantor                        | Low Risk    |


1️⃣1️⃣ Years at Current Residence (ResidenceYears)  
•	Longer residence (~5+ years) → **Lower risk**  
•	Frequent movers (~1 year) → **Higher risk**

1️⃣2️⃣ Property (Property)
| Code  | Description                  | Risk Level  |
|-------|------------------------------|-------------|
| A121  | Real estate                  | Low Risk    |
| A122  | Car or savings ≥ 750 DM      | Medium Risk |
| A123  | Life insurance               | Medium Risk |
| A124  | No property                  | High Risk   |


1️⃣3️⃣ Age (Age)    
•	Older applicants (~45 years) → **Lower risk**  
•	Younger applicants (~20s) → **Higher risk**

1️⃣4️⃣ Other Installment Plans (OtherInstallment)
| Code  | Description  | Risk Level  |
|-------|-------------|-------------|
| A141  | None        | Low Risk    |
| A142  | Bank        | Medium Risk |
| A143  | Stores      | High Risk   |


1️⃣5️⃣ Housing (Housing)  
| Code  | Description            | Risk Level  |
|-------|------------------------|-------------|
| A151  | Rent                   | High Risk   |
| A152  | Own house              | Low Risk    |
| A153  | Free accommodation      | Medium Risk |


1️⃣6️⃣ Number of Existing Credits (ExistingCredits)    
•	1-2 loans → **Lower risk**  
•	>3 loans → **Higher risk**

1️⃣7️⃣ Job Type (Job)
| Code  | Description                         | Risk Level  |
|-------|-------------------------------------|-------------|
| A171  | Unskilled, unemployed              | High Risk   |
| A172  | Unskilled, employed                | Medium Risk |
| A173  | Skilled worker                     | Medium Risk |
| A174  | Highly qualified (e.g., manager)   | Low Risk    |


1️⃣8️⃣ Number of Liable Persons (LiablePersons)    
•	Lower values (~1 person) → **Lower risk**  
•	Higher values (~2+) → **Higher risk**

1️⃣9️⃣ Telephone (Telephone)
| Code  | Description                        | Risk Level  |
|-------|------------------------------------|-------------|
| A191  | No phone                          | High Risk   |
| A192  | Phone registered in own name      | Low Risk    |


2️⃣0️⃣ Foreign Worker (ForeignWorker)
| Code  | Description | Risk Level  |
|-------|------------|-------------|
| A201  | Yes        | Low Risk    |
| A202  | No         | High Risk   |

This dataset and encoding scheme help predict an applicant’s credit risk level efficiently! 🚀

This dataset serves as a benchmark for developing and evaluating credit scoring models, ensuring a reliable foundation for risk assessment. 🚀
____________________________________________________________________________________________________________________________________________________________


## Project Structure  
**🛠 train_model.py (Model Training & API Deployment)**  
This script is responsible for loading a pre-trained machine learning model and deploying it as a Flask API for credit risk assessment. It ensures that input data is processed correctly before making predictions.  
🔹 Key Functions:  
✅ Loads the trained credit risk model, scaler, and default feature values  
✅ Defines the expected feature set based on the training dataset  
✅ Implements a Flask API with a /predict endpoint to accept JSON input and return a credit risk prediction  
🔄 Workflow:  
1️⃣ Receives input data from a JSON request  
2️⃣ Validates and fills missing features using default values  
3️⃣ Applies feature scaling to standardize numerical data  
4️⃣ Uses the trained model to predict credit risk  
5️⃣ Returns a JSON response indicating whether the applicant is Low Risk ✅ or High Risk ⚠  
This API enables seamless real-time credit risk assessment, making it easy to integrate into financial applications. 🚀  

**🌍 app.py (Flask API for Production Deployment)**  
This script serves as the production-ready deployment of the AI-powered credit risk assessment API. It performs the same function as train_model.py, but is optimized for real-world use.  
🔹 Key Functions:  
✅ Implements a Flask REST API for real-time model inference  
✅ Handles POST requests containing applicant details  
✅ Processes input data to ensure it meets model requirements  
✅ Utilizes the trained credit risk model to predict risk level  
🔄 Workflow:  
1️⃣ Receives loan applicant details via a POST request  
2️⃣ Validates and preprocesses input data (handling missing values)  
3️⃣ Applies feature scaling to match training data format  
4️⃣ Makes a prediction using the trained model  
5️⃣ Returns a JSON response indicating whether the applicant is Low Risk ✅ or High Risk ⚠  
This script ensures a scalable and efficient API for seamless credit risk evaluation in production environments. 🚀  

**🎨 credit_risk_app.py (Streamlit UI for Interactive Credit Risk Assessment)**  
This script builds a user-friendly web application for real-time credit risk assessment using Streamlit. It allows users to input their financial details and receive an instant risk prediction with probability scores.  
🔹 Key Features:  
✅ Interactive Web App for seamless credit risk evaluation  
✅ Loads the trained ML model, scaler, label encoders, and default values  
✅ Dynamically generates input fields based on expected features  
✅ Provides a detailed probability-based prediction  
🔄 Workflow:  
1️⃣ Users input their details (both numerical and categorical values)  
2️⃣ Categorical values are encoded using pre-trained label encoders  
3️⃣ Numerical values are scaled to match the training format  
4️⃣ The trained model predicts the risk level (Low Risk / High Risk)  
5️⃣ Probability scores for both risk categories are displayed  
6️⃣ The classification threshold is adjusted for improved accuracy  
This AI-powered credit risk tool enables loan applicants and financial institutions to make informed decisions in an easy-to-use interface. 🚀  

**🔍 check_features.py (Feature Validation Utility)**  
This script helps validate the input features expected by the trained model, ensuring consistency between training and inference.  
🔹 Key Features:  
✅ Loads the scaler to inspect feature requirements  
✅ Checks the number of input features expected by the model  
✅ Useful for debugging feature mismatch errors  
____________________________________________________________________________________________________________________________________________________________  


## 🔄 Workflow:    
1️⃣ Loads the scaler.pkl file used for feature transformation  
2️⃣ Retrieves the number of input features required for model inference  
3️⃣ Helps detect inconsistencies in feature selection or preprocessing  
This utility ensures seamless integration between data preprocessing and model predictions, preventing errors in production deployment. 🚀  
____________________________________________________________________________________________________________________________________________________________  


## 🚀 How It Works  
1️⃣ User Input: The user enters financial and demographic details (e.g., credit history, employment status, loan amount).  
2️⃣ Preprocessing:  
•	Missing features are filled with default values.  
•	Categorical features are encoded.  
•	Numerical features are scaled to match the model’s training format.  
3️⃣ Prediction: The trained model analyzes the input data and classifies the applicant as Low Risk or High Risk.  
4️⃣ Result Display:  
•	The Flask API returns a JSON response with the risk classification.  
•	The Streamlit app provides an interactive UI and displays probability scores for better insight.  
This system enables fast and reliable credit risk assessment, helping financial institutions make informed decisions. 💡  
____________________________________________________________________________________________________________________________________________________________  


## Low-Risk vs. High-Risk Examples    
**📌 ✅ Low-Risk Profile (More Creditworthy Applicant)**  
This applicant has strong financial stability, reliable credit history, and responsible borrowing behavior.  
| Feature              | Optimized Value  | Why It’s Low Risk?                                |
|----------------------|-----------------|--------------------------------------------------|
| Status              | A14              | Favorable status                                |
| Duration            | 12.00 months     | Shorter loan duration reduces risk             |
| Credit History      | A34              | Good credit history                            |
| Purpose             | A40              | No negative impact                             |
| Credit Amount       | 500.00           | Moderate loan amount, avoiding extremes       |
| Savings            | A65              | Highest savings category                      |
| Employment         | A75              | Longest employment tenure                     |
| Installment Rate   | 1.50             | Lower installment burden                      |
| Personal Status    | A91              | Neutral impact                                |
| Debtors           | A101              | No co-debtors (low risk)                      |
| Residence Years    | 5.00             | Stable residence history                      |
| Property          | A121              | Owns valuable property                        |
| Age               | 45.00             | Older applicants tend to be more financially stable |
| Other Installment | A141              | No other installment obligations              |
| Housing           | A152              | Owns a house (reduces risk)                   |
| Existing Credits  | 1.00              | Not too many active loans                     |
| Job               | A174              | Best job category (stable income)             |
| Liable Persons    | 1.00              | Fewer financial dependents                    |
| Telephone         | A191              | Has a registered telephone                    |
| Foreign Worker    | A201              | No additional risk                            |
  
**⚠ High-Risk Profile (Likely to Default)**  
This applicant has unstable financial behavior, high borrowing tendencies, and potential repayment issues.  
Feature	Value	Why It’s High Risk?  
| Feature              | Optimized Value  | Why It’s Low Risk?                                 |
|----------------------|-----------------|---------------------------------------------------|
| Status              | A11              | Unfavorable status                               |
| Duration            | 20.90 months     | Longer loan term increases default risk         |
| Credit History      | A30              | Poor credit history                             |
| Purpose             | A40              | No mitigating effect                           |
| Credit Amount       | 0.99             | Very low loan amount might indicate poor creditworthiness |
| Savings             | A63              | Low savings                                    |
| Employment          | A71              | Shorter employment history (unstable income)   |
| Installment Rate    | 2.97             | Higher installment burden                      |
| Personal Status     | A91              | No positive influence                          |
| Debtors            | A101              | No co-debtors (neutral impact)                 |
| Residence Years     | 2.85             | Shorter residence duration (less stability)    |
| Property           | A124              | No valuable property owned                     |
| Age                | 35.55             | Younger applicants have a higher default rate  |
| Other Installment  | A141              | Existing installment obligations               |
| Housing            | A151              | Rented housing (higher risk)                   |
| Existing Credits   | -0.01             | Possible errors or multiple loans              |
| Job                | A171              | Lower job category (unstable income)           |
| Liable Persons     | 1.16              | More financial dependents                      |
| Telephone          | A191              | No additional risk impact                      |
| Foreign Worker     | A201              | No additional risk impact                      |


💡 Key Takeaways:  
🔹 Applicants with stable employment, good credit history, property ownership, and manageable financial obligations are at lower risk.  
🔹 Factors like longer loan duration, poor savings, unstable job history, and higher installment burdens contribute to a high-risk profile.  
____________________________________________________________________________________________________________________________________________________________  


## 🚀 How to Run the Project    
🛠 Setup Instructions  
1️⃣ Clone the Repository  
    >> git clone https://github.com/your-username/credit-risk-assessment.git    
    >> cd credit-risk-assessment    
2️⃣ Install Dependencies  
    Make sure you have Python installed, then run:  
    >> pip install -r requirements.txt  
3️⃣ Start the Flask API  
    >> python app.py  
   📍 The API will be available at: http://127.0.0.1:5000/predict  
4️⃣ Launch the Streamlit UI  
    >> streamlit run credit_risk_app.py    
    📌 This will open an interactive web app where users can input their details and receive credit risk predictions.  
____________________________________________________________________________________________________________________________________________________________  

    
## 🏆 Key Features   
✔ AI-powered credit risk prediction – Predicts loan applicant risk using machine learning.  
✔ Flask API for easy integration – Provides a REST API for seamless model deployment.  
✔ Streamlit UI for user-friendly risk assessment – Interactive web interface for real-time predictions.  
✔ Probability scores for better transparency – Displays the likelihood of being low or high risk.  
✔ Categorical encoding & feature scaling – Ensures accurate preprocessing of input data.  
✔ Automatic feature validation to prevent errors – Checks input consistency with the trained model.  
____________________________________________________________________________________________________________________________________________________________  


## 🤖 Model Details
•	Trained using advanced Machine Learning algorithms to ensure accurate risk assessment.  
•	Feature scaling applied to normalize numerical inputs for better predictions.  
•	Supports categorical encoding to handle non-numerical data effectively.  
•	Optimized threshold tuning for improved classification performance.  
____________________________________________________________________________________________________________________________________________________________  


## 🚀 Future Enhancements  
•	🔍 Expand feature set to enhance prediction accuracy.  
•	📊 Integrate alternative credit scoring techniques (e.g., behavioral analytics, alternative data sources).  
•	🎨 Enhance UI/UX for a more intuitive and interactive experience.  
____________________________________________________________________________________________________________________________________________________________  


## 📢 Contributing  
We welcome contributions! You can:  
•	🚀 Submit a pull request with improvements or new features.  
•	🛠 Report bugs or suggest enhancements in the Issues section.  
•	⭐ Star this repository if you find it useful and want to support the project!  
📧 Contact  
For any queries or collaboration opportunities, feel free to reach out:  
📩 Email: kartik.nigam78@gmail.com  
____________________________________________________________________________________________________________________________________________________________  
