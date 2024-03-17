# Email-Classifier
Spam/Ham Email Classifier with Logistic Regression

Email spam detection is crucial for maintaining a clean and efficient inbox. In this project, we develop a spam/ham email classifier using a Logistic Regression model. The dataset provided is in CSV format and contains labeled emails as spam or ham (non-spam).

## Dataset Overview
The dataset consists of email samples labeled as spam or ham. Each sample may include various features extracted from the email content, such as word frequency, presence of specific keywords, or other relevant characteristics.

## Project Goals
1. **Data Loading and Preprocessing**
   - Utilize Pandas to load the dataset from the provided CSV file.
   - Perform preprocessing tasks such as data cleaning and normalization.
   - Use a feature vectorizer to convert text data into numerical feature vectors suitable for machine learning algorithms.

2. **Logistic Regression Classification**
   - Implement a Logistic Regression model using scikit-learn.
   - Train the model on the preprocessed email data.
   - Evaluate the model's performance using appropriate metrics such as accuracy, precision, recall, and F1-score.

3. **Email Classification**
   - Develop a function that takes an input email and predicts whether it is spam or ham using the trained Logistic Regression model.
   - Provide an example of how to use the classifier function with sample input emails.

## Usage
1. Clone the repository to your local machine.
2. Download the dataset from the provided link and place it in the appropriate directory.
3. Execute the provided Python script to load, preprocess, train the model, and classify emails.
4. Use the classifier function to predict whether a given email is spam or ham.

## Example Usage


# In the input field just imput for your own email
# Sample input email
input = ["Congratulations! You have been selected as the winner of our monthly prize draw. You've won a brand new iPhone 12 Pro Max!To claim your prize, simply click on the link below and provide your shipping details. Hurry, as this offer is only valid for the next 24 hours!Please note that failure to claim your prize within the specified time frame will result in forfeiture.Thank you for participating in our contest, and we look forward to hearing from you soon!"]

# Transform to form input_features
input_features = feature_extraction.transform(input)

# Predict whether the emails are spam or ham
predInput = model.predict(input_features)

# Print the output to know whether the input email is a spam or ham
print(predInput)

if(predInput[0] == 1):
  print("Its a Ham")
else:
  print("Its a Spam")
