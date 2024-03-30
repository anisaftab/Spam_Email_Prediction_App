import streamlit as st
import joblib
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from email import message_from_string
import re

lgbm_model = joblib.load("lgbm_model.pkl")
sgd_model = joblib.load("sgdClassifier.pkl")

#Function to convert MIME to plain text
def mime_to_text(mime_text):
  #Parse the MIME content
  email_message = message_from_string(mime_text)
  text_content = ""

  #Check if the email is multipart
  if email_message.is_multipart():
      for part in email_message.walk():
          content_type = part.get_content_type()
          content_disposition = str(part.get("Content-Disposition"))

          #Extract plain text
          if content_type == "text/plain" and "attachment" not in content_disposition:
              text = part.get_payload(decode=True)
              try:
                  text_content += text.decode('utf-8')
              except UnicodeDecodeError:
                  text_content += text.decode('utf-8', errors='replace')
          #Convert HTML parts to text
          elif content_type == "text/html" and "attachment" not in content_disposition:
              html = part.get_payload(decode=True)
              try:
                  soup = BeautifulSoup(html.decode('utf-8'), "html.parser")
                  text_content += soup.get_text()
              except UnicodeDecodeError:
                  soup = BeautifulSoup(html.decode('utf-8', errors='replace'), "html.parser")
                  text_content += soup.get_text()
  else:
      #Return text if not multipart
      try:
          text_content = email_message.get_payload(decode=True).decode('utf-8')
      except UnicodeDecodeError:
          text_content = email_message.get_payload(decode=True).decode('utf-8', errors='replace')
  return text_content

#function to convert html to plain text
def html_to_text(text):
  soup = BeautifulSoup(text, 'lxml')
  return soup.get_text()




def main():
    st.title("Email Analysis")
    
    # Text input for email
    email = st.text_area("Enter email here", key="email_input")
    
    # Button to analyze email
    if st.button("Analyze"):
        # Perform analysis and calculate spam percentage
        spam_percentage = analyze_email(email)
        
        # Display spam percentage
        st.write(f"Spam Percentage: {spam_percentage}%")

def analyze_email(email):
    #lightgbm_text = preprocess(email)
    #lgbm_prediction = lgbm_model.predict(lightgbm_text) * 100

    sgd_text = preprocess_email_sgd_logReg(email)
    sgd_prediction = sgd_model.predict(sgd_text) * 100
    
    return max(sgd_prediction, 0)

def preprocess(email):
    email.lower()
    mime_to_text(email)
    html_to_text(email)
    email = re.sub(r"\s+", " ", email)
    email = re.sub(r"[^\w\s]", "", email)
    email.strip()
    preprocess_email_sgd_logReg(email)
    return email


def preprocess_email_lightgbm(email):
    # Add your email preprocessing logic here
    # This is just a placeholder
    
    vectorizer = TfidfVectorizer(max_features=14804, stop_words='english')
    email_array = [email]
    lightgbm_text = vectorizer.fit_transform(email_array).toarray()

    return lightgbm_text

def preprocess_email_sgd_logReg(email):
    #Encoding the features column
    vectorizer = TfidfVectorizer(max_features= 30000, stop_words='english')
    sgd_logReg_text = vectorizer.fit_transform([email,]).toarray()
    return sgd_logReg_text

if __name__ == "__main__":
    main()


'''
# Load the models
    lgbm_model = joblib.load("lgbm_model.pkl")
    logistic_regression_model = joblib.load("logisticRegression.pkl")
    #nn_model = keras.models.load_model("nn_model.keras")
    sgd_classifier_model = joblib.load("sgdClassifier.pkl")
    
    # Preprocess the email
    preprocessed_email = preprocess_email(email)
    
    # Make predictions using the models
    lgbm_prediction = (lgbm_model.predict([preprocessed_email])[0]) * 100
    logistic_regression_prediction = (logistic_regression_model.predict([preprocessed_email])[0])* 100
    # nn_prediction = (nn_model.predict([preprocessed_email])[0]) * 100
    sgd_classifier_prediction = (sgd_classifier_model.predict([preprocessed_email])[0]) * 100
    
    # Calculate the average spam probability
    spam_probability = (lgbm_prediction + logistic_regression_prediction + 0 + sgd_classifier_prediction) / 4
    
    # Convert spam probability to percentage
    spam_percentage = spam_probability 
'''