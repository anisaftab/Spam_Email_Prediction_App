import streamlit as st
import joblib
import tensorflow as tf
from tensorflow import keras
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from email import message_from_string
import warnings
import re

warnings.filterwarnings('ignore')

log_reg_model = joblib.load("log_reg_model.pkl")
sgd_model = joblib.load("sgdClassifier.pkl")
lgbm_model = joblib.load('lgbm_model.pkl')

phish_vectorizer = joblib.load('phish_vectorizer.joblib')
spam_vectorizer = joblib.load('spam_vectorizer.joblib')
nn_tokenizer = joblib.load('nn_tokenizer.joblib')

def custom_initializer(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.1, dtype=dtype)
nn_model = load_model("nn_model.keras", custom_objects={'custom_initializer': custom_initializer})


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
    st.title("Email Spam and Phishing Detector")
    
    #Text input for email
    email = st.text_area("Enter email here", key="email_input")
    #Button to analyze email
    if st.button("Analyze"):
        # Perform analysis and calculate spam percentage
        phish_percentage = phish_analyze(email)
        spam_percentage = spam_analyze(email)

        # Display phishing and spam percentages
        st.write(f"Phishing Email Chance: {phish_percentage}%")
        st.write(f"Spam Email Chance: {spam_percentage}%")



def phish_analyze(email):
    text = preprocess(email)
    text = phish_vectorizer.transform([text])

    log_reg_prediction = log_reg_model.predict(text) * 100
    #sgd_prediction = sgd_model.predict(text) * 100
    average = log_reg_prediction
    return  int(average)


def spam_analyze(email):
    text = preprocess(email)

    spam_text = spam_vectorizer.transform([text])

    nn_text_tokenized = nn_tokenizer.texts_to_sequences([text])
    nn_text_padded = pad_sequences(nn_text_tokenized, maxlen=14804 )

    nn_prediction = nn_model.predict(nn_text_padded) * 100
    lgbm_prediction = lgbm_model.predict(spam_text) * 100
    print(nn_prediction)
    print(lgbm_prediction)
    average = (nn_prediction * 0.8 + lgbm_prediction * 0.2) / 2

    return int(average)

def preprocess(email_text):
    email_text = email_text.lower().strip()
    email_text = re.sub(r"\s+", " ", email_text) 
    email_text = re.sub(r"[^\w\s]", "", email_text) 
    email_text = mime_to_text(email_text)  
    email_text = html_to_text(email_text)  
    return email_text

if __name__ == "__main__":
    main()



# Load the models
#    lgbm_model = joblib.load("lgbm_model.pkl")
#    logistic_regression_model = joblib.load("logisticRegression.pkl")
#    #nn_model = keras.models.load_model("nn_model.keras")
#    sgd_classifier_model = joblib.load("sgdClassifier.pkl")
#    
#    # Preprocess the email
#    preprocessed_email = preprocess_email(email)
#    
#    # Make predictions using the models
#    lgbm_prediction = (lgbm_model.predict([preprocessed_email])[0]) * 100
#    logistic_regression_prediction = (logistic_regression_model.predict([preprocessed_email])[0])* 100
#    # nn_prediction = (nn_model.predict([preprocessed_email])[0]) * 100
#    sgd_classifier_prediction = (sgd_classifier_model.predict([preprocessed_email])[0]) * 100
#    
#    # Calculate the average spam probability
#    spam_probability = (lgbm_prediction + logistic_regression_prediction + 0 + sgd_classifier_prediction) / 4
#    
#    # Convert spam probability to percentage
#    spam_percentage = spam_probability 
