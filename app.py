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
import base64
import quopri
import warnings
import re
from langdetect import detect

warnings.filterwarnings('ignore')


def custom_initializer(shape, dtype=None):
        return tf.random.normal(shape, mean=0.0, stddev=0.1, dtype=dtype)

lgbm_model = joblib.load('lgbm_model.pkl')
nn_model = tf.keras.models.load_model('nn_model.keras', custom_objects={'custom_initializer': custom_initializer})

logistic_regression_model = joblib.load('logistic_regression_model.pkl')
sgd_classifier = joblib.load('sgd_classifier.pkl')

spam_vectorizer = joblib.load('spam_vectorizer.joblib')
phish_vectorizer = joblib.load('phish_vectorizer.joblib')
nn_tokenizer = joblib.load('nn_tokenizer.joblib')


def main():
    st.title("Email Spam and Phishing Detector")
    
    #Initialize the email input in session state if it doesn't exist
    if 'email_input' not in st.session_state:
        st.session_state.email_input = ""


    #Text input for email
    email = st.text_area("Enter email here", height= 250 , key='email_input', value=st.session_state.email_input)

    #Button to analyze email
    if st.button("Analyze"):
        # Perform analysis and calculate spam percentage
        spam_percentage = spam_analyze(email)
        phish_percentage = phish_analyze(email)

        if email == "":
            st.write("Please enter an email to analyze")
        elif heuristic_filter(email):
            st.write("<p style='color:red; font-size:20px;'>Spam/Phishing email!! </p>", unsafe_allow_html=True)
        else:
            spam_flag = None
            phish_flag = None

            if spam_percentage == 0:
                spam_flag = False 
            else: spam_flag = True


            if phish_percentage == 0:
                phish_flag = False 
            else: phish_flag = True

            # Display phishing and spam percentages
            st.write("Spam Email:", spam_flag)
            st.write("Phishing Email:", phish_flag)


            st.write("Developed by Anis, Terrence, and Noah")
            st.write("Project under the supervision of Mohamad Hoda")


    
            

        
def heuristic_filter(email):
    # Check for hidden text using zero-width characters
    if not check_hidden_text(email):
        return True
    # Check for language other than English
    if detect_language(email):
        return True
    if high_risk_words(email):
        return True
    return False

def high_risk_words(email):
    if ( 'urgent send money' in email):
        return True
    return False
# Function to detect the language of the email content
def detect_language(email_content):
    try:
        detected_lang = detect(email_content)
        if detected_lang and detected_lang != 'en':
            return True
    except Exception as e:
        st.error(f"Error occurred during language detection: {e}")
        return None
    return False
    
# Function to check for hidden text using zero-width characters
def check_hidden_text(content):
    hidden_text_patterns = [
        r'[\u200B-\u200D\uFEFF]',  # Zero-width characters
        r'[\u2060-\u2064]'          # Invisible Unicode characters
    ]
    for pattern in hidden_text_patterns:
        if re.search(pattern, content):
            return False
    return True


def spam_analyze(email):
    text = preprocess(email)
    print(text)
    spam_text = spam_vectorizer.transform([text])
    lgbm_prediction = lgbm_model.predict(spam_text) * 100
    print('LGBM prediction:', lgbm_prediction)
    return int(lgbm_prediction)

def phish_analyze(email):
    text = preprocess(email)
    print(text)
    phish_text = phish_vectorizer.transform([text])
    sgd_prediction = sgd_classifier.predict(phish_text) * 100
    logistic_regression_prediction = logistic_regression_model.predict(phish_text) * 100
    return int(sgd_prediction)

def preprocess(text):
    if "------=" in text:
        start_plaintext = text.find("charset=\"Windows-1252\"") + len("charset=\"Windows-1252\"")
        end_plaintext = text.find("------=", start_plaintext)
        plaintext_content = text[start_plaintext:end_plaintext].strip()

        #Decode quoted-printable encoding and clean up the content
        ascii_content = plaintext_content.encode('ascii', 'ignore').decode('ascii')
        decoded_content = quopri.decodestring(ascii_content).decode('windows-1252')
        text = decoded_content.replace("=0A=", "\n").replace("=20", " ").replace("=09", "\t")

        #Replace any MIME encoding artifacts
        text = decoded_content.replace("=0A=", "\n").replace("=20", " ").replace("=09", "\t")

    if "base64" in text.lower():
        charset_match = re.search(r'charset="([^"]+)"', text)
        if charset_match:
            charset = charset_match.group(1)
        encoded_data = text.strip().split('base64')[-1].strip()
        try:
            # Correct the padding for base64 data if necessary
            padding_needed = len(encoded_data) % 4
            if padding_needed:  # Padding needed
                encoded_data += '=' * (4 - padding_needed)
            decoded_data = base64.b64decode(encoded_data)
            text = decoded_data.decode(charset if charset else 'iso-8859-1')
        except Exception as e:
            print(f"Failed to decode base64: {e}")
        
    #Convert html to plain text
    if '<' in text and '>' in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
    

    #Convert all text to lowercase
    text = text.lower()
    #Removing special whitespace characters
    text = re.sub(r"\s+", " ", text)

    #Trim whitespaces
    text = text.strip()

    #Replace phone numbers with placeholder
    phone_pattern = re.compile(
        r'(\+?\d{1,2}[-.\s]?)?'       
        r'(\(?\d{2,4}\)?[-.\s]?)'     
        r'\d{2,3}[-.\s]?'            
        r'\d{3,4}'                    
    )
  #  text = re.sub(phone_pattern, 'PHONE', text)

    url_pattern = r'\b(?:http|ftp|https)://[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/))'
    #text = re.sub(url_pattern, 'URL', text)

    return text

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
