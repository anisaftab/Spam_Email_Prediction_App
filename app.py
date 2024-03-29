import streamlit as st

def main():
    st.title("Email Analysis")
    
    # Text input for email
    email = st.text_area("Enter email here")
    
    # Button to analyze email
    if st.button("Analyze"):
        # Perform analysis and calculate spam percentage
        spam_percentage = analyze_email(email)
        
        # Display spam percentage
        st.write(f"Spam Percentage: {spam_percentage}%")

def analyze_email(email):
    # Add your email analysis logic here
    # This is just a placeholder
    return 75

if __name__ == "__main__":
    main()