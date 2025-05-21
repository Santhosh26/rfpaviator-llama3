import streamlit as st
from login import show_login_page
from RFPAvaitor import rfpmain as home_main
from dotenv import load_dotenv
from src.styles import custom_styles

st.set_page_config(
    page_title="Opentext RFP Avaitor", layout="wide"
    )
# Check if the user is authenticated
def check_authentication():

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if st.session_state["authenticated"]:
        # User is authenticated, display the main app
        home_main()
        custom_styles()
        # Load env variables
        load_dotenv()
    else:
        # User is not authenticated, display the login page
        show_login_page()

if __name__ == "__main__":
    custom_styles()
    check_authentication()
