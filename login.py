import os
import psycopg2
import hashlib
import time
from contextlib import contextmanager
import streamlit as st

# Use environment variables for credentials
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')

POSTGRES_DB = "rfpaviator"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"

def get_db_connection():
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )
    return conn

@contextmanager
def db_cursor():
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        connection.close()
# # Update the authenticate_user function for PostgreSQL
# def authenticate_user(username, password):
#     """Check if the user's credentials are valid by comparing with the database."""
#     with db_cursor() as cursor:
#         cursor.execute('SELECT password_hash fullname FROM users WHERE user_id = %s', (username,))
#         user_data = cursor.fetchone()
    
#     if user_data:
#         password_hash = hashlib.sha256(password.encode()).hexdigest()
#         return password_hash == user_data[0]
#     return False


def authenticate_user(username, password):
    """Check if the user's credentials are valid by comparing with the database."""
    with db_cursor() as cursor:
        cursor.execute('SELECT password_hash, fullname FROM users WHERE user_id = %s', (username,))
        user_data = cursor.fetchone()
    
    if user_data:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash == user_data[0]:
            return True, user_data[1]  # Return True and fullname
    return False, None


def show_login_page():
    # """Display the login form with layout adjustments."""
    # st.markdown("<h1 style='text-align: center; color: black;'>Welcome to RFP Aviator</h1>", unsafe_allow_html=True)

    
    # Create a two-column layout
    col1, col2 = st.columns([2.5, 2])  # Adjust the ratio as needed

    # Column 1: Logo and Header
    with col1:

        for i in range(12):

          st.write("\n") 
        col3, col4,col5 = st.columns([0.2, 0.5,0.3])
        with col4:
            st.image("images/opentext-blog-aviator-mav-coming-soon-1-1024x617.png", width=630)
            st.markdown("<h2 style='text-align: left; color: black;'>Answer RFPs in record time</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: left; color: black;'>Streamline your RFP processes with our AI-powered platform. We help you answer RFP questionnaires</p>", unsafe_allow_html=True)

    # Column 2: Login Form, centered
    with col2:
        for i in range(12):

          st.write("\n") 
 
        
        with st.form(key='login_form', clear_on_submit=True):
            # You can adjust the spacing by adding empty lines
    
            st.subheader("SignIn")
            username_input = st.text_input("Username", key="username_input")
            password_input = st.text_input("Password", type="password", key="password_input")
            submit_button = st.form_submit_button(label='Login')
            
            # Using markdown to create a mailto link
            st.markdown("<a href='mailto:skumar14@opentext.com?subject=exper for RFP Aviator&body=Hi, I would like to sign up for RFP Aviator. Please let me know the next steps.' style='color:white; background-color: #1a6aff; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; border: none; border-radius: 8px;'>Sign Up</a>", unsafe_allow_html=True)
            if submit_button:
                authenticated, fullname = authenticate_user(username_input, password_input)
                if authenticated:
                    st.session_state["authenticated"] = True
                    st.session_state["username_session"] = username_input  # Consider renaming this key if it's actually storing the user ID
                    st.session_state["fullname"] = fullname  # Store the fullname in session state
                    st.rerun()
                else:
                    st.error("Invalid username or password")
                    st.session_state["authenticated"] = False



def show_authenticated_page():
    """Display page after successful authentication."""
    # Session timeout check
    if "last_active" in st.session_state and time.time() - st.session_state.get("last_active", 0) > 3600:
        
        st.session_state["username_session"] = None
        st.session_state["fullname"] = None
        logout_user()
    
    st.title(f"Welcome, {st.session_state['username_session']}!")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username_session"] = None
        st.rerun()
    
    # Update last active time on user interaction
    st.session_state["last_active"] = time.time()


def logout_user():
    st.session_state["authenticated"] = False
    st.rerun()
# def main():

#     if "authenticated" not in st.session_state:
#         st.session_state["authenticated"] = False
    
#     # The check for session timeout is handled within these functions
#     if st.session_state["authenticated"]:
#         show_authenticated_page()
#     else:
#         show_login_page()

# if __name__ == "__main__":
#     main()
