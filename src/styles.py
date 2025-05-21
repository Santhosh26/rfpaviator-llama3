import streamlit as st

def custom_styles():
    st.markdown("""
    <style>
    footer {
        visibility: hidden;
    } 
    #stDecoration {
        # display:none;} 
    .block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
    </style>
    """, unsafe_allow_html=True)
