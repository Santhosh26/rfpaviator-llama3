import streamlit as st

  
# function that shows display message for the RFPresponder page
def upload_instructions():
    
    st.markdown("""
        <style>
        .markdown-font {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        .big-font {
            font-size: 14px !important;
            font-weight: 600;
        }
        .upload-instructions {
            background-color: #f0f2f6;
            padding: 3px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 14px !important; /* Reduced font size */
        }
        </style>
        <div class='markdown-font'>
            <div class='big-font'>Please follow these instructions carefully before your upload:</div>
            <div class='upload-instructions'>
                <p></p>
                <ol class='upload-instructions'>
                    <li class='upload-instructions'>Ensure the RFP/RFI queries are in CSV file format.</li>
                    <li class='upload-instructions'>The CSV should have a column named 'question' under which you can input your list of RFP questions. </li>
                    <li class='upload-instructions'>Click 'Analyze & Respond' to submit your queries.</li>
                    <li class='upload-instructions'>Optionally, you can also perform comptetetive analysis</li>
                </ol>
            </div>
        </div>

    """, unsafe_allow_html=True)
    with open("static/questionnaire-sample.csv", "rb") as file:
        btn = st.download_button(
            label="Questionnaire template",
            data=file,
            file_name="questionnaire-sample.csv",
            mime="text/csv"
          )
