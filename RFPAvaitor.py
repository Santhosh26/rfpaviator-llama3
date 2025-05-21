
import time
import streamlit as st
from src.rfpResponder.rfpResponder import process_csv, download_csv, process_questions
from src.rfpResponder.choose_datastore import select_datastore
from src.rfpResponder.rfp_upload_message import upload_instructions
from src.compare_comp.Asses_Competetion import asses_competition
from src.rfpResponder.chooseProdcut import showProductgroup, showProducts, solutionSelection


# Consent display function
def display_consent():

    with st.container():  # Ensure that consent controls are grouped together
        
        for i in range(12):
            st.write("\n")
        st.header("Consent")
        st.write(f"""
        
                 
        Dear {st.session_state['fullname']},

        Before proceeding, please carefully review the following terms:

        1. **Sensitive Data**: We strongly advise against uploading any sensitive or confidential data to this platform. While we strive to maintain the security and privacy of your data, it's always best to err on the side of caution.

        2. **Redaction of Customer Information**: If you're uploading RFP requirement questions or any related documents, please ensure you've taken the necessary precautions to remove any sensitive customer information or other confidential details from the documents.
                 
        3. **Check your facts**: While we have safeguards, RFP Avaitor may give you inaccurate information. It’s not intended to give advice.

        By clicking "Accept", you acknowledge that you've read and understood the above terms, and you agree to take full responsibility for the data you upload.
        """)
        # Consent buttons
        accept = st.button("Accept", type="primary"  )
        decline = st.button("Decline")

        if accept:
            st.session_state['accepted'] = True
            st.rerun()

        if decline:
            st.session_state['accepted'] = False
            st.error("You must accept the terms and conditions to proceed.")


def add_logo():
    st.image(image='images\Aviator.png')
    # st.markdown(
    #     """
    #     <style>
    #         [data-testid="stSidebarNav"] {
    #             background-image: url('https://media-s3-us-east-1.ceros.com/open-text/images/2023/08/15/679f819f608f65ad8522ee934c399ecc/opentext-ai-aviator-logo-for-srg.png');
    #             background-repeat: no-repeat;
    #             background-size: 310px auto; 
    #             padding-top: 250px;
    #             background-position: 20px 20px;
    #         }
    #         [data-testid="stSidebarNav"]::before {
    #             content: "Opentext RFP Avaitor";
    #             margin-left: 20px;
    #             margin-top: 20px;
    #             font-size: 25px;
    #             position: relative;
    #             top: 50px;
    #         }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

def showlogout_last_active():
    # Session timeout check
    if "last_active" in st.session_state and time.time() - st.session_state.get("last_active", 0) > 1500:
        st.session_state["authenticated"] = False
        st.session_state['accepted'] = False
        st.warning("Session timed out. Please log in again.")
        st.rerun()
    else:
        with st.sidebar:
            add_logo()
            st.write(f"Welcome, {st.session_state['fullname']}!" )
        

        # Logout button in sidebar
        if st.sidebar.button("Logout"):
            logout_user()

    # Update last active time on user interaction
    st.session_state["last_active"] = time.time()

def logout_user():
    st.session_state["authenticated"] = False
    st.rerun()


def rfpmain():

    
    if st.session_state.get('accepted', False):
        # Initialize session state variables if they don't exist
        if 'responses_df' not in st.session_state:
            st.session_state['responses_df'] = None
        if 'questions' not in st.session_state:
            st.session_state['questions'] = []
        if 'product_percentages_df' not in st.session_state:
            st.session_state['product_percentages_df'] = None
        if 'uploaded_csv' not in st.session_state:
            st.session_state['uploaded_csv'] = None  

        if st.session_state['accepted']:
            showlogout_last_active() 
            for i in range(6):

                st.write("\n") 
            st.subheader("RFP Aviator :sunglasses: ")
            st.markdown("<div style="">Your AI assistant, helps respond to your RFP questionnaires</div>&nbsp;", unsafe_allow_html=True)
            upload_instructions()
            #fetch prodcutGroupList
            prodcutGroupList = showProductgroup()
            selected_prodcutgroup = st.selectbox("Choose your Product Group", prodcutGroupList, index=None)

            if selected_prodcutgroup:
                productsList = showProducts(selectedProductGroup=selected_prodcutgroup)
                product = st.selectbox("choose your product..", productsList, index=None)

                if product:
                    solutions = solutionSelection(selectedProduct=product)

                    #tempcode remove later
                    if solutions is None:
                        solutions =[]
                    
                    num_rows = (len(solutions) + 5) // 6  # Calculate number of rows needed, up to 4 solutions per row
                    
                    selectedSolutions = []  # Initialize the list outside the loop

                    for i in range(num_rows):
                        # Calculate start and end indices for the solutions in the current row
                        start_idx = i * 4
                        end_idx = min(start_idx + 6, len(solutions))  # Ensure end_idx does not exceed the list length
                        
                        # Create a row with up to 4 columns
                        cols = st.columns(6)
                        
                        # Place a checkbox for each solution in the row
                        for j in range(start_idx, end_idx):
                            with cols[j % 6]:  # Use modulo to wrap column index back to 0-3
                                # Create a checkbox and add the solution to selectedSolutions if checked
                                is_selected = st.checkbox(solutions[j], key=solutions[j])
                                
                                if is_selected:
                                    selectedSolutions.append(solutions[j])
                    
                    
                    
                    st.session_state['uploaded_csv'] = st.file_uploader("CSV file only", type='csv', key=1)  
                    btDetailed = st.toggle('Detailed RFP Response', False)

                        

                    st.subheader("", divider="blue")
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        with st.container():
                            st.caption("RFP Responder")
                            if st.session_state['uploaded_csv']:  
                                if st.button("Analyze & Respond", key="analyze_respond"):
                                    start_time = time.time()
                                    st.session_state['questions'] = process_csv(uploaded_csv=st.session_state['uploaded_csv'])  
                                    if st.session_state['questions']:
                                        datastore = select_datastore(selected_option=product)
                                        st.session_state['responses_df'], st.session_state['pos_compliance_percentage'] = process_questions(persist_directory=datastore, btDetailed=btDetailed, start_time=start_time, questions=st.session_state['questions'], product=product, solutions=selectedSolutions)

                                if st.session_state['responses_df'] is not None and not st.session_state['responses_df'].empty:
                                    if st.session_state['pos_compliance_percentage'] >  60:

                                        st.success(f"{product} Complaince percentage: {st.session_state['pos_compliance_percentage']}%" )
                                    else:
                                        st.warning(f"{product} Complaince percentage: {st.session_state['pos_compliance_percentage']}%")
                                    st.table(st.session_state['responses_df'])
                                    csv_string = st.session_state['responses_df'].to_csv(index=False)
                                    download_csv(csv_string)

                    # with col2:
                    #     with st.container():
                    #         st.caption("Competitive Analysis - coming soon")
                    #         # if st.session_state['uploaded_csv'] and st.button("Analyze Competition", key="analyze_competition"):
                    #         #     st.session_state['questions'] = process_csv(uploaded_csv=st.session_state['uploaded_csv'])  
                    #         #     if st.session_state['questions']:
                    #         #         time_taken, st.session_state['product_percentages_df'] = asses_competition(questions=st.session_state['questions'], selected_option=product)
                    #         #         st.write(f"Time taken: {time_taken:.2f} minutes")

                    #         #     if st.session_state['product_percentages_df'] is not None:
                    #         #         if st.session_state['pos_compliance_percentage'] > 49:
                    #         #             st.success(f"{product} Complaince percentage: {st.session_state['pos_compliance_percentage']}%" )
                    #         #         else:
                    #         #             st.warning(f"{product} Complaince percentage: {st.session_state['pos_compliance_percentage']}%")
                    #         #         st.table(st.session_state['product_percentages_df'])
            
            else:
                st.warning("Select an option from the above drop down")


    else:
        display_consent()
