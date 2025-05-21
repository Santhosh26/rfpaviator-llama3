
import time
import pandas as pd
import streamlit as st
from src.clean_text import clean_text
from src.rfpResponder.newrfp_qa_chain import rfp_qa_chain
from src.rfpResponder.format_query import format_questions
from src.rfpResponder.response_status import status_checker
from src.rfpResponder.productCompalincePercentage import productCompalincePercentage
from src.rfpResponder.chooseTemplate import returnTemplate

import os
import uuid
    
def download_csv(csv_string):
    st.download_button(
    label="Download response.csv", data=csv_string, file_name="responses.csv", mime="text/csv")

# def download_xlsx(xlsx):
    
#     ste.download_button("Download responses.xlsx", xlsx, "responses.xlsx")


    


def save_df_as_xlsx(df, product_name, directory_path="generatedResponses"):
    # Ensure the base directory exists
    base_dir = os.path.join(directory_path, product_name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Generate a random filename
    file_name = f"{uuid.uuid4()}.xlsx"
    file_path = os.path.join(base_dir, file_name)
    
    # Save the dataframe to an Excel file
    df.to_excel(file_path, index=False)
    
    return file_path

def process_questions(persist_directory, btDetailed, start_time, questions, product, solutions):
    
    responses_list = [] 
    promptTemplate = returnTemplate(product=product, btDetailed=btDetailed)
    qa_chain = rfp_qa_chain(persist_directory=persist_directory, template=promptTemplate, temp=btDetailed) 
    
    # Remove special symbols from each question
    questions = [clean_text(question) for question in questions]
    formated_questions = [format_questions(question, product=product, solutions=solutions) for question in questions]
    
    #Progress tracking 
    Progress_text = "Please wait while processing the response.."
    Progress_bar = st.progress(0, text=Progress_text)
    Progress_Count = len(questions)
    current_count = 0
    

    for formatted_query, original_question in zip(formated_questions, questions):
    # Code to process each question using LLM
        
        try:
            response = qa_chain.invoke(formatted_query)

        except Exception as e:
            print(f"Error invoking qa_chain: {e}")
            response = None
            st.error("An error occurred while processing the question.")
            continue
        # docs = response.get('source_documents', [])
        
        # # Print the retrieved documents
        # print("Retrieved Documents:", docs)
        # for doc in docs:
        #     print("retrived docs::")
        #     print(doc.page_content) 
        result = response
     
        status = status_checker(result)
       


        responses_list.append({
            "Question": original_question,
            "Answer": result,
            "Status": status
            })

     
        #progress bar calculations
        current_count += 1
        progress_percentage = int((current_count / Progress_Count) * 100)
        Progress_bar.progress(progress_percentage, text=Progress_text)
        
        #Add a small delay for better visualization of progress bar 
        time.sleep(0.01)

    # to extract status list and store positive percentage
    status_list = [response["Status"] for response in responses_list] 
    complaincePercentage = productCompalincePercentage(status_list=status_list)
    pos_complaincePercentage = complaincePercentage[0]
    
    # End timing
    end_time = time.time()


    # Calculate the total time taken
    total_time_seconds = end_time - start_time

    # Convert the total time from seconds to minutes
    total_time_minutes = total_time_seconds / 60
    
    # Convert the list of dictionaries into a DataFrame
    responses_df = pd.DataFrame(responses_list)
    save_df_as_xlsx(df=responses_df,product_name=product )
    #print time taken to finish operation
    st.text(f"Time taken for generating responses: {total_time_minutes:.2f} minutes.")

    return responses_df, pos_complaincePercentage 
    
def process_csv(uploaded_csv):
    if uploaded_csv is None:
        st.error("No CSV file uploaded.")
        return None

    if uploaded_csv.size == 0:
        st.error("The uploaded CSV file is empty.")
        return None

    df = pd.read_csv(uploaded_csv, encoding='utf-8')

    # Convert the first row of the DataFrame to lowercase to uniformity
    first_row_lower = df.columns[0].lower()
    # Check if the first row contains 'question' or 'questions'
    if 'question' not in first_row_lower and 'questions' not in first_row_lower:
        st.error("The uploaded CSV does not have a 'question' or 'questions' column in the first row.")
        return None

    questions = df['question'].dropna().tolist()  # Using dropna() to remove NaN values

    if not questions:
        st.error("The 'question' column contains no questions.")
        return None

    return questions






        