from langchain_huggingface import HuggingFaceEmbeddings
from langchain_aws import BedrockLLM
from langchain_chroma import Chroma
from src.rfpResponder.return_llm_model import get_llm
# from return_llm_model import get_llm
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from langchain_core.runnables import RunnablePassthrough

# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = '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'
# os.environ['LANGCHAIN_PROJECT'] = 'Comp_analysis'

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rfp_qa_chain(persist_directory, template, temp):
        try:
            
            modelId = get_llm()
            
            temperature = 1 if temp else 0  

        # Setup the embeddings function and ChromaDB vector store
            try:
                
                embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
                vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            except Exception as e:
                print(f"Error setting up embeddings or vector store: {e}")
                return None

        # Initialize BedrockLLM
            try:
                
                llm = BedrockLLM(
                    credentials_profile_name="default", 
                    model_id=modelId, 
                    model_kwargs={"temperature": temperature}, 
                    region='us-west-2'
                )
            except Exception as e:
                print(f"Error initializing BedrockLLM: {e}")
                return None

            # Setting up retriever and prompt
            try:
                
                retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'lambda_mult': 0.1})
                prompt = ChatPromptTemplate.from_template(template)
            except Exception as e:
                print(f"Error setting up retriever or prompt template: {e}")
                return None

            # Construct the QA chain
            try:
                
                qa_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                return qa_chain
            except Exception as e:
                print(f"Error constructing QA chain: {e}")
                return None

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        

# qa_chain = rfp_qa_chain(persist_directory="chromastore\\NetIQ_RFP", template="""You are an expert Identity and access management with extensive experience in Opentext NetIQ. Answer the following question based on the provided context. If you don't have enough information to answer, state that you don't know.

# Instructions to be followed:

# Begin your response with either "Yes," or "No," (without any additional characters or formatting), depending on the sentiment of your answer. Use "Yes," for positive sentiment and "No," for negative sentiment.
# Always provide an explanation for your answer.
# Do not mention that your answer is based on retrieved information.
# Stick to answering the specific question asked without engaging in extended discussions.
# Do not direct users to external documentation or sources.
# Do not include any formatting characters (like asterisks, underscores, or markdown syntax) in your response.
# Context: {context}  

# Question: {question}

# Answer:""", temp=None)

# # To use the chain, ensure that 'invoke' method is correctly called with a dictionary that has a 'question' key.
# abc= "Does NetIQ Implementation and Integration: Integrate IAM with existing systems such as HR systems, ERP, and other enterprise applications."

# response = qa_chain.invoke(abc)

# print("Response:", response)
