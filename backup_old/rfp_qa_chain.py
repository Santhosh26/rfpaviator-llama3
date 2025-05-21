from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
# from langchain_community.llms import ollama
from langchain.prompts import PromptTemplate
from src.rfpResponder.return_llm_model import get_llm
from langchain_aws import BedrockLLM
def rfp_qa_chain(persist_directory, template, temp):
    modelId = get_llm()
    if temp:
        temprature = 1
    else:
        temprature = 0

    # Setup the embeddings function and ChromaDB vector store
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    llm = BedrockLLM(
        credentials_profile_name="default", model_id=modelId, model_kwargs={"temperature": 1}
    )
    # Setup the retriever from the ChromaDB vector store
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'lambda_mult': 0.1})
    # retriever = vectordb.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.05})
    # system_message_template = SystemMessagePromptTemplate.from_template(template=template)
    prompt = PromptTemplate.from_template(template=template) #ChatPromptTemplate.from_template([system_message_template])
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm,                #openai.OpenAI(), gpt-4-1106-preview, gpt-3.5-1106 
                                            retriever=retriever,chain_type_kwargs={"verbose":True, "prompt": prompt}) #,chain_type_kwargs={"verbose":True}, chain_type_kwargs={"prompt": template})
    
    qa_chain.return_source_documents = True
    
    # qa_chain.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template=template
        
    # print(qa_chain)
    return qa_chain


