from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import re

from langchain_community.callbacks import get_openai_callback

def process_cb_openai(callbacks_list):
    # print(f"cb: {callbacks_list}")
    total_tokens_used = 0
    total_cost = 0.0

    for cb in callbacks_list:
        # Extract tokens used and cost from the callback output
        tokens_used_match = re.search(r"Tokens Used: (\d+)", cb)
        cost_match = re.search(r"Total Cost \(USD\): \$(\d+\.\d+)", cb)

        if tokens_used_match and cost_match:
            total_tokens_used += int(tokens_used_match.group(1))
            total_cost += float(cost_match.group(1))
    
    total_token_cost_message= f"Total tokens used for RFP response: {total_tokens_used} and total cost is: USD{total_cost:.4f}"
    return total_token_cost_message

def pretty_print_docs(docs):
    doclen =len(docs)
    print("document lenght", doclen )
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def create_retrieval_qa_chain(persistent_directory, question):
    # Setup the embeddings function and ChromaDB vector store
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = chroma.Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Setup the retriever from the ChromaDB vector store
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'lambda_mult': 0.1, 'k': 20})

    compressor = LLMChainExtractor.from_llm(OpenAI(temperature=0))
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    # Initialize the OpenAI language model for GPT-4

    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)  # Adjust engine if necessary
    template = f"""You are an expert Applciation Security architect with extensive experience in Opentext Fortify.
         Your task is to answer questions using the context provided. 
         In cases where you lack sufficient information in the provided context, simply respond with 'I don't know.' 
         Don't attempt to conjecture or fabricate responses. 
         When answering, begin your responses with either 'Yes,' or 'No,' depending on sentiment of the response. 
         Offer an explanation related to your answer. 
         Refrain from engaging in extended conversations or discussions beyond the scope of the direct question. 
         Don't direct users to documentation or other sources.
         In your answer or response, dont mention about documentation references.
         In your answer or response, dont mention about datasheet references.
         In your answer or response, dont mention about provided context references.
        Question: {{question}}
        Context: {{context}}
        """
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    qa_chain.return_source_documents = True  # Ensure source documents are returned
    qa_chain.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template=template


    #Contextual retriver

        
    # compressor = LLMChainExtractor.from_llm(OpenAI(temperature=0))
    # compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    # docs = compression_retriever.get_relevant_documents(question)
    # pretty_print_docs( docs)
    return qa_chain

def answer_question(qa_chain, question):
    # Invoke the QA chain to get the answer and retrieved documents
    callbacks_list = []
    with get_openai_callback() as cb: 
        response = qa_chain.invoke({'query': question})
        
        answer = response['result']
    callbacks_list.append(str(cb))
    print(process_cb_openai(callbacks_list))
    # docs = response.get('source_documents', [])
    
    # # Print the retrieved documents
    # print("Retrieved Documents:", docs)
    # for doc in docs:
    #     print("inside for")
    #     print(doc.page_content)  # Adjust according to the actual structure of `doc`
    
    # Print the answer
    print("\nAnswer:", answer)

if __name__ == "__main__":
    # After creating the QA chain
    product = "Fortify"  # or other products as per your requirements
    persistent_directory = "chromastore\Fortify_RFP"
    question = "List all Fortify supported programming languages"
    
    qa_chain = create_retrieval_qa_chain(persistent_directory=persistent_directory,question=question)
    # print("qachain:", qa_chain)
    # customize_prompt_template(qa_chain, product)
    answer_question(qa_chain, question)
