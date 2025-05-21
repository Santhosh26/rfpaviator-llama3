import os
import sys
import traceback
from langchain_community.document_loaders import PDFMinerLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import chroma

def main():
    if len(sys.argv) > 1:
        root_directory_path = sys.argv[1]  # The root directory from command-line argument
    else:
        print("Please provide the directory path as a command-line argument.")
        sys.exit(1)

    for root, dirs, files in os.walk(root_directory_path):
        # Skip the root directory itself for creating a vector store
        if root == root_directory_path:
            continue
        process_files(root, files)

def process_files(directory_path, files_in_directory):
    # Base directory for chromastore
    base_persist_dir = "./chromastore/test/"
    directory_name = os.path.basename(directory_path)
    persist_dir = os.path.join(base_persist_dir, f"{directory_name}_RFP")

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    for file_type, loader_class in [('.pdf', PDFMinerLoader), ('.docx', Docx2txtLoader), ('.xlsx', UnstructuredExcelLoader), ('.txt', TextLoader)]:
        if any(file.endswith(file_type) for file in files_in_directory):
            loader = DirectoryLoader(directory_path, glob=f'*{file_type}', loader_cls=loader_class, loader_kwargs={'encoding': 'UTF-8'} if file_type == '.txt' else {})
            doc = loader.load()
            chunks = all_splitter(doc)
            clean_chunks = clean_chunk(chunks)

            chroma.Chroma.from_documents(documents=clean_chunks, embedding=embeddings, persist_directory=persist_dir)
            print(f"Processed and stored {file_type} files in {directory_path}")
        else:
            print(f"No new {file_type} files detected in {directory_path}")

def all_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    return chunks

def clean_chunk(chunks):
    for chunk in chunks:
        updated_content = chunk.page_content.replace('\u200b', '')
        chunk.page_content = updated_content
    return chunks

if __name__ == "__main__":
    main()
