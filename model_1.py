# from langchain_community.document_loaders import BSHTMLLoader
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.document_loaders import DirectoryLoader
# import os

# os.environ['HUGGING_FACE_TOKEN']="HUGGING_FACE_TOKEN"

# # # Load HTML files from the folder
# loader = DirectoryLoader(
#     "supremecourt",  # Specify the folder path
#     glob="*.htm",            # Use glob to load only .htm files
#     loader_cls=BSHTMLLoader  # Use BSHTMLLoader for parsing HTML
# )


# # Load documents from the directory
# documents = loader.load()

# # Split the documents 
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# split_docs = text_splitter.split_documents(documents)

# # Embed documents
# embeddings=HuggingFaceBgeEmbeddings(
#     model_name="intfloat/multilingual-e5-large-instruct",
#     model_kwargs={'device':'cpu'},
#     encode_kwargs={'normalize_embeddings':True}
# )
# docsearch = Chroma.from_documents(split_docs, embeddings)

# # Now you can query your documents, for example:
# query = "WHAT IS THE JUDGEMENT SUMMARY OF THIS CASE?"
# results = docsearch.similarity_search(query)

# # Display results
# for result in results:
#     print(result.page_content)

# from langchain_community.document_loaders import BSHTMLLoader
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.document_loaders import DirectoryLoader
# import os

# # Load Metadata
# metadata_file_path = "sc_all_metadata"

# def load_metadata(file_path):
#     metadata_dict = {}
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             # Only process lines that contain a colon
#             if ":" in line:
#                 filename, metadata = line.strip().split(":", 1)
#                 metadata_dict[filename] = metadata
#             else:
#                 # Handle lines without the expected structure (e.g., skip or log them)
#                 print(f"Skipping malformed line: {line.strip()}")
#     return metadata_dict


# metadata_dict = load_metadata(metadata_file_path)

# # Load HTML files from the folder
# loader = DirectoryLoader(
#     "supremecourt",  # Specify the folder path
#     glob="*.htm",    # Use glob to load only .htm files
#     loader_cls=BSHTMLLoader  # Use BSHTMLLoader for parsing HTML
# )

# # Load documents from the directory and attach metadata
# documents = loader.load()
# for doc in documents:
#     filename = doc.metadata["source"]  # Assuming `source` contains the filename
#     if filename in metadata_dict:
#         doc.metadata["metadata"] = metadata_dict[filename]  # Add metadata

# # Split the documents 
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# split_docs = text_splitter.split_documents(documents)

# # Embed documents
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name="intfloat/multilingual-e5-large-instruct",
#     model_kwargs={'device':'cpu'},
#     encode_kwargs={'normalize_embeddings':True}
# )

# docsearch = Chroma.from_documents(split_docs, embeddings)

# # Now you can query your documents, including their metadata
# query = "WHAT IS THE JUDGEMENT SUMMARY OF THIS CASE?"
# results = docsearch.similarity_search(query)

# # Display results with metadata
# for result in results:
#     print(result.page_content)
#     print(f"Metadata: {result.metadata}")

# from langchain_community.document_loaders import BSHTMLLoader
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.document_loaders import DirectoryLoader
# import os

# # Load Metadata
# metadata_file_path = "sc_all_metadata"

# def load_metadata(file_path):
#     metadata_dict = {}
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             # Split the line by commas, while ensuring to strip any extra spaces
#             fields = [field.strip('"') for field in line.strip().split(",")]
#             if len(fields) > 16:  # Ensure there are enough fields
#                 # Use the first field (case ID or similar) as the key and the rest as metadata
#                 metadata_dict[fields[0]] = {
#                     "court": fields[1],
#                     "case_type": fields[2],
#                     "case_no": fields[3],
#                     "date": fields[4],
#                     "parties": fields[5],
#                     "counsel": fields[6],
#                     "citation": fields[7],
#                     "judges": fields[8],
#                     "statutes": fields[9],
#                     "bench": fields[10],
#                     "status": fields[11],
#                     "url": fields[12],
#                     "other_info": fields[13],
#                     "no_of_judges": fields[14],
#                     "petitioner": fields[15],
#                     "respondent": fields[16]
#                 }
#             else:
#                 print(f"Skipping malformed line (not enough fields): {line.strip()}")
#     return metadata_dict

# metadata_dict = load_metadata(metadata_file_path)

# # Load HTML files from the folder
# loader = DirectoryLoader(
#     "supremecourt",  # Specify the folder path
#     glob="*.htm",    # Use glob to load only .htm files
#     loader_cls=BSHTMLLoader  # Use BSHTMLLoader for parsing HTML
# )

# # Load documents from the directory and attach metadata
# documents = loader.load()
# for doc in documents:
#     filename = doc.metadata["source"]  # Assuming `source` contains the filename
#     case_id = os.path.basename(filename).split('.')[0]  # Extract the case ID from the filename
#     if case_id in metadata_dict:
#         doc.metadata.update(metadata_dict[case_id])  # Add metadata

# # Split the documents 
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# split_docs = text_splitter.split_documents(documents)

# # Embed documents
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name="intfloat/multilingual-e5-large-instruct",
#     model_kwargs={'device':'cpu'},
#     encode_kwargs={'normalize_embeddings':True}
# )

# docsearch = Chroma.from_documents(split_docs, embeddings)

# # Now you can query your documents, including their metadata
# query = "WHAT IS THE JUDGEMENT SUMMARY OF THIS CASE?"
# results = docsearch.similarity_search(query)

# # Display results with metadata
# for result in results:
#     print(result.page_content)
#     print(f"Metadata: {result.metadata}")

from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from dotenv import load_dotenv  # Importing 'load_dotenv' to load environment variables from a .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load Metadata
metadata_file_path = "sc_all_metadata"

def load_metadata(file_path):
    metadata_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = [field.strip('"') for field in line.strip().split(",")]
            if len(fields) > 16:
                metadata_dict[fields[0]] = {
                    "court": fields[1],
                    "case_type": fields[2],
                    "case_no": fields[3],
                    "date": fields[4],
                    "parties": fields[5],
                    "counsel": fields[6],
                    "citation": fields[7],
                    "judges": fields[8],
                    "statutes": fields[9],
                    "bench": fields[10],
                    "status": fields[11],
                    "url": fields[12],
                    "other_info": fields[13],
                    "no_of_judges": fields[14],
                    "petitioner": fields[15],
                    "respondent": fields[16]
                }
            else:
                print(f"Skipping malformed line (not enough fields): {line.strip()}")
    return metadata_dict

metadata_dict = load_metadata(metadata_file_path)

# Load HTML files from the folder
loader = DirectoryLoader(
    "supremecourt",
    glob="*.htm",
    loader_cls=BSHTMLLoader
)

# Load documents from the directory and attach metadata
documents = loader.load()
for doc in documents:
    filename = doc.metadata["source"]
    if filename in metadata_dict:
        doc.metadata.update(metadata_dict[filename])

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

# Embed documents
embeddings = HuggingFaceBgeEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

docsearch = Chroma.from_documents(split_docs, embeddings)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# Query with LLM
query = "What is the judgment summary of Civil Appeal No. 4079 of 2004 case?"
results = docsearch.similarity_search(query)

# Use LLM to generate responses based on the results
for result in results:
    print(result.page_content)
    print(f"Metadata: {result.metadata}")
    
    # Use the LLM to generate a response
    response = llm.invoke(result.page_content)  # Pass the content to the LLM
    print(f"LLM Response: {response}")  # Print the response