from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
import sys
import json

class LLM_Models:
    @staticmethod
    def llama2(user_message):
        try:
            chat_history = []
            query = user_message
            data_path = "data/merged_data_withFinalAspects_1.csv"
            DB_FAISS_PATH = "vectorstore/db_faiss"
            n_gpu_layers = 1
            n_batch = 1000
            model_path = "models/llama-2-7b-chat.Q4_K_S.gguf"
            loader = CSVLoader(file_path=data_path, encoding="utf-8", csv_args={'delimiter': ','})
            data = loader.load()
            #print(data)
            # Split the text into Chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(data)
            #print(len(text_chunks))
            # Download Sentence Transformers Embedding From Hugging Face
            embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
            # COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
            docsearch = FAISS.from_documents(text_chunks, embeddings)
            docsearch.save_local(DB_FAISS_PATH)
            llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                n_ctx=2048,
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                verbose=True, # Verbose is required to pass to the callback manager
                )
            qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
            result = qa({"question":query, "chat_history":chat_history})
            #print(result['answer'])
            chat_history = chat_history.append(result['answer'])
            #print(result)
            response = {"Query": user_message, "Response": result['answer']}
            return response    
        except ValueError:
            raise ValueError('Invalid input. Please provide valid inputs')
