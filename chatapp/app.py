"""
Chat with your PDFs locally
"""

import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from POE import load_chat_id_map, clear_context, send_message, get_latest_message, set_auth

#Auth
set_auth('Quora-Formkey','f72407929b50a310a71f0bfab73594cc')
set_auth('Cookie','m-b=fo4BP-aQf5SXPYGN7sDb3A==')

bots = {1:'capybara', 2:'beaver', 3:'a2_2', 4:'a2', 5:'chinchilla', 6:'nutria'}
bot = bots[int(1)]
print("The selected bot is : ", bot)

chat_id = load_chat_id_map(bot)
clear_context(chat_id)
print("Context is now cleared")



def query_formatter(docs, query):
    """formate the queries"""
    return f""" Use only below 'Context' and find the answer to the 'Question'. '
    Be precise and to the point while answering. format your answers in markdown where necessary.
    'Context: {docs}'
    'Question: {query}'"""


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by https://github.com/programmingbuddah')


def main():
    st.header("Chat with PDF")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        with st.spinner("Processing..."):
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size = 1000,
                chunk_overlap  = 200,
                length_function = len,
            )

            chunks = text_splitter.split_text(text=text)


            
            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    vector_store = pickle.load(f)
                st.write("Embeddings loaded from Disk")
            else:
                embedding = HuggingFaceEmbeddings(
                    model_name = "sentence-transformers/all-mpnet-base-v2",
                    model_kwargs = {'device': 'cpu'},
                    encode_kwargs = {'normalize_embeddings': False}
                )

                vector_store = FAISS.from_texts(chunks, embedding=embedding)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(vector_store, f)
                st.write("Embedding Computation completed.")

            query = st.chat_input("Say something")
            if query:
                with st.chat_message("user"):
                    st.write(query)
            if query:
                docs = vector_store.similarity_search(query=query)
                message = query_formatter(docs=docs, query=query)
                with st.spinner("Getting your answer..."):
                    send_message(message,bot,chat_id)
                    reply = get_latest_message(bot)
                    with st.chat_message("assistant"):
                        st.write(reply)







if __name__ == '__main__':
    main()