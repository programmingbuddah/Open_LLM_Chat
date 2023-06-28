import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle



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


            embedding = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-mpnet-base-v2",
                model_kwargs = {'device': 'cpu'},
                encode_kwargs = {'normalize_embeddings': False}
            )

            store = FAISS.from_texts(chunks, embedding=embedding)
            store_name = pdf.name[:-4]
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(store, f)




            st.write(chunks)





if __name__ == '__main__':
    main()