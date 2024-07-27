import streamlit as st

import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests import get
import pandas as pd

import langchain
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_groq import ChatGroq
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from docx import Document
import time

from langchain.document_loaders import UnstructuredURLLoader

st.title('Live News Summarizer')


st.sidebar.write('Get your groq API key at https://groq.com/')



GROQ_API = st.sidebar.text_input("ENTER YOUR GROQ API KEY (IT'S FREE!!)",type="password")

if st.button('Generate summaries'):
    with st.spinner('Hold tight, generating summary...'):
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive'}


        html_content = get('https://www.livemint.com/',headers=headers)
        soup = BeautifulSoup(html_content.text, 'html.parser')



        urls = []
        urls_html = soup.find_all('h3')
        for url in urls_html:
            # print(url)
            x = url.find_all('a')
            for i in x:
                urls.append(i.get('href'))



        try:
            llm = ChatGroq(api_key = GROQ_API,model_name ='llama3-8b-8192',max_tokens=10000,temperature=0.1)
        except:
            st.write('enter valid groq api key')
            exit()



        def summarize(chunks):

            chunks_prompt = '''
                You are a expert in news summarization. Generate a summary of the following news article,keeping all the important details.\n
                Don't write anything which is not provided in the news article: \n {text}.

                '''
            prompt_template = PromptTemplate(input_variables=['text'],
                                        template=chunks_prompt)
                
            final_prompt = """
            The following is set of summaries:

            {text}.\n\n

            Generate a comprehensive summary from these set of summaries in form of points. Make sure to include every important point \n.
            Helpful Answer:"""

            final_template = PromptTemplate(input_variables=['text'],template=final_prompt)

            summary_chain = load_summarize_chain(llm=llm ,
                                        chain_type= 'map_reduce', 
                                        verbose=False,
                                        map_prompt = prompt_template, # prompt for each chunk
                                        combine_prompt = final_template #final prompt
                                                ) 
            return summary_chain.invoke(chunks)

        try:
            loader = UnstructuredURLLoader(urls=urls[0:15])
            doc = loader.load()
            summary = summarize(doc)
            st.write(summary['output_text'])
            with st.spinner('loading more news...'):
                time.sleep(20)
                loader = UnstructuredURLLoader(urls=urls[15:30])
                doc = loader.load()
                summary = summarize(doc)
                st.write(summary['output_text'])
            with st.spinner('loading more news...'):
                time.sleep(25)
                loader = UnstructuredURLLoader(urls=urls[30:45])
                doc = loader.load()
                summary = summarize(doc)
                st.write(summary['output_text'])
                st.write(urls)
            with st.spinner('loading more news...'):
                time.sleep(30)
                loader = UnstructuredURLLoader(urls=urls[45:60])
                doc = loader.load()
                summary = summarize(doc)
                st.write(summary['output_text'])
                st.write(urls)
            with st.spinner('loading more news...'):
                time.sleep(35)
                loader = UnstructuredURLLoader(urls=urls[60:75])
                doc = loader.load()
                summary = summarize(doc)
                st.write(summary['output_text'])
                st.write(urls)
            with st.spinner('loading more news...'):
                time.sleep(40)
                loader = UnstructuredURLLoader(urls=urls[75:90])
                doc = loader.load()
                summary = summarize(doc)
                st.write(summary['output_text'])
                st.write(urls)
        except:
            st.write('Oops, something wrong happened')
        
        

    



# import streamlit as st

# import requests
# import pandas as pd
# from bs4 import BeautifulSoup
# from requests import get
# import pandas as pd

# import langchain
# from langchain.llms import Cohere
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains.summarize import load_summarize_chain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
# from langchain_groq import ChatGroq
# import re
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from docx import Document
# import time

# from langchain.document_loaders import UnstructuredURLLoader

# st.title('Live News Summarizer')


# st.sidebar.write('Get your groq API key at https://groq.com/')
# # with st.spinner('Please wait...'):
# #     time.sleep(5)


# # st.sidebar.title('Customize Your Summary')
# # st.sidebar.markdown("**Choose Summary Length:**")
# # max_tokens = st.sidebar.select_slider(
# #     'For a very crisp summary, choose 1000 tokens.\n'
# #     'For a concise two-liner, choose 5000 tokens.\n'
# #     'For a detailed summary, choose 10000 tokens.',
# #     options=[1000, 5000, 10000],
# #     format_func=lambda x: f"{x} tokens"
# # )


# GROQ_API = st.sidebar.text_input("ENTER YOUR GROQ API KEY (IT'S FREE!!)",type="password")


# if st.button('Generate summaries'):
#     with st.spinner('Hold tight, generating summary...'):
#         headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
#                 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#                 'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
#                 'Accept-Encoding': 'none',
#                 'Accept-Language': 'en-US,en;q=0.8',
#                 'Connection': 'keep-alive'}

#         html_content = get('https://www.thehindu.com/sci-tech/technology/',headers=headers)

#         soup = BeautifulSoup(html_content.text, 'html.parser')

#         urls = []
#         titles = []
#         urls_html = soup.find_all('h3',class_='title')
#         for url in urls_html:
#             # print(url)
#             x = url.find_all('a')
#             for i in x:
#                 urls.append(i.get('href'))


#         loader = UnstructuredURLLoader(urls=urls)
#         doc = loader.load()

#         # GROQ

#         # GROQ_API = st.sidebar.text_input("ENTER YOUR GROQ API KEY (IT'S FREE!!)",type="password")
    
#         # st.slider(label = 'Select the ')

#         try:
#             llm = ChatGroq(api_key = GROQ_API,model_name ='llama3-8b-8192',max_tokens=10000,temperature=0.1)
#         except:
#             st.write('enter valid groq api key')
#             exit()


#     # llm = ChatGroq(api_key = GROQ_API,model_name ='llama3-8b-8192',max_tokens=2000,temperature=0.1)
#     # llm = ChatGroq(api_key = GROQ_API, model_name='gemma2-9b-it', max_tokens=1000,temperature=0.1)


#     # Embedding model
#     # hf = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


#     # def pdf_loading(path): 

#     #     loader = PyPDFLoader(path)
#     #     docs = loader.load()
#     #     return docs


#     # # Remove excessive tabs and newlines
#     # def clean(text):
#     #     cleaned_text = re.sub(r'\t+', ' ', text)  # Replace multiple tabs with a single space
#     #     cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  # Replace multiple newlines with a single newline
#     #     return cleaned_text

#     # def apply_cleaning(docs):
#     #     for i in range(0,len(docs)):
#     #         docs[i].page_content = clean(docs[i].page_content)
#     #     return docs
        
#     # def create_chunks(docs):
#     #     chunks = RecursiveCharacterTextSplitter(separators=['\n','.'],chunk_size=500,
#     #     chunk_overlap=50).split_documents(docs)
#     #     return chunks


#     # def initialize(path):
#     #     docs  = pdf_loading(path)
#     #     docs = apply_cleaning(docs)
#     #     chunks  = create_chunks(docs)
#     #     print(f'len chunks = {len(chunks)}')
#         # return chunks


#         def summarize(chunks):

#             chunks_prompt = '''
#                 You are a expert in news summarization. Generate a summary of the following news article,keeping all the important details.\n
#                 Don't write anything which is not provided in the news article: \n {text}.

#                 '''
#             prompt_template = PromptTemplate(input_variables=['text'],
#                                         template=chunks_prompt)
                
#             final_prompt = """
#             The following is set of summaries:

#             {text}.\n\n

#             Generate a comprehensive summary from these set of summaries in form of points. Make sure to include every important point \n.
#             Helpful Answer:"""

#             final_template = PromptTemplate(input_variables=['text'],template=final_prompt)

#             summary_chain = load_summarize_chain(llm=llm ,
#                                         chain_type= 'map_reduce', 
#                                         verbose=False,
#                                         map_prompt = prompt_template, # prompt for each chunk
#                                         combine_prompt = final_template #final prompt
#                                                 ) 
#             return summary_chain.invoke(chunks)


#         summary = summarize(doc)
#         st.write(summary['output_text'])
#         st.write(urls)
    
