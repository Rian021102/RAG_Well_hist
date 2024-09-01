import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import streamlit as st

# Title
st.title("CHAT WITH YOUR DOCUMENTS")
st.header("Pick a Wellbore for Chatting")
api_key = os.environ["MISTRAL_API_KEY"]

def get_data(path):
    df=pd.read_csv(path)
    df=df[['Wellbore','Remark']]
    return df

def get_wellbore(df):
    wellbore = st.sidebar.selectbox('Select Wellbore', df['Wellbore'].unique())
    st.write(wellbore)
    #when selected, create a new dataframe with only the selected wellbore
    wellbore_df = df[df['Wellbore'] == wellbore].copy()
    st.write(wellbore_df)
    return wellbore_df

def convert_to_text(wellbore_df):
    #drop Wellbore columns
    wellbore_df.drop(columns='Wellbore', inplace=True)
    #convert to text without header
    path='/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/remark_test.txt'
    with open(path, 'a') as f:
        df_string = wellbore_df.to_string(header=False, index=False)
        f.write(df_string)

def main():
    path=('/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/drilling_report_comp.csv')
    df=get_data(path)
    wellbore_df = get_wellbore(df)
    text=convert_to_text(wellbore_df)
    # Split text into chunks 
    loader=TextLoader("/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/remark_test.txt")
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    # Define the embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    # Create the vector store 
    vector = FAISS.from_documents(documents, embeddings)
    # Define a retriever interface
    retriever = vector.as_retriever()
    # Define LLM
    model = ChatMistralAI(mistral_api_key=api_key)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # Create prompt box
    input_text = st.text_input("Ask a question")
    if st.button('Submit'):
        response = retrieval_chain.invoke({'input': input_text})
        if 'answer' in response:
            # Display only the "answer" part of the response
            st.text_area("Response", value=response['answer'], height=200, disabled=True)
        else:
            # If 'answer' key is not in response, handle the error or display a default message
            st.text_area("Response", value="No detailed answer available.", height=200, disabled=True)

   
if __name__ == '__main__':
    main()
  