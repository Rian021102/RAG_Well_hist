import streamlit as st
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure to check and correct paths and imports as needed

st.title("Ask Anything About Your Drilling Reports!")
st.header("Extract information directly from your reports")

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
    path='/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/remark_test_02.txt'
    with open(path, 'a') as f:
        df_string = wellbore_df.to_string(header=False, index=False)
        f.write(df_string)

def main():
    path=('/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/drilling_report_comp.csv')
    df=get_data(path)
    wellbore_df = get_wellbore(df)
    text=convert_to_text(wellbore_df)
    # Split text into chunks 
    loader=TextLoader("/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/remark_test_02.txt")
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=30)
    documents = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(documents=documents, embedding=embeddings,persist_directory='/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/Chroma_DB')
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # Set up the local model:
    local_model = "llama3.1"
    llm = ChatOllama(model=local_model, num_predict=400,
                    stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])

    # Set up the RAG chain:
    prompt_template = """
    <|start_header_id|>user<|end_header_id|>
    You are a drilling engineer and an analyst. Your task is reading the drilling report,
    withing the context of the drilling report, answer the question based on the context of the document.
    All you need to answer what you can find in the document. If you can't find the answer, you can say "I don't know".
    Question: {question}
    Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Querying the LLM (oviously to test here you must ask a relevant question of your data)
    question = st.text_input("Ask a question")
    response = rag_chain.invoke({"question": question})
    if isinstance(response, dict):
        # Assume 'answer' is the correct key when response is a dictionary
        answer = response.get('answer', 'No answer found in response')
    else:
        # Directly use the response if it's not a dictionary (likely a string)
        answer = response

    st.text_area("Response", value=answer, height=200, disabled=True)


    
if __name__ == "__main__":
    main()

