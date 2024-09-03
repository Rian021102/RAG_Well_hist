from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def loader_text(path):
    loader = TextLoader(path)
    docs = loader.load()
    return docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=30)
    documents = text_splitter.split_documents(docs)
    return documents

def create_db(documents, embeddings):
    db = Chroma.from_documents(documents=documents, embedding=embeddings,persist_directory='/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/Chroma_DB')
    return db

def create_retriever(db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

def create_llm(local_model):
    llm = ChatOllama(model=local_model, num_predict=400,
                 stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    return llm

def create_rag_chain(retriever, llm):
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

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    path="/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/remark.txt"
    docs = loader_text(path)
    documents = split_text(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = create_db(documents, embeddings)
    retriever = create_retriever(db)
    local_model = "llama3.1"
    llm = create_llm(local_model)
    rag_chain = create_rag_chain(retriever, llm)
    question = "List all the problem during drilling?"
    print(question)
    print(rag_chain.invoke(question))

if __name__ == "__main__":
    main()


