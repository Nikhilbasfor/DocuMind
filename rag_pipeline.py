from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(pages)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)

def create_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
    Use the following context to answer the question.
    If you don't know the answer, say you don't know.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain, retriever

def process_pdf_and_query(pdf_path, question):
    try:
        images = convert_from_path(pdf_path, poppler_path=r'C:\poppler\poppler-26.02.0\Library\bin')
        pages = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"page": i}
                ))

        if not pages:
            return {
                "answer": "Could not extract text from this PDF even with OCR.",
                "sources": []
            }

        chunks = split_into_chunks(pages)

        if not chunks:
            return {
                "answer": "PDF loaded but could not be split into chunks.",
                "sources": []
            }

        vector_store = create_vector_store(chunks)
        chain, retriever = create_qa_chain(vector_store)

        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)

        if hasattr(answer, 'content'):
            answer_text = answer.content
        elif isinstance(answer, str):
            answer_text = answer
        else:
            answer_text = str(answer)

        return {
            "answer": answer_text,
            "sources": source_docs if source_docs else []
        }

    except Exception as e:
        import traceback
        raise Exception(traceback.format_exc())