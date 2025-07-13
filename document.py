
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class DocumentProcessor:
    def __init__(self):
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def extract_text(self, file):
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            text = file.read().decode("utf-8")
        return text

    def process_document(self, file):
        text = self.extract_text(file)
        chunks = self.text_splitter.split_text(text)
        vectorstore = FAISS.from_texts(chunks, self.embeddings)
        return text, chunks, vectorstore

    def generate_summary(self, text):
        from transformers import pipeline
        
        
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            # model="google/pegasus-xsum",
            device=-1  
        )
        
        
        max_chunk = 1024
        text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        
        summaries = []
        for chunk in text_chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            
        return " ".join(summaries)