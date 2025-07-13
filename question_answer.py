


from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class QASystem:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        
        
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            device=-1  
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
    def answer_question(self, question):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        result = qa({"query": question})
        answer = result["result"]
        sources = "\n".join([doc.page_content[:200] + "..." for doc in result["source_documents"]])
        
        return f"{answer}\n\nSources:\n{sources}"