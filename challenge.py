
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

class Challenge:
    def __init__(self):
       
        model_name = "distilgpt2"  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            device=-1  
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def generate_questions(self, document_text):
        prompt = f"""
        Generate three questions based on this document:
        {document_text[:3000]}
        
        For each question, provide:
        1. Question: [question]
        2. Answer: [answer]
        3. Reference: [text reference]
        """
        
        response = self.llm(prompt, max_length=600)
        return self._parse_questions(response)
    
    def _parse_questions(self, text):
        questions = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        current_q = {}
        for line in lines:
            if line.lower().startswith("question"):
                if current_q:
                    questions.append(current_q)
                current_q = {"question": line.split(":", 1)[1].strip()}
            elif line.lower().startswith("answer"):
                current_q["answer"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("reference"):
                current_q["reference"] = line.split(":", 1)[1].strip()
        
        if current_q:
            questions.append(current_q)
        
        return questions[:3]
    
    def evaluate_answer(self, question, correct_answer, user_answer):
        prompt = f"""
        Compare these answers for the question: {question}
        
        Correct Answer: {correct_answer}
        User Answer: {user_answer}
        
        Provide feedback on the user's answer:
        1. Accuracy (0-100%)
        2. What they got right
        3. What they missed
        4. How to improve
        
        Feedback:
        """
        
        return self.llm(prompt, max_length=400)


