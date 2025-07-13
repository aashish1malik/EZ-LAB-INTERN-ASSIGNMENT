import streamlit as st
from document import DocumentProcessor
from question_answer import QASystem
from challenge import Challenge
import time

def main():
    st.set_page_config(page_title="AI ", layout="wide")
    st.title("📄 AI Assistant")
    
    
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "questions" not in st.session_state:
        st.session_state.questions = None
    
    
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
        
        if uploaded_file:
            processor = DocumentProcessor()
            
            with st.spinner("Processing document..."):
                start_time = time.time()
                try:
                    text, chunks, vectorstore = processor.process_document(uploaded_file)
                    summary = processor.generate_summary(text)
                    
                    st.session_state.text = text
                    st.session_state.chunks = chunks
                    st.session_state.vectorstore = vectorstore
                    st.session_state.summary = summary
                    st.session_state.document_processed = True
                    
                    st.success(f"Processed in {time.time()-start_time:.1f}s")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.get("document_processed", False):
        st.subheader("Document Summary")
        st.write(st.session_state.summary)
        
        mode = st.radio(
            "Select mode:",
            ["Ask Anything","Challenge Me"],
            horizontal=True
        )
        
        if mode == "Ask Anything":
            st.subheader("Ask About the Document")
            question = st.text_input("Your question:")
            
            if question:
                with st.spinner("Finding answer..."):
                    qa_system = QASystem(st.session_state.vectorstore)
                    answer = qa_system.answer_question(question)
                
                st.markdown("### Answer")
                st.write(answer.split("Sources:")[0])
                
                with st.expander("View sources"):
                    st.write(answer.split("Sources:")[1])
        
        else:
            st.subheader("Challenge Mode")
            
            if st.button("Generate Questions"):
                with st.spinner("Creating questions..."):
                    challenge_system = Challenge()
                    st.session_state.questions = challenge_system.generate_questions(
                        st.session_state.text
                    )
            
            if st.session_state.get("questions"):
                for i, q in enumerate(st.session_state.questions):
                    # st.markdown(f"**Question {i+1}:** {q['question']}")
                    question_text = q.get('question')
                    if question_text:
                               st.markdown(f"**Question {i+1}:** {question_text}")
                    else:
                            st.warning(f"Missing 'question' in item {i+1}: {q}")
                            continue

                    user_answer = st.text_area(
                        f"Your answer {i+1}",
                        key=f"answer_{i}",
                        height=100
                    )
                    
                    if user_answer:
                        with st.spinner("Evaluating..."):
                            challenge_system = Challenge()
                            feedback = challenge_system.evaluate_answer(
                                q["question"],
                                q["answer"],
                                user_answer
                            )
                        
                        st.markdown("**Feedback:**")
                        st.write(feedback)
                        
                        with st.expander("Correct Answer"):
                            st.write(q["answer"])
                        with st.expander("Reference"):
                            st.write(q["reference"])
    else:
        st.info("Upload a document to begin")
        # st.markdown("""
        # **Note:** This version runs entirely on your local CPU with open-source models.
        # First-time use will download models (1-2GB total).
        # Processing may be slower than cloud-based solutions.
        # """)

if __name__ == "__main__":
    main()


