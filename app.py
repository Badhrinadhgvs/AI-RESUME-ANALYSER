import streamlit as st
import PyPDF2
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.agent import Agent
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f'LogFile\\resume_analyzer_{datetime.now().strftime("%Y%m%d")}.log',  # Log file with date
    level=logging.INFO,  # Log INFO level and above
    format='%(asctime)s - %(levelname)s - %(message)s'  # Timestamp, level, message
)

# Set up Streamlit page configuration
st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")
st.html("""<style>
                    .stSubheader {
                        text-decoration: solid;
                    }
                    .stMarkdown{
                        border: 3px solid #ffffffff;
                        padding:20px;
                    }
                    </style>""")
st.title("Resume Analyzer")
st.write("Upload your resume (PDF) and get personalized suggestions with a touch of AI magic!")

api_key = st.text_input("Enter your Groq API Key", type="password", help="Get your API key from the Groq website.")

def extract_text_from_pdf(file):
    logging.info("Starting PDF text extraction")
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        logging.info("PDF text extraction completed successfully")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {str(e)}")
        raise

def create_assistant(api_key):
    logging.info("Creating AI assistant")
    if not api_key:
        logging.warning("No API key provided")
        st.error("Please enter a valid Groq API Key to proceed.")
        return None
    
    try:
        # Initialize the Groq client with the API key
        groq_client = Groq(api_key=api_key,id="llama3-70b-8192")
        
        # Create the Agent with the Groq client as the model
        assistant = Agent(
            model=groq_client,  # Pass the Groq client instance here
            show_tool_calls=False,
            description="You are an expert HR consultant tasked with analyzing resumes and providing actionable improvement suggestions.",
            markdown=True
        )
        logging.info("AI assistant created successfully")
        return assistant
    except Exception as e:
        logging.error(f"Failed to create assistant: {str(e)}")
        st.error(f"Error initializing assistant: {str(e)}")
        return None

def analyze_resume(resume_text, api_key):
    logging.info("Starting resume analysis")
    assistant = create_assistant(api_key)
    if assistant is None:
        logging.warning("Assistant creation failed, skipping analysis")
        return None
    
    prompt = f"""
    Analyze the following resume and provide detailed suggestions for improvement. Focus on content, structure, and presentation. Provide specific examples where possible.

    Resume:
    {resume_text}
    
    Note:
    - If the Uploaded file text is not like resume then say it as It is not a resume.

    Format your response as:
    - **Strengths**: [List strengths]
    - **Areas for Improvement**: [List areas with suggestions]
    - **Overall Suggestions**: [General advice]
    - Make Sure that the output is small and Concise. This Is ImpoRtant
    - For Every Aspect 5 Lines is Max.
    """
    try:
        response = assistant.run(prompt, stream=False, model="llama3-70b-8192")
        logging.info("Resume analysis completed successfully")
        return response
    except Exception as e:
        logging.error(f"Failed to analyze resume: {str(e)}")
        st.error(f"Analysis failed: {str(e)}")
        return None

def main():
    logging.info("Application started")
    if not api_key:
        logging.warning("API key not entered")
        st.warning("Please enter your Groq API Key above to use the app.")
        return

    logging.info(f"API key entered: {api_key[:4]}**** (masked for security)")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], help="Only PDF files are supported.")

    if uploaded_file is not None:
        logging.info(f"File uploaded: {uploaded_file.name}")
        with st.spinner("Extracting text from your resume..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                st.success("Resume text extracted successfully! 🎉")
            except Exception as e:
                st.error(f"Failed to extract text: {str(e)}")
                logging.error(f"Text extraction error displayed to user: {str(e)}")
                return

        with st.expander("View Extracted Resume Text"):
            st.text_area("Resume Content", resume_text, height=300)
            logging.info("Extracted resume text displayed to user")

        if st.button("Analyze Resume"):
            logging.info("Analyze Resume button clicked")
            with st.spinner("Analyzing your resume..."):
                suggestions = analyze_resume(resume_text, api_key)
                if suggestions:

                    st.subheader("Resume Analysis Results")
                    st.markdown(body=suggestions.content)
                    logging.info("Resume analysis results displayed")
    else:
        logging.info("No file uploaded yet")
        st.info("Please upload a PDF resume to get started! 📑")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application crashed: {str(e)}")
        st.write(f"Sorry an error Occurred: {e}")