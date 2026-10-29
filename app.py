import os
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from rag.rag_pipeline import retrieve

load_dotenv()

# ── LLM setup ─────────────────────────────────────────────────

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

# ── Tab 1: Ask about Job Descriptions ─────────────────────────

def ask_jd(question):
    """Retrieve relevant JD chunks and answer the question using LLM"""
    if not question.strip():
        return "Please enter a question."

    context = retrieve(question, k=6)

    llm = get_llm()
    messages = [

        SystemMessage(content="""You are a helpful career assistant with access to job description data.
        Answer the user's question using the job description context provided.
        - If asked about salary/compensation, list EACH role with its exact compensation range and clearly state which is highest.
        - If asked to compare roles, make a structured comparison with specific details.
        - Always mention the role title and company name when referencing a specific job.
        - Be specific and factual. Do not generalize or summarize vaguely.
        - If the answer is not in the context, say so clearly."""),
        HumanMessage(content=f"""Context from job descriptions:
{context}

Question: {question}""")
    ]

    response = llm.invoke(messages)
    return response.content


# ── Tab 2: Resume Matcher ──────────────────────────────────────

def match_resume(resume_text):
    """Compare resume against all JDs in the knowledge base and score match"""
    if not resume_text.strip():
        return "Please paste your resume text."

    context = retrieve(resume_text, k=5)

    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a career counselor and hiring expert.
        Analyze how well the resume matches the job descriptions provided.
        Give a structured response with:
        1. Best matching role and why (with match score out of 10)
        2. Key matching skills
        3. Missing skills or gaps
        4. Specific suggestions to improve the resume for these roles
        Be honest and constructive."""),
        HumanMessage(content=f"""Job Descriptions from our database:
{context}

Candidate Resume:
{resume_text}

Analyze the match and provide detailed feedback.""")
    ]

    response = llm.invoke(messages)
    return response.content


# ── Tab 3: Cover Letter Generator ─────────────────────────────

def generate_cover_letter(resume_text, job_title, company_name):
    """Generate a tailored cover letter using resume + relevant JD from knowledge base"""
    if not resume_text.strip():
        return "Please paste your resume text."
    if not job_title.strip():
        return "Please enter the job title."

    query   = f"{job_title} {company_name}"
    context = retrieve(query, k=3)

    llm = get_llm()
    messages = [
        SystemMessage(content="""You are an expert career coach who writes compelling cover letters.
        Write a professional, personalized cover letter based on the resume and job description.
        The letter should:
        - Be 3-4 paragraphs long
        - Highlight relevant experience and skills from the resume
        - Show genuine interest in the specific role and company
        - End with a confident call to action
        - Sound natural and human, not AI-generated"""),
        HumanMessage(content=f"""Job Description context:
{context}

Candidate Resume:
{resume_text}

Write a cover letter for the position of {job_title} at {company_name}.""")
    ]

    response = llm.invoke(messages)
    return response.content


# ── Gradio UI ──────────────────────────────────────────────────

with gr.Blocks(title="Job Description RAG Assistant") as demo:

    gr.Markdown("# 💼 Job Description RAG Assistant")
    gr.Markdown("Ask questions about job descriptions, match your resume, or generate a cover letter — powered by RAG")

    with gr.Tabs():

        # Tab 1
        with gr.TabItem("🔍 Ask About JDs"):
            gr.Markdown("Ask anything about the job descriptions in the knowledge base.")
            q_input  = gr.Textbox(lines=2, placeholder="e.g. What skills are needed for the NLP Engineer role?")
            q_output = gr.Textbox(lines=10, label="Answer")
            q_btn    = gr.Button("Ask", variant="primary")
            q_btn.click(ask_jd, inputs=q_input, outputs=q_output)

            gr.Examples(
                examples=[
                    ["What Python skills are required across the job descriptions?"],
                    ["Which role pays the highest salary?"],
                    ["What is the experience required for the Data Scientist role?"],
                    ["Which roles mention RAG or LangChain?"],
                ],
                inputs=q_input
            )

        # Tab 2
        with gr.TabItem("📄 Resume Matcher"):
            gr.Markdown("Paste your resume below to see which roles you match best and get feedback.")
            resume_input  = gr.Textbox(lines=15, placeholder="Paste your full resume text here...")
            match_output  = gr.Textbox(lines=15, label="Match Analysis")
            match_btn     = gr.Button("Analyze My Resume", variant="primary")
            match_btn.click(match_resume, inputs=resume_input, outputs=match_output)

        # Tab 3
        with gr.TabItem("✉️ Cover Letter Generator"):
            gr.Markdown("Generate a tailored cover letter based on your resume and the target role.")
            cl_resume  = gr.Textbox(lines=10, placeholder="Paste your resume text here...")
            cl_title   = gr.Textbox(lines=1, placeholder="e.g. Data Scientist")
            cl_company = gr.Textbox(lines=1, placeholder="e.g. TechCorp Analytics")
            cl_output  = gr.Textbox(lines=20, label="Generated Cover Letter")
            cl_btn     = gr.Button("Generate Cover Letter", variant="primary")
            cl_btn.click(
                generate_cover_letter,
                inputs=[cl_resume, cl_title, cl_company],
                outputs=cl_output
            )

demo.launch()
