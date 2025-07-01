import os
import logging
from typing import Dict, List, TypedDict
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import Docx2txtLoader
from langchain_excel_loader import StructuredExcelLoader
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from azure.storage.blob import BlobServiceClient
import tempfile
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import re
from html import unescape

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Azure OpenAI configuration
os.environ["AZURE_OPENAI_API_KEY"] = 'FSuCtBuSxPsucoXyB7UTQJScxjxvYmBhE7m9d6EsplS6z2fzkmGMJQQJ99BDACYeBjFXJ3w3AAAAACOGp9ow'
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-zxsoh2784ai662228447616.cognitiveservices.azure.com/"
embedding_deployment = "text-embedding-3-large"
chat_deployment = "grok-3"

# Azure Blob Storage configuration
BLOB_ACCOUNT_NAME = "demoappstore"
BLOB_CONTAINER_NAME = "documents"
BLOB_SAS_TOKEN = 'sp=racwdli&st=2025-06-24T01:53:00Z&se=2026-04-26T09:53:00Z&spr=https&sv=2024-11-04&sr=c&sig=96gpxVNbNcqLyRL9Qkr%2FUVKeu3edyUMuuhH7KcLnX70%3D'
BLOB_SERVICE_URL = f"https://{BLOB_ACCOUNT_NAME}.blob.core.windows.net?{BLOB_SAS_TOKEN}"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'docx', 'xlsx'}

# Define state for LangGraph - Added report_path field
class ReportState(TypedDict):
    user_input: str
    uploaded_files: List[str]
    intent: str
    financial_docs: List
    quarterly_docs: List
    marketing_docs: List
    operations_docs: List
    hr_docs: List
    extracted_sections: Dict[str, str]
    progress: List[str]
    response: str
    report_path: str  # Added this field

# Initialize Azure OpenAI embeddings and LLM
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_deployment,
    openai_api_version="2023-05-15"
)
llm = AzureChatOpenAI(
    azure_deployment=chat_deployment,
    openai_api_version="2024-08-01-preview",
    temperature=0.7
)

# Prompt templates
intent_prompt = PromptTemplate(
    template="""Classify the user's intent based on the following input. The user may provide text, files, or both.
    - If the text includes phrases like 'generate report', 'create report', 'quarterly report', 'business report', 'analyze documents', or 'process files', or if files are uploaded, classify the intent as 'generate_report'.
    - Otherwise, classify the intent as 'casual_conversation'.
    - Return only the intent as a single word or phrase.

    Text: {text}
    Files uploaded: {files}

    Answer: """,
    input_variables=["text", "files"]
)

casual_prompt = PromptTemplate(
    template="""You are a friendly conversational AI. Respond to the user's input in a casual, engaging manner.
    Input: {text}
    Answer: """,
    input_variables=["text"]
)

prompt_templates = {
    "financial": PromptTemplate(
        template="""You are a financial analyst. Use the following context from an income statement to provide a detailed analysis for 2024.
        Focus on sales, COGS, expenses, and net income. Highlight trends, significant changes, and positive/negative aspects.
        The context is from an Excel sheet with financial data for 2024, where each row represents accounts and columns represent months or the full year (Year 2024). Focus on the relevant rows (e.g., SALES, NET SALES) for revenue-related queries.
        Note that Column B represents the total value for Year 2024, and subsequent columns represent monthly data from Dec 2024 to Jan 2024.
        
        COST OF GOODS SOLD -> this is the main category, thats why there are no values
        These are the subcategories under COST OF GOODS SOLD in the next row:
        DAYCARE MEALS & REFRESHMENTS
        SALARY
        BONUS
        EPF - EMPLOYER
        SOCSO - EMPLOYER
        EIS - EMPLOYER
        PRINTING & STATIONERY
        The empty row after the final subcategories is intentional, it indicates the end of the COST OF GOODS SOLD section, and the total value of COST OF GOODS SOLD.
        
        EXPENSES -> this is the main category, thats why there are no values
        These are the subcategories under EXPENSES in the next row:
        ADVERTISEMENT & MARKETING EXPENSES
        CASUAL WAGES
        CLEANING FEE
        FINE & PENALTY
        HRDF
        PETROL
        RENEWAL & SUBSCRIPTION FEE
        RENTAL
        TRAVELLING EXPENSES
        TOLL
        TEACHER MEALS & REFRESHEMENT
        TELEPHONE & INTERNET
        TRANSPORTATION
        UPKEEP OF MOTOR VEHICLE
        UPKEEP OF OFFICE
        UPKEEP OF OFFICE EQUIPMENT
        UNIFORM
        WATER & ELECTRICITY
        The empty row after the final subcategories is intentional, it indicates the end of the EXPENSES section, and the total value of EXPENSES.
        Context: {context}
        Answer: """,
        input_variables=["context"]
    ),
    "highlights": PromptTemplate(
        template="""Extract and summarize key highlights from the provided context for 2024, including major achievements, milestones, future projections or significant events.
        Context: {context}
        Answer: """,
        input_variables=["context"]
    ),
    "marketing": PromptTemplate(
        template="""Analyze the marketing report context for 2024. Summarize revenue performance, customer acquisition/retention, campaign effectiveness, and market trends.
        Context: {context}
        Answer: """,
        input_variables=["context"]
    ),
    "operations": PromptTemplate(
        template="""From the operations report context for 2024, summarize efficiency metrics, supply chain updates, quality/compliance, and process improvements.
        Context: {context}
        Answer: """,
        input_variables=["context"]
    ),
    "hr": PromptTemplate(
        template="""From the HR report context for 2024, summarize headcount, hiring, employee engagement, training, and diversity metrics.
        Context: {context}
        Answer: """,
        input_variables=["context"]
    )
}

report_prompt = PromptTemplate(
    template="""You are a business report writer. Using the provided insights, generate a structured Quarterly Business Report based on the information obtained from insights in markdown format, following this table of contents:
- Executive Summary
  - Key Highlights
  - Summary of Financial and Operational Performance
  - Strategic Priorities
- Company Overview
  - Mission, Vision, and Values
- Financial Performance
  - Income Statement Summary
  - Balance Sheet Overview
  - Cash Flow Analysis
  - Budget vs. Actuals
  - Forecast for Next Quarter
- Sales & Marketing
  - Revenue Performance
  - Customer Acquisition and Retention
  - Campaign Effectiveness
  - Market Trends and Competitive Analysis
- Operations
  - Efficiency Metrics
  - Supply Chain and Logistics
  - Quality and Compliance Updates
  - Process Improvements
- Product & Innovation
  - Product Development Updates
  - Roadmap Progress
  - R&D Initiatives
  - Customer Feedback and Product Performance
- Customer Success & Support
  - Satisfaction Metrics
  - Support Ticket Trends
  - Key Account Updates
  - Retention and Churn Analysis
- Human Resources
  - Headcount and Hiring Updates
  - Employee Engagement and Culture
  - Training and Development
  - Diversity and Inclusion Metrics
- Strategic Initiatives
  - Status of Major Projects
  - Cross-Functional Collaboration
  - Risks and Mitigation Strategies
- Goals & Priorities for Next Quarter
  - Strategic Goals
  - Departmental Objectives
  - Key Performance Indicators

Insights:
- Financial: {financial}
- Highlights: {highlights}
- Marketing: {marketing}
- Operations: {operations}
- HR: {hr}

For sections without provided insights, use placeholders like '[Placeholder: Data not available]'. Format the output as markdown with appropriate headings and bullet points where needed.
Answer: ```markdown
# Quarterly Business Report
## Q4 2024

# Table of Contents
- Executive Summary
- Company Overview
- Financial Performance
- Sales & Marketing
- Operations
- Product & Innovation
- Customer Success & Support
- Human Resources
- Strategic Initiatives
- Goals & Priorities for Next Quarter
```""",
    input_variables=["financial", "highlights", "marketing", "operations", "hr"]
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_blob(file, filename):
    try:
        blob_service_client = BlobServiceClient(account_url=BLOB_SERVICE_URL)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(file, overwrite=True)
        logger.info(f"Uploaded {filename} to blob storage")
        return True
    except Exception as e:
        logger.error(f"Error uploading to blob: {str(e)}")
        return False

def download_from_blob(filename):
    """Download a file from Azure Blob Storage to a temporary file"""
    try:
        blob_service_client = BlobServiceClient(account_url=BLOB_SERVICE_URL)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(filename)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        
        # Download blob data and write to temp file
        blob_data = blob_client.download_blob().readall()
        temp_file.write(blob_data)
        temp_file.close()
        
        logger.info(f"Downloaded {filename} from blob storage to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Error downloading from blob: {str(e)}")
        return None

def markdown_to_pdf(markdown_content, output_path):
    """
    Convert markdown content to PDF using ReportLab with proper formatting
    """
    try:
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            spaceBefore=0,
            alignment=TA_CENTER,
            textColor='black',
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=5,
            alignment=TA_CENTER,
            textColor='black',
            fontName='Helvetica'
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor='black',
            fontName='Helvetica-Bold'
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor='black',
            fontName='Helvetica-Bold'
        )
        
        heading3_style = ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor='black',
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=2,
            alignment=TA_JUSTIFY,
            textColor='black',
            fontName='Helvetica'
        )
        
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            spaceBefore=2,
            leftIndent=20,
            bulletIndent=10,
            textColor='black',
            fontName='Helvetica'
        )
        
        # Parse markdown content
        story = []
        lines = markdown_content.split('\n')
        in_list = False
        current_list_items = []
        
        def process_inline_formatting(text):
            """Process inline markdown formatting"""
            # Bold text
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            # Italic text  
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            # Code formatting
            text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
            # Remove any remaining markdown links but keep the text
            text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
            return text
        
        def add_list_items():
            """Add accumulated list items to story"""
            if current_list_items:
                for item in current_list_items:
                    story.append(Paragraph(f"• {item}", bullet_style))
                current_list_items.clear()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines but add spacing
            if not line:
                if in_list:
                    add_list_items()
                    in_list = False
                story.append(Spacer(1, 6))
                i += 1
                continue
            
            # Handle headings
            if line.startswith('# '):
                if in_list:
                    add_list_items()
                    in_list = False
                
                text = process_inline_formatting(line[2:].strip())
                if 'Quarterly Business Report' in text:
                    story.append(Paragraph(text, title_style))
                elif 'Q4 2024' in text or any(quarter in text for quarter in ['Q1', 'Q2', 'Q3', 'Q4']):
                    story.append(Paragraph(text, subtitle_style))
                else:
                    story.append(Paragraph(text, heading1_style))
                    
            elif line.startswith('## '):
                if in_list:
                    add_list_items()
                    in_list = False
                text = process_inline_formatting(line[3:].strip())
                story.append(Paragraph(text, heading2_style))
                
            elif line.startswith('### '):
                if in_list:
                    add_list_items()
                    in_list = False
                text = process_inline_formatting(line[4:].strip())
                story.append(Paragraph(text, heading3_style))
                
            # Handle list items
            elif line.startswith('- ') or line.startswith('* '):
                text = process_inline_formatting(line[2:].strip())
                current_list_items.append(text)
                in_list = True
                
            # Handle numbered lists
            elif re.match(r'^\d+\.\s', line):
                if in_list:
                    add_list_items()
                    in_list = False
                text = re.sub(r'^\d+\.\s', '', line)
                text = process_inline_formatting(text)
                story.append(Paragraph(f"• {text}", bullet_style))
                
            # Handle table of contents differently
            elif 'Table of Contents' in line:
                if in_list:
                    add_list_items()
                    in_list = False
                story.append(Paragraph('Table of Contents', heading2_style))
                
            # Regular paragraph
            else:
                if in_list:
                    add_list_items()
                    in_list = False
                    
                if line and not line.startswith('```'):  # Skip code blocks
                    text = process_inline_formatting(line)
                    # Don't add empty placeholder paragraphs
                    if not ('[Placeholder:' in text and ']' in text):
                        story.append(Paragraph(text, body_style))
            
            i += 1
        
        # Add any remaining list items
        if in_list:
            add_list_items()
        
        # Build PDF
        doc.build(story)
        logger.info(f"Successfully created PDF: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        return False

def detect_intent_node(state: ReportState) -> Dict:
    try:
        state['progress'].append("Detecting user intent...")
        files_str = ", ".join(state['uploaded_files']) if state['uploaded_files'] else "None"
        intent = llm.invoke(intent_prompt.format(
            text=state['user_input'] or "No text provided",
            files=files_str
        )).content.strip()
        state['progress'].append(f"Intent detected: {intent}")
        return {"intent": intent, "progress": state['progress']}
    except Exception as e:
        logger.error(f"Error detecting intent: {str(e)}")
        state['progress'].append(f"Error detecting intent: {str(e)}")
        raise

def casual_conversation_node(state: ReportState) -> Dict:
    try:
        state['progress'].append("Generating conversational response...")
        response = llm.invoke(casual_prompt.format(text=state['user_input'])).content
        state['progress'].append("Conversational response generated")
        return {"response": response, "progress": state['progress']}
    except Exception as e:
        logger.error(f"Error in casual conversation: {str(e)}")
        state['progress'].append(f"Error in casual conversation: {str(e)}")
        raise

def load_documents(state: ReportState) -> Dict:
    try:
        state['progress'].append("Loading documents from Azure Blob Storage...")
        blob_service_client = BlobServiceClient(account_url=BLOB_SERVICE_URL)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        doc_splits = {
            "financial_docs": [],
            "quarterly_docs": [],
            "marketing_docs": [],
            "operations_docs": [],
            "hr_docs": []
        }
        text_splitter = CharacterTextSplitter(chunk_size=50000, chunk_overlap=100)
        blob_files = {
            "financial": "financial_statement.xlsx",
            "quarterly": "quarterly_highlights.docx",
            "marketing": "marketing_report.docx",
            "operations": "operational_report.docx",
            "hr": "HR_report.docx"
        }

        for key, file_name in blob_files.items():
            blob_client = container_client.get_blob_client(file_name)
            if blob_client.exists():
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_name.split('.')[-1]}") as temp_file:
                    blob_data = blob_client.download_blob().readall()
                    temp_file.write(blob_data)
                    temp_file_path = temp_file.name
                if file_name.endswith('.xlsx'):
                    excel_loader = StructuredExcelLoader(temp_file_path)
                    docs = excel_loader.load()
                else:
                    loader = Docx2txtLoader(temp_file_path)
                    docs = loader.load()
                if docs:
                    state_key = f"{key}_docs"
                    doc_splits[state_key] = text_splitter.split_documents(docs)
                    state['progress'].append(f"Loaded {len(doc_splits[state_key])} {key} document chunks")
                else:
                    state['progress'].append(f"No content loaded from {file_name}")
                os.remove(temp_file_path)
            else:
                state['progress'].append(f"Blob {file_name} does not exist")
        return {**doc_splits, "progress": state['progress']}
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        state['progress'].append(f"Error loading documents: {str(e)}")
        raise

def extract_insights(state: ReportState) -> Dict:
    logger.info("Entering extract_insights")
    try:
        state['progress'].append("Extracting insights from documents...")
        extracted_sections = {}
        for doc_type, docs in [
            ("financial", state["financial_docs"]),
            ("highlights", state["quarterly_docs"]),
            ("marketing", state["marketing_docs"]),
            ("operations", state["operations_docs"]),
            ("hr", state["hr_docs"])
        ]:
            logger.info(f"Processing {doc_type} with {len(docs)} documents")
            if not docs:
                logger.warning(f"No documents for {doc_type}")
                extracted_sections[doc_type] = f"[Placeholder: No {doc_type} data available]"
                state['progress'].append(f"No documents for {doc_type}")
                continue
            logger.info(f"Creating FAISS vector store for {doc_type}")
            vector_store = FAISS.from_documents(docs, embeddings)
            logger.info(f"Setting up RetrievalQA chain for {doc_type}")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": min(len(docs), 2)}),
                chain_type_kwargs={"prompt": prompt_templates[doc_type]}
            )
            logger.info(f"Invoking QA chain for {doc_type}")
            result = qa_chain.invoke({"query": f"List out in detail {doc_type} and gather insights"})
            logger.info(f"QA chain completed for {doc_type}")
            extracted_sections[doc_type] = result["result"]
            state['progress'].append(f"Extracted insights for {doc_type}")
        logger.info("Exiting extract_insights successfully")
        return {"extracted_sections": extracted_sections, "progress": state['progress']}
    except Exception as e:
        logger.error(f"Error in extract_insights: {str(e)}", exc_info=True)
        state['progress'].append(f"Error extracting insights: {str(e)}")
        raise

def generate_markdown_report(state: ReportState) -> Dict:
    try:
        state['progress'].append("Generating markdown report...")
        extracted_sections = state["extracted_sections"]
        financial = extracted_sections.get("financial", "[Placeholder: Financial data not available]").split("Answer: ")[-1]
        highlights = extracted_sections.get("highlights", "[Placeholder: Highlights not available]").split("Answer: ")[-1]
        marketing = extracted_sections.get("marketing", "[Placeholder: Marketing data not available]").split("Answer: ")[-1]
        operations = extracted_sections.get("operations", "[Placeholder: Operations data not available]").split("Answer: ")[-1]
        hr = extracted_sections.get("hr", "[Placeholder: HR data not available]").split("Answer: ")[-1]

        report_content = llm.invoke(report_prompt.format(
            financial=financial,
            highlights=highlights,
            marketing=marketing,
            operations=operations,
            hr=hr
        )).content
        
        markdown_content = report_content
        if "```markdown" in report_content:
            markdown_content = report_content.split("```markdown")[1].split("```")[0].strip()

        state['progress'].append("Converting markdown to PDF...")
        
        # Create PDF filename
        pdf_filename = f"quarterly_business_report.pdf"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_path = temp_pdf_file.name

        # Convert Markdown to PDF using ReportLab
        if markdown_to_pdf(markdown_content, temp_pdf_path):
            state['progress'].append("PDF generated successfully")
            
            # Upload PDF to Azure Blob Storage
            try:
                with open(temp_pdf_path, "rb") as pdf_file:
                    if upload_to_blob(pdf_file, pdf_filename):
                        state['progress'].append(f"PDF report uploaded to blob storage as {pdf_filename}")
                    else:
                        logger.warning("Failed to upload PDF to blob storage, but PDF was generated successfully")
            except Exception as upload_error:
                logger.warning(f"Failed to upload PDF to blob storage: {str(upload_error)}")
                state['progress'].append("PDF generated but upload to blob storage failed")
            
            # Clean up temporary file
            os.remove(temp_pdf_path)
        else:
            raise Exception("Failed to generate PDF from markdown content")

        return {
            "extracted_sections": extracted_sections,
            "report_path": pdf_filename,  # This will now be properly included in the state
            "response": markdown_content,
            "progress": state['progress']
        }
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        state['progress'].append(f"Error generating report: {str(e)}")
        raise

# Define LangGraph workflow
workflow = StateGraph(ReportState)
workflow.add_node("detect_intent", detect_intent_node)
workflow.add_node("casual_conversation", casual_conversation_node)
workflow.add_node("load_documents", load_documents)
workflow.add_node("extract_insights", extract_insights)
workflow.add_node("generate_markdown_report", generate_markdown_report)

# Define edges
workflow.set_entry_point("detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    lambda state: state["intent"],
    {
        "casual_conversation": "casual_conversation",
        "generate_report": "load_documents"
    }
)
workflow.add_edge("casual_conversation", END)
workflow.add_edge("load_documents", "extract_insights")
workflow.add_edge("extract_insights", "generate_markdown_report")
workflow.add_edge("generate_markdown_report", END)

# Compile workflow
app_workflow = workflow.compile()

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        logger.info("Received /chat request")
        data = request.form
        text = data.get('text', '')
        files = request.files.getlist('files')
        logger.info(f"Text: {text}, Files: {[f.filename for f in files]}")
        uploaded_files = []

        # Handle file uploads
        if files:
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    new_filename = filename
                    logger.info(f"Uploading file: {filename}")
                    if upload_to_blob(file, new_filename):
                        uploaded_files.append(new_filename)
            if not uploaded_files and files:
                return jsonify({"response": "No valid files uploaded.", "type": "text", "download_available": False}), 400

        # Initialize state
        initial_state = ReportState(
            user_input=text,
            uploaded_files=uploaded_files,
            intent="",
            financial_docs=[],
            quarterly_docs=[],
            marketing_docs=[],
            operations_docs=[],
            hr_docs=[],
            extracted_sections={},
            progress=[],
            response="",
            report_path=""  # Initialize report_path
        )

        # Run workflow
        logger.info("Running workflow")
        result = app_workflow.invoke(initial_state)
        response_type = "markdown" if result["intent"] == "generate_report" else "text"
        logger.info(f"Workflow result: {result}")
        
        # Check if it's a report generation and we have a report path
        download_available = result["intent"] == "generate_report" and bool(result.get("report_path"))
        
        # Prepare response data
        response_data = {
            "response": result["response"],
            "type": response_type,
            "progress": result["progress"],
            "download_available": download_available,
            "pdf_filename": result.get("report_path") if download_available else None
        }
        
        logger.info(f"Download available: {download_available}")
        logger.info(f"PDF filename: {result.get('report_path')}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}", "type": "text", "download_available": False}), 500

@app.route('/download/<filename>')
def download_pdf(filename):
    """Download PDF file from Azure Blob Storage"""
    try:
        # Validate filename to prevent path traversal attacks
        if not filename.endswith('.pdf') or '..' in filename or '/' in filename:
            return jsonify({"error": "Invalid filename"}), 400
            
        # Download file from blob storage to temporary location
        temp_file_path = download_from_blob(filename)
        
        if temp_file_path is None:
            return jsonify({"error": "File not found or download failed"}), 404
        
        def remove_file(response):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")
            return response
        
        # Send file and clean up after sending
        return send_file(
            temp_file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)