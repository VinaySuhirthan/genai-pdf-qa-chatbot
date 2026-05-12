## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
In many cases, users need specific information from large documents without manually searching through them. A question-answering chatbot can address this problem by:

    Parsing and indexing the content of a PDF document.
    Allowing users to ask questions in natural language.
    Providing concise and accurate answers based on the content of the document.

The implementation will evaluate the chatbot’s ability to handle diverse queries and deliver accurate responses.

#### STEP 1: 
Load and Parse PDF
Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: 
Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: 
Initialize the LangChain QA Pipeline
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

#### STEP 4: 
Handle User Queries
Process user queries, retrieve relevant document sections, and generate responses.

#### STEP 5: 
Evaluate Effectiveness
Test the chatbot with a variety of queries to assess accuracy and reliability.

### PROGRAM:
```py
import os
import uuid
import panel as pn
import param
from dotenv import load_dotenv

# --- Langchain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Define constants for easy management
LLM_MODEL_NAME = "gpt-3.5-turbo"
UPLOAD_DIR = "uploads"

def load_db(file_path, chain_type, k):
    """Loads a PDF, splits it, creates embeddings, and sets up a conversational chain."""
    # Create the upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 1. Load documents
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # 3. Define embedding model
    embeddings = OpenAIEmbeddings()

    # 4. Create an in-memory vector database
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # 5. Define the retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # 6. Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

class ChatbotApp(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])
    
    def __init__(self, **params):
        super(ChatbotApp, self).__init__(**params)
        self.panels = []
        self.qa = None  # Initialize qa as None. It will be loaded after file upload.

    def call_load_db(self, count):
        """Handles the file upload and database loading process."""
        if file_input.value is None:
            return pn.pane.Markdown("Status: Please upload a PDF file.")
        
        try:
            # --- KEY CHANGE: Generate a unique filename ---
            unique_filename = f"{uuid.uuid4()}.pdf"
            save_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save the file locally with the unique name
            with open(save_path, 'wb') as f:
                f.write(file_input.value)

            button_load.button_style = "outline"
            button_load.name = "Loading..."
            
            # Load the database with the uniquely named file
            self.qa = load_db(save_path, "stuff", 4)
            
            button_load.button_style = "solid"
            button_load.name = "Load DB"
            self.clr_history()
            return pn.pane.Markdown(f"Status: Loaded `{file_input.filename}`")
        except Exception as e:
            return pn.pane.Markdown(f"Error: Could not load the file. {e}")

    def convchain(self, query):
        """Handles the conversation logic."""
        if not query:
            return pn.WidgetBox(scroll=True)
            
        # --- KEY CHANGE: Check if the database is loaded ---
        if not self.qa:
            return pn.WidgetBox(pn.Row('ChatBot:', pn.pane.Markdown("Please upload a PDF document first.", width=600)), scroll=True)
        
        result = self.qa.invoke({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  # Clear input widget
        return pn.WidgetBox(*self.panels, scroll=True)

    def clr_history(self, count=0):
        """Clears the chat history."""
        self.chat_history = []
        self.panels = []
        self.answer = ""

# --- Instantiate the App and Define UI Widgets ---
cb = ChatbotApp()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput(placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp)

# --- Assemble the Dashboard Layout ---
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# Chat With Your PDF')),
    pn.Tabs(
        ('Conversation', pn.Column(
            pn.Row(inp),
            pn.layout.Divider(),
            pn.panel(conversation, loading_indicator=True, height=300),
        )),
        ('Configuration', pn.Column(
            pn.Row(file_input, button_load),
            pn.Row(bound_button_load),
            pn.layout.Divider(),
            pn.Row(button_clearhistory),
        ))
    )
)

# To run this, save it as a .py file and run `panel serve your_script_name.py` in your terminal.
# dashboard.servable()
```

### OUTPUT:

### Conversation


<img width="1553" height="672" alt="ss" src="https://github.com/user-attachments/assets/3fc49354-eb69-4faa-90ea-0240e8baf210" />


### Database
<img width="931" height="403" alt="image" src="https://github.com/user-attachments/assets/20f13dc0-842f-4ebc-875b-22b9ae4d798d" />

### Chat History
<img width="931" height="335" alt="image" src="https://github.com/user-attachments/assets/ade0e0ba-9208-43f5-b520-0a80fee3b5f9" />


### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
