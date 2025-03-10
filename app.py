import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage,HumanMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]

# Selecting models for llm
models = ["gemma2-9b-it","mixtral-8x7b-32768","llama-3.3-70b-versatile"]

# Initialize the database, here it is postgres
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# create sql query generator
def get_sql_chain(db):
    template = """
        You are a data analyst at a company. You are interacting with a user
        who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would
        answer the user's question. take teh conversation historyinto account.

        <SCHEMA>{schema}</SCHEMA>

        Conversational History: {chat_history}

        write onlly the SQL query and nothing else. Do not wrap the SQL query
        in any other text, not even backticks.

        For example:
        Question: How many phones are in the database that are cheaper than 500 dollars?
        SQL Query: SELECT COUNT(*) FROM "tbl_phones" WHERE "price" < 500;  

        Question: Name 5 brand models?
        SQL Query: SELECT "model" FROM tbl_phones LIMIT 5;

        Your turn:

        Question: {question}
        SQL QUERY: 
    """
    prompt = ChatPromptTemplate.from_template(role= "user",template=template)
    
    llm = ChatGroq(api_key = groq_api_key, model_name= models[0])

    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# create response generator 
def get_response(user_query:str, db:SQLDatabase, chat_history:list):
    sql_chain = get_sql_chain(db)

    if not isinstance(chat_history, list):
        chat_history = [] 

    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query and sql response, write a natural language response. You do not let user know
        that you are accessing the database. You reply saying "As far as i know".

        <SCHEMA>{schema}</SCHEMA>
        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User Question: {question}   
        SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = llm = ChatGroq(api_key = groq_api_key, model_name= models[1])
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema = lambda _: db.get_table_info(),
            response = lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    #use invoke for direct output, stream for streaming
    return chain.stream({
            "question": user_query, 
            "chat_history": chat_history
        })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content = "I'm here to help you query your database."),
    ]

# Titlebar of the streamlit app
st.set_page_config( page_title="Chat with SQL", page_icon=":speech_balloon: ")
st.title("Chat with Postgres")

# Connection with the database
with st.sidebar:
    st.subheader("Settings")
    st.write("Connect with the database and converse")
    st.text_input("Host",value="localhost" ,key="Host")
    st.text_input("User",value="postgres" ,key="User")
    st.text_input("Password",type="password",value="sigdel" ,key="Password")
    st.text_input("Port",value="8080" ,key="Port")
    st.text_input("Database",value="db_phone" ,key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to the databse..."):
            if "db" not in st.session_state:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to Database !...")

#show messsages from chat history as AI or Human(classify the user type)
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append( HumanMessage(content=user_query) )

    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
       # markdown for GitHub-flavored Markdown write_stream for streaming
       response = st.write_stream( get_response(user_query, st.session_state.db, st.session_state.chat_history) ) # type: ignore
    st.session_state.chat_history.append(AIMessage(content=response))
