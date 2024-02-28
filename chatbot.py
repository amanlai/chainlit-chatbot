import os
import re
import io
import zipfile
from datetime import datetime
from dotenv import load_dotenv

import chainlit as cl
from chainlit.server import app
from fastapi import Request
from fastapi.responses import HTMLResponse
import pymongo
import certifi


from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool, AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate#, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# (from langchain.agents.agent_toolkits.conversational_retrieval.openai_functions 
#  import create_conversational_retrieval_agent)
# from langchain.chains import ConversationalRetrievalChain
# from langchain.agents.format_scratchpad import format_to_openai_function_messages
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
# from langchain_community.vectorstores import Chroma
from lib.tools import get_tools
from ingest import IngestData


load_dotenv()

# SYSTEM_TEMPLATE = (
#     "You are a helpful bot. "
#     "If you do not know the answer, just say that you do not know, do not try to make up an answer."
# )
persist_directory = os.environ.get("PERSIST_DIRECTORY", 'db')
source_directory = os.environ.get("DOCUMENT_SOURCE_DIR", 'docs')

database_type = 'faiss'

os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', 'dummy_key')

model_name = os.environ.get("MODEL_NAME", 'gpt-3.5-turbo')
use_client = os.environ.get("USE_CLIENT", 'True') == 'True'

botname = os.environ.get("BOTNAME", "Personal Assistant")
verbose = os.environ.get("VERBOSE", 'True') == 'True'
stream = os.environ.get("STREAM", 'True') == 'True'
mongo_uri = os.environ['MONGO_URI']


##########################################################
############### MONGODB CONNECTION #######################
##########################################################

def init_connection():
    try:
        mongo_client = pymongo.MongoClient(
            mongo_uri, 
            server_api=pymongo.server_api.ServerApi('1'), 
            tlsCAFile=certifi.where()
        )
        db = mongo_client['chattabot']
        collection = db['data']
        print("Connection to MongoDB successful")
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return
    return collection


def ingest_into_collection(collection, business_name, data):
    """
    Ingest data into MongoDB
    """
    collection.delete_one({'_id': business_name})
    collection.insert_one({"_id": business_name, **data})


def update_collection(collection, business_name, data):
    """
    Update certain key-value pairs in documents in a collection
    """
    collection.update_one(
        {'_id': business_name}, 
        {"$set": data}
    )


def copy_from_collection(
        data_processor, 
        collection, 
        business_name, 
        old_config=False
):
    # get dictionary of attributes from MongoDB
    data = next(collection.find({'_id': business_name}))
    # read the archived index files from mongodb into a binary stream
    index_files = io.BytesIO(data.get("index_files"))
    # extract the index files into the persist directory
    with zipfile.ZipFile(index_files) as zip_file:
        zip_file.extractall(persist_directory)
    # now that we have the index files stored in persist directory
    # we can get the vector stores from them
    vector_store = data_processor.get_vector_store()
    if old_config:
        message_prompt = data.get("message_prompt", 'Ask me anything!')
        system_message = data.get("system_message", "")
        temperature = data.get("temperature", 0)
        return vector_store, message_prompt, system_message, temperature
    else:
        return vector_store




##########################################################
############ AGENT AND TOOL DEFINITION ###################
##########################################################

def build_tools(vector_store, k=5):
    """
    Creates the list of tools to be used by the Agent Executor
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": k}
    )

    retriever_tool = create_retriever_tool(
        retriever,
        "search_document",
        """Searches and retrieves information from the vector store """
        """to answer questions whose answers can be found there.""",
    )

    tools = [*get_tools(), retriever_tool]
    return tools


def create_agent(vector_store, temperature, system_message):

    sys_msg = f"""You are a helpful assistant. 
    Respond to the user as helpfully and accurately as possible.

    It is important that you provide an accurate answer. 
    If you're not sure about the details of the query, don't provide an answer; 
    ask follow-up questions to have a clear understanding.

    Use the provided tools to perform calculations and lookups related to the 
    calendar and datetime computations.

    If you don't have enough context to answer question, 
    you should ask user a follow-up question to get needed info. 
    
    Always use tools if you have follow-up questions to the request or 
    if there are questions related to datetime.
    
    For example, given question: "What time will the restaurant open tomorrow?", 
    follow the following steps to answer it:
    
      1. Use get_day_of_week tool to find the week day name of tomorrow.
      2. Use search_document tool to see if the restaurant is open on that week day name.
      3. The restaurant might be closed on specific dates such as a Christmas Day, 
         therefore, use get_date tool to find calendar date of tomorrow.
      4. Use search_document tool to see if the restaurant is open on that date.
      5. Generate an answer if possible. If not, ask for clarifications.
    

    Don't make any assumptions about data requests. 
    For example, if dates are not specified, you ask follow up questions. 
    There are only two assumptions you can make about a query:
    
      1. if the question is about dates but no year is given, 
         you can assume that the year is {datetime.today().year}.
      2. if the question includes a weekday, you can assume that 
         the week is the calendar week that includes the date {datetime.today().strftime('%m-%d-%Y')}.

    Dates should be in the format mm-dd-YYYY.
    
    {system_message}

    If you can't find relevant information, instead of making up an answer, 
    say "Let me connect you to my colleague".
    """

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    tools = build_tools(vector_store)

    # system_message = SystemMessage(content=sys_msg)
    # agent_executor = create_conversational_retrieval_agent(
    #     llm=llm, 
    #     tools=tools, 
    #     system_message=system_message,
    #     verbose=True, 
    #     max_token_limit=200
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=10,
        handle_parsing_errors=True,
        early_stopping_method = 'generate',
        callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=stream)]
    )
    cl.user_session.set('agent_executor', agent_executor)


##########################################################
############## CHAINLIT APP DEFINITION ###################
##########################################################

@cl.action_callback("action_button")
async def on_action(action):
    print("The user clicked on the action button!")
    return action


# App Hooks
@cl.on_chat_start
async def main():

    """ Startup """

    message_prompt = "Ask me anything!"
    system_message = ""
    temperature = 0

    # process file ingestion
    response = await cl.AskActionMessage(
        content="Do you want to use previously uploaded file or do you want to a new file?", 
        actions=[
            cl.Action(name="Use previous file", value="old_file"),
            cl.Action(name="Use new file", value="new_file")
        ]
    ).send()

    # process the response
    if response:
        # instantialize data ingester / getter
        data_processor = IngestData(database_type=database_type)
        vector_store = None

        if response.get("value") == "new_file":

            files = None
            # Wait for the user to upload a file
            while files is None:
                files = await cl.AskFileMessage(
                    content="Please upload a pdf file!", accept=["application/pdf"]
                ).send()

            pdf_file = files[0].path
            vector_store = data_processor.build_embeddings(pdf_file)

    # process business type
    bus_res = await cl.AskUserMessage(
        content="Which business are you going to ask about today?", 
    ).send()

    try:
        business_name = bus_res.get('output')
    except KeyError:
        cl.Text("must supply business name")


    db_res = await cl.AskActionMessage(
        content="Do you want to use previous configurations or do you want to create a new one?", 
        actions=[
            cl.Action(name="Use previous configurations", value="old_config"),
            cl.Action(name="Create new configurations", value="new_config")
        ]
    ).send()

    if db_res:
        # initiate mongodb connection
        collection = init_connection()
        # path to archived index files
        path = os.path.join(source_directory, "index_files.zip")

        if db_res.get("value") == "new_config":
            # if beginning from scratch
            data = {
                "message_prompt": message_prompt,
                "system_message": system_message,
                "temperature": temperature,
            }
            try:
                # save the archived index files into mongodb
                with open(path, 'rb') as f:
                    index_files = f.read()
                os.remove(path)
                # add the index files into the basic configuration
                data |= {"index_files": index_files}
                ingest_into_collection(collection, business_name, data)
            except FileNotFoundError:
                update_collection(collection, business_name, data)
                vector_store = copy_from_collection(data_processor, collection, business_name)
        else:
            if vector_store is None:
                tpl = copy_from_collection(data_processor, collection, business_name, old_config=True)
                vector_store, message_prompt, system_message, temperature = tpl
            else:
                ValueError("New files do not have previous configurations.")

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    cl.user_session.set('vector_store', vector_store)

    # wait for user question
    await cl.Avatar(name=botname, path="./public/logo_dark.png").send()
    await cl.Message(content=message_prompt, author=botname).send()

    cl.user_session.set('chat_history', [])
    create_agent(vector_store, temperature, system_message)



##########################################################
########### CREATING A REPLY TO A QUESTION ###############
##########################################################

@cl.on_message
async def on_message(question):

    # creating a reply
    chat_history = cl.user_session.get('chat_history')
    agent_executor = cl.user_session.get('agent_executor')

    result = await agent_executor.ainvoke(
        {"input": question.content, "chat_history": chat_history}
    )

    if verbose:
        print("This is the result of the chain:")
        print(result)

    answer = result['output']
    print("\n\n\n")
    print(answer)
    print("\n\n\n")
    answer = re.sub("^System: ", "", re.sub("^\\??\n\n", "", answer))
    chat_history.extend(
        (HumanMessage(content=question.content), AIMessage(content=answer))
    )
    cl.user_session.set('chat_history', chat_history)

    if verbose:
        print("main")
        print(f"result: {answer}")

        await cl.Message(
        content=answer, 
        #elements=process_response(res), 
        author=botname
    ).send()


# Custom Endpoints
@app.get("/botname")
def get_botname(request:Request):
    if verbose:
        print(f"calling botname: {botname}")
    return HTMLResponse(botname)