import os
import re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import chainlit as cl
from chainlit.server import app
from chainlit.context import init_http_context
from fastapi import Request
from fastapi.responses import HTMLResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from twilio.http.async_http_client import AsyncTwilioHttpClient
from lib.agents import get_agent_executor
from utils.ingest import IngestData
from utils.mongo_utils import init_connection
from utils.helpers import get_message, get_action, get_file
from utils.mongo_utils import mongodb_interaction
from utils.password_management import get_hashed_password, check_password
load_dotenv()

database_type = os.environ.get("VECTOR_STORE_TYPE", "faiss")
database_name = os.environ.get("DB_NAME", "chattabot")
user_name = os.environ.get('LOGIN_USERNAME')
hashed_password = get_hashed_password(os.environ.get("LOGIN_PASSWORD"))
botname = os.environ.get("BOTNAME", "Personal Assistant")
verbose = os.environ.get("VERBOSE", 'True') == 'True'
port = int(os.getenv("PORT", 8000))

# twilio config
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_phone = os.getenv("TWILIO_PHONE")

#########################################################
################# AUTHENTICATION ########################
#########################################################

@cl.password_auth_callback
def auth_callback(user, pw):
    if user == user_name and check_password(pw, hashed_password):
        return cl.User(
            identifier=user_name,
            metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


async def create_agent(vs, temp, sys_msg):
    global agent_executor_global
    agent_executor = await get_agent_executor(vs, temp, sys_msg)
    cl.user_session.set('agent_executor', agent_executor)
    agent_executor_global = agent_executor


##########################################################
############## CHAINLIT APP DEFINITION ###################
##########################################################

# App Hooks
@cl.on_chat_start
async def main():

    """ Startup """

    msg_pmpt = "Ask me anything!"
    sys_mg = ""
    t = 0
    vs = None
    # initiate mongodb connection
    conn = await init_connection()
    db = conn[database_name]

    # process business type
    name = await get_message("Which business are you going to ask about today?")
    # process file ingestion
    response = await get_action("Do you want to use previously uploaded file or do you want to a new file?")
    # process the response
    if response:
        # instantialize data ingester / getter
        dp = IngestData(database_type=database_type)

        if response == "new":
            files = None
            # Wait for the user to upload a file
            while files is None:
                files = await get_file("Please upload a pdf file!")
            pdf_file = files[0].path
            vs = dp.build_embeddings(pdf_file)

    db_res = await get_action("Do you want to use previous configurations or do you want to create a new one?")
    
    if db_res:
        vs, msg_pmpt, sys_mg, t = await mongodb_interaction(db_res, db, name, msg_pmpt, dp, vs)
        
    # saving the vector store in a user session
    cl.user_session.set('vector_store', vs)

    # wait for user question
    await cl.Avatar(name=botname, path="./public/logo_dark.png").send()
    await cl.Message(content=msg_pmpt, author=botname).send()

    global chat_history_global
    chat_history_global = []
    cl.user_session.set('chat_history', [])
    await create_agent(vs, t, sys_mg)

    # close db connection
    conn.close()



##########################################################
########### CREATING A REPLY TO A QUESTION ###############
##########################################################

async def create_answer(question):

    global agent_executor_global
    global chat_history_global
    chat_history = cl.user_session.get('chat_history', chat_history_global)
    agent_executor = cl.user_session.get('agent_executor', agent_executor_global)
    result = await agent_executor.ainvoke(
        {"input": question, "chat_history": chat_history}
    )
    answer = result['output']
    answer = re.sub("^System: ", "", re.sub("^\\??\n\n", "", answer))
    chat_history.extend(
        (HumanMessage(content=question), AIMessage(content=answer))
    )
    cl.user_session.set('chat_history', chat_history)
    chat_history_global = chat_history
    return answer


# @cl.on_message
# async def on_message(msg):

#     # creating a reply
#     question = msg.content
#     answer = await create_answer(question)

#     await cl.Message(
#         content=answer,
#         author=botname
#     ).send()


##########################################################
##################### API ENDPOINTS ######################
##########################################################

# custom endpoints
@app.get("/botname")
def get_botname(request: Request):
    print(request)
    if verbose:
        print(f"calling botname: {botname}")
    return HTMLResponse(botname)


# endpoint to verify the API status
@app.get("/healthcheck")
async def root():
    return {"message": "Status: OK"}


async def send_sms(message, to_phone_number):
    """ Send SMS text message and return the message id """
    http_client = AsyncTwilioHttpClient()
    client = Client(account_sid, auth_token, http_client=http_client)
    message = await client.messages.create_async(
        body=message,
        from_=from_phone,
        to=to_phone_number
    )
    return message.status, message.sid


@app.post("/sms")
async def chat(request: Request):
    """Respond to incoming calls with a simple text message."""
    
    init_http_context()
    
    response = MessagingResponse()

    # receive question in SMS
    fm = await request.form()
    to_phone_number = fm.get("From")
    question = fm.get("Body")

    # creating a reply
    answer = await create_answer(question)
    response.message(answer)
    mstatus, msid = await send_sms(answer, to_phone_number)
    print(f'Sending the answer: "{answer}" from {from_phone} to {to_phone_number}.')
    print(f"Message status: {mstatus}; message SID: {msid}\n")

    return {"status": "OK", "content": str(response), "answer": answer, "question": question}


if __name__ == "__main__":
    app.run(debug=True, port=port)