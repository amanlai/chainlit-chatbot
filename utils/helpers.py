import chainlit as cl

# callback for clicking buttons
@cl.action_callback("action_button")
async def on_action(action):
    print("The user clicked on the action button!")
    return action


async def get_message(message):
    msg = await cl.AskUserMessage(content=message, timeout=1000).send()
    return msg.get("output")


async def get_action(message):
    response = await cl.AskActionMessage(
        content=message,
        actions=[
            cl.Action(name="Previous", value="old"),
            cl.Action(name="New", value="new")
        ],
        timeout=1000
    ).send()
    return response.get("value")


async def get_file(message):
    response = await cl.AskFileMessage(
        content=message,
        accept=["application/pdf"],
        timeout=1000
    ).send()
    return response


async def get_user_input():
    name = await get_message("Which business are you going to ask about today?")
    response = await get_action("Do you want to use previously uploaded file or do you want to a new file?")
    db_res = await get_action("Do you want to use previous configurations or do you want to create a new one?")
    return name, response, db_res

