import os
import io
import zipfile
import pymongo
import certifi
from .helpers import get_message

# MongoDB config
mongo_uri = os.environ.get("MONGO_URI")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "../db")
source_directory = os.environ.get("DOCUMENT_SOURCE_DIR", "../docs")


async def init_connection():
    try:
        mongo_client = pymongo.MongoClient(
            mongo_uri,
            server_api=pymongo.server_api.ServerApi('1'),
            tlsCAFile=certifi.where()
        )
        print("Connection to MongoDB successful")
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return
    return mongo_client


async def ingest_into_collection(collection, business_name, data):
    """
    Ingest data into MongoDB
    """
    collection.delete_one({'_id': business_name})
    collection.insert_one({"_id": business_name, **data})


async def update_collection(collection, business_name, data):
    """
    Update certain key-value pairs in documents in a collection
    """
    collection.update_one(
        {'_id': business_name},
        {"$set": data}
    )


async def copy_from_collection(
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

async def mongodb_interaction(
        db_res, db, business_name, message_prompt, data_processor, vector_store
):
    # path to archived index files
    path = os.path.join(source_directory, "index_files.zip")
    collection = db[business_name]

    if db_res == "new":
        # ask for system message, message_prompt, temperature
        system_message = await get_message("System message:")
        temperature = float(await get_message("Temperature:"))

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
            await ingest_into_collection(collection, business_name, data)
        except FileNotFoundError:
            await update_collection(collection, business_name, data)
        vector_store = await copy_from_collection(data_processor, collection, business_name)
    else:
        if vector_store is None:
            tpl = await copy_from_collection(data_processor, collection, business_name, old_config=True)
            return tpl
        else:
            ValueError("New files do not have previous configurations.")
    return vector_store, message_prompt, system_message, temperature