#! ../venv/Scripts/python.exe
import os
import glob
import shutil
from dotenv import load_dotenv
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
load_dotenv()

# Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY", '../db')
source_directory = os.environ.get("DOCUMENT_SOURCE_DIR", '../docs')
verbose = os.environ.get("VERBOSE", 'True').lower() in ('true', '1', 't')
# aws
aws_enable = os.environ.get("AWS_ENABLE", 'False').lower() in ('true', '1', 't')
aws_access_key = os.environ.get("AWS_ACCESS_KEY", "")
aws_secret_key = os.environ.get("AWS_SECRET_KEY", "")
aws_region = os.environ.get("AWS_REGION", "us-east-1")
aws_bucket_name = os.environ.get("AWS_BUCKET_NAME", "")


class CustomOpenAIEmbeddingFunction(OpenAIEmbeddings):

    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)

    def __call__(self, input):
        return self._embed_documents(input)



class IngestData:

    def __init__(
            self, 
            api_key=None, 
            model_name="text-embedding-ada-002", 
            database_type='faiss',
            collection_name="chroma", 
            host="localhost", 
            port="8000", 
            use_client=True,
            chunk_size=256,
            chunk_overlap=10,
    ):
        # embedding function to be used in collection
        self.model_name = model_name
        self.database_type = database_type
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.use_client = use_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.database_type == 'chroma' and self.use_client:
            # self.db_client = chromadb.PersistentClient(
            #     path=persist_directory, 
            #     settings=Settings(allow_reset=True),
            # )
            # must run `chroma run --path db`` first
            self.db_client = chromadb.HttpClient(
                port=self.port,
                host=self.host,
                settings=Settings(allow_reset=True),
            )
            # define embedding model to be used for collections
            self.embeddings = CustomOpenAIEmbeddingFunction(
                openai_api_key=os.environ['OPENAI_API_KEY'] if api_key is None else api_key
            )
        else:
            self.embeddings = OpenAIEmbeddings()



    def load_document(self, filename):
        """
        Load PDF files as LangChain Documents
        """
        # filename = next(glob.iglob(os.path.join(source_dir, "**/*.pdf"), recursive=True))
        
        ext = "." + filename.rsplit(".", 1)[-1]
        if ext != '.pdf':
            raise ValueError(f"Unsupported file extension '{ext}'")

        documents = []
        loader = PyPDFLoader(filename)
        try:
            documents = loader.load()
        except Exception as err:
            print(f"ERROR: {filename}", err)
        return documents
    

    # def load_documents(self, source_dir, ignored_files=[]):
    #     """
    #     Loads all documents from the source documents directory, ignoring specified files
    #     """
    #     all_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    #     filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    #     with Pool(processes=os.cpu_count()) as pool:
    #         results = []
    #         with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
    #             for docs in pool.imap_unordered(load_document, filtered_files):
    #                 results.extend(docs)
    #                 pbar.update()
    #     return results


    def chunk_data(self, filename):
        """
        Load the document, split it and return the chunks
        """
        # load document
        documents = self.load_document(filename)
        if not documents:
            print("No new documents to load")
            exit(0)
        # split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks of text (max. {self.chunk_size} tokens each)")
        return chunks


    def build_embeddings(self, filename):
        """
        Create embeddings and save them in a Chroma vector store
        Returns the indexed db
        """
        chunks = self.chunk_data(filename)
        print("\n\n\nChunking complete...\n")
        print(f"{len(chunks)} chunks were created.\n")
        print(f"Creating embedding. May take some minutes...")

        if self.database_type == 'chroma':
            if self.use_client:
                # reset client
                self.db_client.reset()
                # create a collection
                collection = self.db_client.get_or_create_collection(
                    name=self.collection_name, 
                    embedding_function=self.embeddings,
                )
                # add text to the collection
                for doc in chunks:
                    collection.add(
                        ids=[str(uuid.uuid1())], 
                        metadatas=doc.metadata, 
                        documents=doc.page_content
                    )

                # instantiate chroma db
                vector_store = self.get_vector_store()

            else:
                # create vector store from documents using OpenAIEmbeddings
                vector_store = Chroma.from_documents(
                    chunks, 
                    self.embeddings, 
                    persist_directory=persist_directory, 
                )
        else:
            # s3_bucket = None
            # if aws_enable:
            #     session = boto3.Session(
            #         aws_access_key_id=aws_access_key,
            #         aws_secret_access_key=aws_secret_key
            #     )
            #     s3 = session.resource('s3')
            #     s3_bucket = s3.Bucket(aws_bucket_name)
            #     self.fetch_aws_documents(source_directory, s3_bucket)

            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(persist_directory)

            # zip archive the FAISS index files into source directory
            shutil.make_archive(
                os.path.join(source_directory, "index_files"),
                'zip',
                persist_directory
            )

            # if aws_enable:
            #     self.upload_database(persist_directory, s3_bucket)
            print(f"Ingestion complete!")
        return vector_store


    def get_vector_store(self):
        """
        Returns the existing vector store
        """
        if self.database_type == 'chroma':
            if self.use_client:
                vector_store = Chroma(
                    client=self.db_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory, 
                )
            else:
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                )
        else:
            vector_store = FAISS.load_local(persist_directory, self.embeddings)
        return vector_store


    def fetch_aws_documents(self, source_dir, source_bucket):

        print("fetching aws documents")
        for s3_object in source_bucket.objects.all():
            path, filename = os.path.split(s3_object.key)
            if verbose:
                print(f"downloading {path}/{filename}")
            source_bucket.download_file(s3_object.key, os.path.join(source_dir, filename))


    def upload_database(self, db_dir):

        print("upload_database")
        all_files = []
        for ext in ["faiss", "pkl"]:
            files = glob.glob(os.path.join(db_dir, f"**/*{ext}"), recursive=False)
            all_files.extend(files)