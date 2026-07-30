import os
import json
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from backend.utils import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Database")

class JsonDBCollection:
    """A mock MongoDB collection mapping to a local JSON file."""
    def __init__(self, db_path, collection_name):
        self.filepath = os.path.join(db_path, f"{collection_name}.json")
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading local JSON DB: {e}")
                self.data = []
        else:
            self.data = []
            self._save_data()

    def _save_data(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, default=str)
        except Exception as e:
            logger.error(f"Error saving local JSON DB: {e}")

    def insert_one(self, document):
        if "_id" not in document:
            import uuid
            document["_id"] = str(uuid.uuid4())
        self.data.append(document)
        self._save_data()
        return document

    def find_one(self, query):
        self._load_data()
        for doc in self.data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    def find(self, query=None):
        self._load_data()
        if not query:
            return self.data
        results = []
        for doc in self.data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                results.append(doc)
        return results

    def count_documents(self, query=None):
        self._load_data()
        if not query:
            return len(self.data)
        count = 0
        for doc in self.data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                count += 1
        return count


    def update_one(self, query, update_data):
        self._load_data()
        for doc in self.data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                # Apply $set if present, otherwise set directly
                set_vals = update_data.get("$set", update_data)
                for k, v in set_vals.items():
                    doc[k] = v
                self._save_data()
                return doc
        return None

    def delete_one(self, query):
        self._load_data()
        for i, doc in enumerate(self.data):
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                deleted = self.data.pop(i)
                self._save_data()
                return deleted
        return None

class LocalJsonDatabase:
    """Fallback database mapping to JSON collections."""
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.collections = {}

    def get_collection(self, name):
        if name not in self.collections:
            self.collections[name] = JsonDBCollection(self.db_path, name)
        return self.collections[name]

# Global DB connection manager
db_client = None
db = None
is_mongodb = False

try:
    # Attempt to connect to live MongoDB
    db_client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=2000)
    # Trigger a call to check if connection works
    db_client.admin.command('ping')
    db = db_client[config.DATABASE_NAME]
    is_mongodb = True
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.warning(f"MongoDB connection failed: {e}. Falling back to local file-based database.")
    local_db_path = os.path.join(config.BASE_DIR, "local_db")
    db = LocalJsonDatabase(local_db_path)
    is_mongodb = False

def get_collection(collection_name):
    if is_mongodb:
        return db[collection_name]
    else:
        return db.get_collection(collection_name)

def serialize_document(doc):
    if not doc:
        return None
    from bson import ObjectId
    import numpy as np
    
    def serialize_value(val):
        if isinstance(val, ObjectId):
            return str(val)
        elif isinstance(val, (np.float32, np.float64)):
            return float(val)
        elif isinstance(val, (np.int32, np.int64)):
            return int(val)
        elif isinstance(val, np.ndarray):
            return [serialize_value(item) for item in val.tolist()]
        elif isinstance(val, dict):
            return {k: serialize_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [serialize_value(item) for item in val]
        return val

    if isinstance(doc, dict):
        return {k: serialize_value(v) for k, v in doc.items()}
    return serialize_value(doc)
