import numpy as np
from bson import ObjectId

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

def serialize_document(doc):
    if not doc:
        return None
    if isinstance(doc, dict):
        return {k: serialize_value(v) for k, v in doc.items()}
    return serialize_value(doc)

def serialize_documents(docs):
    return [serialize_document(d) for d in docs]

def make_response(data=None, success=True, message=None):
    resp = {"success": success}
    if message is not None:
        resp["message"] = message
    if data is not None:
        resp["data"] = data
        if isinstance(data, dict):
            # Double envelope: expose keys on the root for direct frontend access
            resp.update(data)
    return resp
