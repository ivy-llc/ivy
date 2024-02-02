from pymongo import MongoClient
from bson import ObjectId

import json
import sys


def _write_auth_to_file(endpoint, obj_id):
    client = MongoClient(endpoint)
    db = client.gcp_auth
    collection = db.creds
    object_id_to_find = ObjectId(str(obj_id))

    creds = collection.find_one({"_id": object_id_to_find})
    del creds['_id']

    with open("gcp_auth.json", "w") as f:
        json.dump(creds, f, indent=4)


if __name__ == "__main__":
    endpoint, obj_id = sys.argv[1], sys.argv[2]
    _write_auth_to_file(endpoint, obj_id)
