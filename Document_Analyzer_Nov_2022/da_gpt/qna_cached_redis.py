# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:32:50 2023

@author: ELECTROBOT
"""

import redis
import pickle

# Connect to the Redis cache system
r = redis.Redis(host='localhost', port=6379, db=0)

def load_qna_cached():
    global qna_cached

    # Check if the QNA object is in the cache
    qna_cached = r.get("qna")
    if qna_cached:
        # Deserialize the QNA object if it's in the cache
        qna_cached = pickle.loads(qna_cached)
    else:
        # Load the QNA object from disk if it's not in the cache
        with open("qna.pickle", "rb") as f:
            qna_cached = pickle.load(f)

        # Store the QNA object in the cache for subsequent requests
        r.set("qna", pickle.dumps(qna_cached))