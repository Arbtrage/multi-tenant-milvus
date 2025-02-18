from datetime import datetime
from functools import lru_cache
from typing import Dict
import threading
import time
from pymilvus import Collection

from app.core.config import COLLECTIONS_CONFIG, COLLECTION_TIMEOUT

collection_access_times: Dict[str, datetime] = {}

_background_task_running = False


@lru_cache(maxsize=len(COLLECTIONS_CONFIG))
def get_collection(collection_name: str) -> Collection:
    collection_access_times[collection_name] = datetime.now()
    collection = Collection(collection_name)
    collection.load()
    return collection


def check_and_unload_inactive_collections():
    current_time = datetime.now()

    for name, last_access in list(collection_access_times.items()):
        if current_time - last_access > COLLECTION_TIMEOUT:
            try:
                collection = Collection(name)
                collection.release()
                print(f"Unloaded collection {name}")
                get_collection.cache_clear()
                del collection_access_times[name]
            except Exception as e:
                print(f"Error unloading collection {name}: {e}")


def start_background_unloader():
    global _background_task_running
    if _background_task_running:
        return

    def unloader_task():
        global _background_task_running
        while True:
            check_and_unload_inactive_collections()
            time.sleep(60)

    thread = threading.Thread(target=unloader_task, daemon=True)
    thread.start()
    _background_task_running = True