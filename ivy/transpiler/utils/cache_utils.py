"""Main file for holding logic related to caching."""

# global
from __future__ import annotations
import asyncio
import threading
from types import FunctionType, MethodType
from typing import List, Dict, Set, Optional, Any, Tuple, Union, TYPE_CHECKING
from collections import OrderedDict, defaultdict
import dill
import os
from dataclasses import dataclass
import logging
from pathlib import Path
import time

# local
from . import pickling_utils

if TYPE_CHECKING:
    from ..translations.data.object_like import (
        BaseObjectLike,
    )
    from ..translations.data.global_like import (
        GlobalObjectLike,
    )
    from .ast_utils import FromImportObj, ImportObj

"""
Asyncio Helpers
"""


def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = loop.create_task(coroutine)
            return asyncio.run_coroutine_threadsafe(task, loop).result()
        else:
            return loop.run_until_complete(coroutine)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)


"""
Caching Helpers
"""

def get_ivy_root() -> Path:
    if "IVY_ROOT" not in os.environ:
        # traverse backwards through the directory tree, searching for .ivy
        current_dir = os.getcwd()
        ivy_folder = None
        
        # Keep traversing until we hit the root directory
        while current_dir != os.path.dirname(current_dir):  # Stop at root directory
            possible_ivy_folder = os.path.join(current_dir, ".ivy")
            if os.path.isdir(possible_ivy_folder):
                ivy_folder = possible_ivy_folder
                break
            current_dir = os.path.dirname(current_dir)
        
        if ivy_folder:
            os.environ["IVY_ROOT"] = ivy_folder
        else:
            # If no .ivy folder was found, create one in the cwd
            ivy_folder = os.path.join(os.getcwd(), ".ivy")
            os.mkdir(ivy_folder)
            os.environ["IVY_ROOT"] = ivy_folder
    else:
        ivy_folder = os.environ["IVY_ROOT"]
        # If the IVY_ROOT environment variable is set, check if it points to a valid .ivy folder
        if not os.path.isdir(ivy_folder):
            # If not, raise an exception explaining that the user needs to set it to a valid .ivy folder
            raise Exception(
                "IVY_ROOT environment variable is not set to a valid directory. "
                "Please create a hidden folder '.ivy' and set IVY_ROOT to this location "
                "to set up your local Ivy environment correctly."
            )
        # If the IVY_ROOT environment variable is set and points to a valid .ivy folder,
        # inform the user about preserving the tracer and transpiler caches across multiple machines
        # logging.warning(
        #     "To preserve the tracer and transpiler caches across multiple machines, ensure that "
        #     "the relative path of your projects from the .ivy folder is consistent across all machines. "
        #     "You can do this by adding .ivy to your home folder and placing all projects in the same "
        #     "place relative to the home folder on all machines."
        # )
    return Path(ivy_folder)


def ensure_cache_directory():
    """
    Create and return the path to the _cache directory for storing pickled files.
    """
    # create the _cache directory to store the pickled files
    project_root = os.getcwd()
    if "tracer-transpiler" in __file__:
        cache_dir = os.path.join(project_root, "ivy_repo", "ivy", "compiler", "_cache")
    else:
        cache_dir = os.path.join(str(get_ivy_root().absolute()), "compiler", "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cache_sort_key(unit):
    """
    Provide a sorting key for cache units based on priority functions and depth.
    """

    priority_functions = [
        "handle_get_item",
        "handle_set_item",
        "handle_methods",
        "handle_transpose_in_input_and_output",
        "store_config_info",
    ]

    function_name = unit.object_like.name

    # Check if the function name is in the priority list
    is_priority = any(function_name.__contains__(pf) for pf in priority_functions)

    # Return a tuple: (is_priority, -depth)
    # This will sort priority functions first, then by descending depth
    return (not is_priority, -unit.depth)


"""
Core Data Structures
"""


@dataclass
class AtomicCacheUnit:
    """
    A dataclass representing a unit of cached data for code generation.

    Attributes:
        ast_root (Any): The ast_root object for code generation.
        object_like (BaseObjectLike): The object-like representation.
        cacher (Cacher): The cacher instance used.
        ast_transformer (Transformer): The AST transformer instance used.
        depth (int): The depth of this unit in the object hierarchy.

    Methods:
        generate_source_code(unit, output_dir): Static method to generate source code.
        __repr__(): Returns a string representation of the AtomicCacheUnit.
    """

    ast_root: Any
    object_like: "BaseObjectLike"
    globals: List[GlobalObjectLike]
    imports: set[ImportObj]
    from_imports: set[FromImportObj]
    circular_reference_object_likes: Set[BaseObjectLike]
    source: str
    target: str
    object_like_bytes_to_translated_object_str_cache: (
        ObjectLikeBytesToTranslatedObjectStringCache
    )
    import_statement_cache: ImportStatementCache
    global_statement_cache: GlobalStatementCache
    emitted_source_cache: EmittedSourceCache
    depth: int

    @staticmethod
    def generate_source_code(
        unit: AtomicCacheUnit,
        output_dir: str,
        base_output_dir: str,
        from_cache: bool = False,
    ):
        from ivy.transpiler.utils import ast_utils

        return ast_utils.generate_source_code(
            ast_root=unit.ast_root,
            object_like=unit.object_like,
            globals=unit.globals,
            imports=unit.imports,
            from_imports=unit.from_imports,
            circular_reference_object_likes=unit.circular_reference_object_likes,
            source=unit.source,
            target=unit.target,
            object_like_bytes_to_translated_object_str_cache=unit.object_like_bytes_to_translated_object_str_cache,
            import_statement_cache=unit.import_statement_cache,
            global_statement_cache=unit.global_statement_cache,
            emitted_source_cache=unit.emitted_source_cache,
            output_dir=output_dir,
            base_output_dir=base_output_dir,
            from_cache=from_cache,
        )

    def __repr__(self):
        name = self.object_like.qualname
        return f"AtomicCacheUnit({name})"


class InMemoryCache:
    """
    An in-memory cache for storing objects

    Attributes:
        _cache (dict): Stores the cached items with top-level keys.
        locks (dict): Locks for synchronizing access to each sub-cache.

    Methods:
        __init__(): Initializes the cache.
        get(top_key, sub_key): Retrieves an item from the cache.
        set(top_key, sub_key, value): Adds an item to the cache.
        update(top_key, updates): Updates multiple items in a sub-cache.
        exist(top_key, sub_key): Checks if an item exists in a sub-cache.
        clear(): Clears the entire cache.
        get_all(top_key): Retrieves all items from a sub-cache.
    """

    def __init__(self):
        """
        Initializes the in-memory cache with a specified capacity.
        """
        self._cache = {}
        self.locks = {}

    def _get_lock(self, top_key=None):
        """
        Retrieves a lock for a specific top-level key, creating one if necessary.

        Args:
            top_key: The top-level key for which to retrieve the lock.

        Returns:
            A lock for the specified top-level key.
        """
        if top_key is None:
            return threading.Lock()
        if top_key not in self.locks:
            lock = threading.RLock()
            self.locks[top_key] = lock
        return self.locks[top_key]

    def get_keys(self):
        return list(self._cache.keys())

    def get(self, top_key, sub_key, default=None):
        """
        Retrieves an item from the cache.

        Args:
            top_key: The top-level key of the sub-cache.
            sub_key: The sub-key of the item to retrieve.

        Returns:
            The cached item if found, otherwise None.
        """
        with self._get_lock(top_key):
            if (
                top_key not in self._cache
                or sub_key
                and sub_key not in self._cache[top_key]
            ):
                return default
            return self._cache[top_key][sub_key]

    def set(self, top_key, sub_key, value):
        """
        Sets an item in the cache against the given top-level key and sub-key.

        Args:
            top_key: The top-level key of the sub-cache.
            sub_key: The sub-key under which the item is stored.
            value: The item to cache.
        """
        with self._get_lock(top_key):
            if top_key not in self._cache:
                self._cache[top_key] = OrderedDict()
            self._cache[top_key][sub_key] = value

    def update(self, top_key, updates: dict):
        """
        Updates the cache with multiple items.

        Args:
            top_key: The top-level key of the sub-cache.
            updates (dict): A dictionary containing the items to update in the cache.
        """
        with self._get_lock(top_key):
            if top_key not in self._cache:
                self._cache[top_key] = OrderedDict()
        for sub_key, value in updates.items():
            self.set(top_key, sub_key, value)

    def exist(self, top_key, sub_key):
        """
        Checks if an item exists in the cache.

        Args:
            top_key: The top-level key of the sub-cache.
            sub_key: The sub-key of the item to check for.

        Returns:
            True if item exists, otherwise False.
        """
        with self._get_lock(top_key):
            return top_key in self._cache and sub_key in self._cache[top_key]

    def clear(self):
        """
        Clears the entire cache.
        """
        with self._get_lock():
            self._cache.clear()

    def get_all(self, top_key):
        """
        Retrieves all items with keys starting with a given top_key.

        Args:
            top_key: The top_key to match keys against.

        Returns:
            A dictionary of matching items.
        """
        with self._get_lock(top_key):
            if top_key not in self._cache:
                return {}
            return self._cache[top_key]


class PreloadedObjectCache:
    """
    A cache system for storing and retrieving preloaded objects and their associated units.

    This class manages a cache of AtomicCacheUnit objects, providing methods for adding,
    retrieving, and persisting cache data. It also handles serialization and deserialization
    of cache contents. This class manages in-memory caching.

    Attributes:
        _cache (InMemoryCache): An in-memory cache for fast access.
        _cache_dir (str): Directory for storing cache files.
        lock (Lock): Lock for synchronizing access to the in-memory cache.

    Methods:
        add(key, unit, current_call_tree=None): Add a unit to the cache.
        get(key): Retrieve units for a given key.
        exist(key): Check if a key exists in the cache.
        save_file(cache_file, cache_data): Save the cache to a file.
        load_file(cache_file): Load the cache from a file.
        pre_load_cache(): Load a preloaded cache.
        save_preloaded_cache(): Save the current cache as preloaded.
        clear(): Clear the cache.
    """

    def __init__(self):
        self._cache = InMemoryCache()
        self._cache_dir = ensure_cache_directory()
        self._cache_files = self._get_cache_files()
        self._no_cache = len(self._cache_files) == 0
        self.lock = threading.Lock()
        # Event to signal when cache is ready
        self.cache_populated = False
        self.cache_ready = {
            cache_file: threading.Event() for cache_file in self._cache_files
        }

    def _to_top_key(self, source: str, target: str) -> str:
        # Generates a qualified key that is the filename for the
        # .pkl file for the corresponding stage of the translator
        # based on `source` and `target`
        return f"{source}_to_{target}_translation_cache.pkl"

    def _from_top_key(self, top_key: str) -> Tuple[str, str]:
        # Extracts the source and target from a given cache filename string.
        main_part = top_key[: -len("_translation_cache.pkl")]

        # Split by '_to_'
        parts = main_part.split("_to_")

        if len(parts) != 2:
            raise ValueError("Invalid cache filename format")

        source, target = parts
        return source, target

    def _get_cache_files(self, saving: bool = False) -> List[str]:
        # If we are saving/updating the cache
        # we will save the .pkl files corresponding to
        # the source, target pairs that were seen during
        # translation
        if saving:
            return self._cache.get_keys()
        elif getattr(self, "_cache_files", None) and self._cache_files:
            return self._cache_files
        else:
            self._cache_files = [
                cache_file
                for cache_file in os.listdir(self._cache_dir)
                if cache_file.endswith(".pkl")
            ]
            return self._cache_files

    def get(
        self,
        key: Union[MethodType, FunctionType, type],
        source: str = "torch",
        target: str = "torch_frontend",
    ):
        # Generate the top level cache key indicating the current stage of translation
        top_key = self._to_top_key(source=source, target=target)

        # Ensure cache is loaded
        if not self._no_cache and top_key in self.cache_ready:
            self.cache_ready[top_key].wait()

        key_hash = pickling_utils.get_object_hash(key)

        # Generate the top level cache key indicating the current stage of translation
        top_key = self._to_top_key(source=source, target=target)

        return self._cache.get(top_key, key_hash)

    def set(
        self,
        key: Union[MethodType, FunctionType, type],
        unit: AtomicCacheUnit,
        current_call_tree: Union[MethodType, FunctionType, type] = None,
        source: str = "torch",
        target: str = "torch_frontend",
    ):
        # Generate the top level cache key indicating the current stage of translation
        top_key = self._to_top_key(source=source, target=target)

        # Ensure cache is loaded
        if not self._no_cache and top_key in self.cache_ready:
            self.cache_ready[top_key].wait()

        key_hash = pickling_utils.get_object_hash(key)

        key_hash = (
            key_hash
            if unit.depth == 0 or current_call_tree is None
            else pickling_utils.get_object_hash(current_call_tree)
        )

        with self.lock:
            # Add to InMemoryCache
            existing_units = self._cache.get(top_key, key_hash, [])
            existing_units.append(unit)
            self._cache.set(top_key, key_hash, existing_units)

    def update(
        self,
        updates: dict,
        source: str = "torch",
        target: str = "torch_frontend",
    ):
        # Generate the top level cache key indicating the current stage of translation
        top_key = self._to_top_key(source=source, target=target)

        with self.lock:
            self._cache.update(top_key, updates)

    def exist(
        self,
        key: Union[MethodType, FunctionType, type],
        source: str = "torch",
        target: str = "torch_frontend",
    ) -> bool:
        # Generate the top level cache key indicating the current stage of translation
        top_key = self._to_top_key(source=source, target=target)

        # Ensure cache is loaded
        if not self._no_cache and top_key in self.cache_ready:
            self.cache_ready[top_key].wait()

        key_hash = pickling_utils.get_object_hash(key)

        with self.lock:
            return self._cache.exist(top_key, key_hash)

    def get_all(
        self,
        source: str = "torch",
        target: str = "torch_frontend",
    ):
        # Generate the top level cache key indicating the current stage of translation
        top_key = self._to_top_key(source=source, target=target)

        # Ensure cache is loaded
        if not self._no_cache and top_key in self.cache_ready:
            self.cache_ready[top_key].wait()

        with self.lock:
            return self._cache.get_all(top_key)

    async def save_file_async(self, cache_file, cache_data):
        try:
            logging.debug(f"Saving file: {cache_file}")
            cache_file_path = os.path.join(self._cache_dir, cache_file)
            with open(cache_file_path, "wb") as f:
                dill.dump(cache_data, f)
            logging.debug(f"Finished saving file: {cache_file}")
        except Exception as e:
            logging.debug(f"Error saving file {cache_file}: {e}")
            raise

    def save_file(self, cache_file, cache_data):
        try:
            logging.debug(f"Saving file: {cache_file}")
            cache_file_path = os.path.join(self._cache_dir, cache_file)
            with open(cache_file_path, "wb") as f:
                dill.dump(cache_data, f)
            logging.debug(f"Finished saving file: {cache_file}")
        except Exception as e:
            logging.debug(f"Error saving file {cache_file}: {e}")
            raise

    async def load_file_async(self, cache_file):
        try:
            logging.debug(f"Loading file: {cache_file}")
            cache_file_path = os.path.join(self._cache_dir, cache_file)
            if os.path.exists(cache_file_path):
                with open(cache_file_path, "rb") as f:
                    result = dill.load(f)
                logging.debug(f"Finished loading file: {cache_file}")
                return result
            else:
                logging.debug(
                    f"Cache file {cache_file_path} not found. Starting with an empty cache."
                )
                return {}
        except Exception as e:
            logging.debug(f"Error loading file {cache_file}: {e}")
            raise

    def load_file(self, cache_file):
        try:
            logging.debug(f"Loading file: {cache_file}")
            cache_file_path = os.path.join(self._cache_dir, cache_file)
            if os.path.exists(cache_file_path):
                with open(cache_file_path, "rb") as f:
                    result = dill.load(f)
                logging.debug(f"Finished loading file: {cache_file}")
                return result
            else:
                logging.debug(
                    f"Cache file {cache_file_path} not found. Starting with an empty cache."
                )
                return {}
        except Exception as e:
            logging.debug(f"Error loading file {cache_file}: {e}")
            raise

    async def deserialize_and_cache_async(self, cache_file, cache_data):
        # Deserialize cache data and set it in both caches
        source, target = self._from_top_key(cache_file)
        try:
            logging.debug(f"Deserializing cache for file: {cache_file}")
            deserialized_data = await asyncio.get_event_loop().run_in_executor(
                None,
                pickling_utils.ACUPickler.deserialize_cache,
                cache_data,
                cache_file,
            )

            # Update the cache with the deserialized data
            self.update(deserialized_data, source=source, target=target)

            # Signal that the cache is ready
            if cache_file in self.cache_ready:
                self.cache_ready[cache_file].set()

            logging.debug(f"Finished deserializing and caching for file: {cache_file}")
        except Exception as e:
            logging.debug(f"Error deserializing and caching file {cache_file}: {e}")
            raise e

    def deserialize_and_cache(self, cache_file, cache_data):
        # Deserialize cache data and set it in both caches
        source, target = self._from_top_key(cache_file)
        try:
            logging.debug(f"Deserializing cache for file: {cache_file}")
            deserialized_data = pickling_utils.ACUPickler.deserialize_cache(
                cache_data, cache_file
            )

            # Update the cache with the deserialized data
            self.update(deserialized_data, source=source, target=target)

            # Signal that the cache is ready
            if cache_file in self.cache_ready:
                self.cache_ready[cache_file].set()

            logging.debug(f"Finished deserializing and caching for file: {cache_file}")
        except Exception as e:
            logging.debug(f"Error deserializing and caching file {cache_file}: {e}")
            raise e

    async def get_cache_and_serialize_async(self, cache_file):
        # Fetch the catch data against the cache file
        source, target = self._from_top_key(cache_file)
        cache_data = self.get_all(source=source, target=target)

        # Empty the object_module list to avoid pickling issues.
        def clear_obj_mod(transformer):
            transformer.object_module = []

        for units in cache_data.values():
            [clear_obj_mod(unit.ast_transformer) for unit in units]

        try:
            logging.debug(
                f"Serializing cache for file: {cache_file} with len(cache_data): {len(cache_data)}"
            )
            serialized_data = await asyncio.get_event_loop().run_in_executor(
                None, pickling_utils.ACUPickler.serialize_cache, cache_data, cache_file
            )
            logging.debug(f"Finished seserializing for file: {cache_file}")
            return serialized_data
        except Exception as e:
            logging.debug(f"Error serializing file {cache_file}: {e}")
            raise e

    def get_cache_and_serialize(self, cache_file):
        # Fetch the catch data against the cache file
        source, target = self._from_top_key(cache_file)
        cache_data = self.get_all(source=source, target=target)

        try:
            logging.debug(
                f"Serializing cache for file: {cache_file} with len(cache_data): {len(cache_data)}"
            )
            serialized_data = pickling_utils.ACUPickler.serialize_cache(
                cache_data, cache_file
            )
            logging.debug(f"Finished seserializing for file: {cache_file}")
            return serialized_data
        except Exception as e:
            logging.debug(f"Error serializing file {cache_file}: {e}")
            raise e

    async def load_preloaded_cache_async(self):
        """
        Loads all preloaded cache files into the in-memory cache.

        This method loads data from cache files, deserializes them, and stores
        them in the in-memory cache.
        """
        start_time = time.time()
        cache_files = self._get_cache_files()
        try:
            logging.debug("Starting to load preloaded cache files.")
            cache_data = await asyncio.gather(
                *[self.load_file_async(file) for file in cache_files]
            )
            logging.debug("Finished loading all cache files.")
            await asyncio.gather(
                *[
                    self.deserialize_and_cache_async(file, data)
                    for file, data in zip(cache_files, cache_data)
                ]
            )
            logging.debug("Finished loading and deserializing all cache files.")
        except asyncio.CancelledError:
            logging.debug("Cache loading was cancelled.")
        except Exception as e:
            logging.debug(f"Unexpected error during cache loading: {e}")
            raise
        finally:
            self.cache_populated = True
        pre_load_cache_time = time.time() - start_time
        logging.debug("Cache preload completed.")
        logging.debug(
            f"Time taken for pre_load_cache: {pre_load_cache_time:.6f} seconds"
        )

    def load_preloaded_cache(self):
        """
        Loads all preloaded cache files into the in-memory cache.

        This method loads data from cache files, deserializes them, and stores
        them in the in-memory cache.
        """
        start_time = time.time()
        cache_files = self._get_cache_files()
        try:
            logging.debug("Starting to load preloaded cache files.")
            cache_data = [self.load_file(file) for file in cache_files]
            logging.debug("Finished loading all cache files.")
            [
                self.deserialize_and_cache(file, data)
                for file, data in zip(cache_files, cache_data)
            ]
            logging.debug("Finished loading and deserializing all cache files.")
        except Exception as e:
            logging.debug(f"Unexpected error during cache loading: {e}")
            raise
        finally:
            self.cache_populated = True
        pre_load_cache_time = time.time() - start_time
        logging.debug("Cache preload completed.")
        logging.debug(
            f"Time taken for pre_load_cache: {pre_load_cache_time:.6f} seconds"
        )

    async def save_preloaded_cache_async(self):
        """
        Saves all preloaded cache files into the in-memory cache.

        This method loads data from cache files, deserializes them, and stores
        them in the in-memory cache.
        """
        start_time = time.time()
        cache_files = self._get_cache_files(saving=True)
        try:
            logging.debug(f"Starting to save preloaded cache files: {cache_files}.")
            cache_data = await asyncio.gather(
                *[self.get_cache_and_serialize_async(file) for file in cache_files]
            )
            logging.debug("Finished serializing all cache files.")
            await asyncio.gather(
                *[
                    self.save_file_async(file, data)
                    for file, data in zip(cache_files, cache_data)
                ]
            )
        except asyncio.CancelledError:
            logging.debug("Cache saving was cancelled.")
        except Exception as e:
            logging.debug(f"Unexpected error during cache saving: {e}")
            raise
        save_pre_loaded_cache_time = time.time() - start_time
        logging.debug("Preloaded cache save completed.")
        logging.debug(
            f"Time taken for saving pre_loaded_cache: {save_pre_loaded_cache_time:.6f} seconds"
        )

    def save_preloaded_cache(self, cache_files: Union[str, List[str]] = None):
        """
        Saves all preloaded cache files into the in-memory cache.

        This method loads data from cache files, deserializes them, and stores
        them in the in-memory cache.
        """
        start_time = time.time()
        if cache_files and not isinstance(cache_files, list):
            cache_files = [cache_files]
        cache_files = (
            self._get_cache_files(saving=True) if not cache_files else cache_files
        )
        try:
            logging.debug(f"Starting to save preloaded cache files: {cache_files}.")
            cache_data = [self.get_cache_and_serialize(file) for file in cache_files]
            logging.debug("Finished serializing all cache files.")
            [self.save_file(file, data) for file, data in zip(cache_files, cache_data)]
        except Exception as e:
            logging.debug(f"Unexpected error during cache saving: {e}")
            raise
        save_pre_loaded_cache_time = time.time() - start_time
        logging.debug("Preloaded cache save completed.")
        logging.debug(
            f"Time taken for saving pre_loaded_cache: {save_pre_loaded_cache_time:.6f} seconds"
        )

    def start_cache_loading_async(self):
        if self.cache_populated:
            return
        logging.debug("Starting cache preload.")
        thread = threading.Thread(target=self._run_cache_loading, daemon=True)
        thread.start()

    def start_cache_loading(self):
        if self.cache_populated:
            return
        logging.debug("Starting cache preload.")
        self.load_preloaded_cache()

    def start_cache_saving_async(self):
        logging.debug("Starting cache saving.")
        thread = threading.Thread(target=self._run_cache_saving, daemon=True)
        thread.start()

    def start_cache_saving(self):
        logging.debug("Starting cache saving.")
        self.save_preloaded_cache()

    def _run_cache_loading(self):
        try:
            run_async(self.load_preloaded_cache_async())
        except Exception as e:
            logging.debug(f"Error in cache loading thread: {e}")
            raise e

    def _run_cache_saving(self):
        try:
            run_async(self.save_preloaded_cache_async())
        except Exception as e:
            logging.debug(f"Error in cache saving thread: {e}")
            raise e

    def clear(self):
        self._cache.clear()


PRELOAD_CACHE = PreloadedObjectCache()


class ProgramCache:
    """
    Base class for caching the transformed functions/classes. The instance of this
    class is
    """

    _singleton_instance = None

    def __new__(cls, *args, singleton=False, **kwargs):
        if not singleton:
            return super(ProgramCache, cls).__new__(cls, *args, **kwargs)

        if cls._singleton_instance is None:
            cls._singleton_instance = super(ProgramCache, cls).__new__(
                cls, *args, **kwargs
            )
        return cls._singleton_instance

    def __init__(self, **kwargs) -> None:
        self._cache = {}

    def clear(self):
        self._cache = {}

    def cache(self, obj, func):
        self._cache[obj] = func

    def get(self, obj, default=[]):
        return self._cache.get(obj) or default

    def exist(self, obj):
        return obj in self._cache


class CodeToAstCache(ProgramCache):
    """
    Derived class for caching the AST tree for a given source code.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(CodeToAstCache, *args, singleton=singleton, **kwargs)


class ObjectLikeBytesToTranslatedObjectStringCache(ProgramCache):
    """
    Derived class for caching the translated name string against the func-like/type-like object dumped strings.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(
            ObjectLikeBytesToTranslatedObjectStringCache,
            *args,
            singleton=singleton,
            **kwargs,
        )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._id_to_cache_object_like_bytes = {}

    def cache(self, object_like: BaseObjectLike, func_str: str):
        object_like_dump = object_like.dumps()
        super().cache(object_like_dump, func_str)
        key_hash = pickling_utils.get_object_hash(object_like)
        self._id_to_cache_object_like_bytes[key_hash] = object_like_dump

    def get(self, object_like: BaseObjectLike) -> str:
        key_hash = pickling_utils.get_object_hash(object_like)
        cached_obj_like_bytes = self._id_to_cache_object_like_bytes.get(key_hash, None)
        return super().get(cached_obj_like_bytes, [])

    def exist(self, object_like: BaseObjectLike) -> bool:
        key_hash = pickling_utils.get_object_hash(object_like)
        return key_hash in self._id_to_cache_object_like_bytes

    def clear(self):
        super().clear()
        self._id_to_cache_object_like_bytes = {}


class SuperMethodsInectedClassObjectIdsCache(ProgramCache):
    """
    Derived class for caching the ids of type-like objects for whom the
    super (base) class methods have been injected. Used with frontend classes.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(
            SuperMethodsInectedClassObjectIdsCache,
            *args,
            singleton=singleton,
            **kwargs,
        )

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._cache = set()

    def clear(self):
        self._cache = set()

    def cache(self, obj):
        self._cache.add(obj)

    def get(self, obj):
        pass


class GlobalObjectCache(ProgramCache):
    """
    Derived class for caching globals defined and/or imported
    in the source code. Used by the BaseGlobalsTransformer.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(GlobalObjectCache, *args, singleton=singleton, **kwargs)

    def __init__(self, **kwargs):
        super().__init__()
        self._cache = {}

    def clear(self):
        self._cache = {}

    def cache(self, target_str, glob_obj):
        self._cache[target_str] = glob_obj

    def get(self, glob_obj):
        return self._cache.get(glob_obj, None)

    def exist(self, glob_obj):
        return glob_obj in self._cache


class GlobalStatementCache(ProgramCache):
    """
    Derived class for caching global statements injected into the source code.
    Used by the _inject_globals helper in ast_utils.py.
    NOTE: this is different from the GlobalObjectCache class which is a cache
    for global objects(GlobalObjectLike) captured during the transformation process.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(
            GlobalStatementCache, *args, singleton=singleton, **kwargs
        )

    def __init__(self, **kwargs):
        super().__init__()
        self._cache = defaultdict(
            set
        )  # mapping of module name --> set of global statements

    def clear(self):
        self._cache = defaultdict(set)

    def cache(self, filename, glob_stmt):
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        self._cache[filename].add(glob_stmt)

    def get(self, filename, glob_stmt):
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        global_statements = self._cache.get(filename, set())
        if glob_stmt in global_statements:
            return glob_stmt
        return None

    def exist(self, filename, glob_stmt):
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        return glob_stmt in self._cache[filename]


class ImportStatementCache(ProgramCache):
    """
    Derived class for caching import statements injected into the source code.
    Used by the _inject_builtin_imports helper in ast_utils.py.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(
            ImportStatementCache, *args, singleton=singleton, **kwargs
        )

    def __init__(self, **kwargs):
        super().__init__()
        self._cache = defaultdict(
            set
        )  # mapping of module name --> set of import statements

    def clear(self):
        self._cache = defaultdict(set)

    def cache(self, filename, import_stmt):
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        self._cache[filename].add(import_stmt)

    def get(self, filename, import_stmt):
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        import_statements = self._cache.get(filename, set())
        if import_stmt in import_statements:
            return import_stmt
        return None

    def exist(self, filename, import_stmt):
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        return import_stmt in self._cache[filename]


class EmittedSourceCache(ProgramCache):
    """
    Derived class for caching translated objects for which we have already emitted the source code.
    The reason for having this cache is to avoid duplicating a function/class within the same module.
    Used by the generate_source_code helper in ast_utils.py.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(EmittedSourceCache, *args, singleton=singleton, **kwargs)

    def __init__(self, **kwargs):
        super().__init__()
        self._forward_cache: Dict[str, Set[str]] = defaultdict(
            set
        )  # module_name -> set of obj_hashes
        self._reverse_cache: Dict[str, Set[str]] = defaultdict(
            set
        )  # obj_hash -> set of module_name

    def clear(self):
        self._forward_cache = defaultdict(set)
        self._reverse_cache = defaultdict(set)

    def cache(self, filename: str, obj_hash: str):
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        self._forward_cache[filename].add(obj_hash)
        self._reverse_cache[obj_hash].add(filename)

    def get(self, filename: str, obj_hash: str) -> Optional[str]:
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        hash_set = self._forward_cache.get(filename, set())
        if obj_hash in hash_set:
            return obj_hash
        return None

    def get_modules(self, filename: str, obj_hash: str) -> Optional[str]:
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        module_set = self._reverse_cache.get(obj_hash, set())
        if filename in module_set:
            return module_set
        return set()

    def exist(self, filename: str, obj_hash: str) -> bool:
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        return obj_hash in self._forward_cache[filename]

    def exist_hash(self, filename: str, obj_hash: str) -> bool:
        assert filename.endswith(".py"), "Filename must end with .py, got: {}".format(
            filename
        )
        return filename in self._reverse_cache[obj_hash]


class SuperMethodToImportReprsCache(ProgramCache):
    """
    Derived class for caching the import strings used by a super class method.
    Used with frontend classes.
    """

    def __new__(cls, *args, singleton=False, **kwargs):
        return super().__new__(
            SuperMethodToImportReprsCache, *args, singleton=singleton, **kwargs
        )

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._cache = defaultdict(list)

    def clear(self):
        self._cache = defaultdict(list)


class Cacher:
    """Main class responsible for holding all the caches needed by S2S."""

    def __init__(
        self,
        code_to_ast_cache: Optional[CodeToAstCache] = None,
        object_like_bytes_to_translated_object_str_cache: Optional[
            ObjectLikeBytesToTranslatedObjectStringCache
        ] = None,
        super_methods_injected_cls_object_ids_cache: Optional[
            SuperMethodsInectedClassObjectIdsCache
        ] = None,
        super_method_to_import_reprs_cache: Optional[
            SuperMethodToImportReprsCache
        ] = None,
        globals_cache: Optional[GlobalObjectCache] = None,
        singleton: bool = True,
    ) -> None:
        # TODO: Refactor this s.t. translator-specific caches are loaded
        # based on the source and target pair e.g. if frontends are involved,
        # we need to load the super method caches as well so this should be configurable
        self.code_to_ast_cache: CodeToAstCache = (
            code_to_ast_cache
            if code_to_ast_cache
            else CodeToAstCache(singleton=singleton)
        )
        self.object_like_bytes_to_translated_object_str_cache: (
            ObjectLikeBytesToTranslatedObjectStringCache
        ) = (
            object_like_bytes_to_translated_object_str_cache
            if object_like_bytes_to_translated_object_str_cache
            else ObjectLikeBytesToTranslatedObjectStringCache(singleton=singleton)
        )
        self.super_methods_injected_cls_object_ids_cache: (
            SuperMethodsInectedClassObjectIdsCache
        ) = (
            super_methods_injected_cls_object_ids_cache
            if super_methods_injected_cls_object_ids_cache
            else SuperMethodsInectedClassObjectIdsCache(singleton=singleton)
        )
        self.super_method_to_import_reprs_cache: SuperMethodToImportReprsCache = (
            super_method_to_import_reprs_cache
            if super_method_to_import_reprs_cache
            else SuperMethodToImportReprsCache(singleton=singleton)
        )
        self.globals_cache: GlobalObjectCache = (
            globals_cache if globals_cache else GlobalObjectCache(singleton=singleton)
        )
        self.global_statement_cache: GlobalStatementCache = GlobalStatementCache(
            singleton=singleton
        )
        self.import_statement_cache: ImportStatementCache = ImportStatementCache(
            singleton=singleton
        )
        self.emitted_source_cache: EmittedSourceCache = EmittedSourceCache(
            singleton=singleton
        )
        self.caches: List[ProgramCache] = [
            self.code_to_ast_cache,
            self.object_like_bytes_to_translated_object_str_cache,
            self.super_methods_injected_cls_object_ids_cache,
            self.super_method_to_import_reprs_cache,
            self.globals_cache,
            self.global_statement_cache,
            self.import_statement_cache,
            self.emitted_source_cache,
        ]

    def clear(self):
        for cache in self.caches:
            cache.clear()
