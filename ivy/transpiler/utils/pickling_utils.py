"""Main file for running serialization/deserialization of the cache."""

# global
import hashlib
import inspect
from typing import Dict, List
import gast
from tqdm import tqdm
import inspect
import hashlib
import types
import logging

# local
from .type_utils import Types


def get_object_hash(obj):
    """
    Generate a hash for a given object, object-like, function, method, class or builtins.
    """
    from ..translations.data.object_like import BaseObjectLike
    from .cache_utils import AtomicCacheUnit

    # If the object is a BaseObjectLike, use its hash method
    if isinstance(obj, BaseObjectLike):
        return obj.get_object_hash()

    # If the object is an AtomicCacheUnit, get hash for its object_like
    if isinstance(obj, AtomicCacheUnit):
        if isinstance(obj.object_like, dict):
            return obj.object_like["object_hash"]
        else:
            return get_object_hash(obj.object_like)

    # Use a stable attribute for hashing enum instances
    if isinstance(obj, Types):
        return hashlib.md5(f"{obj.__class__.__name__}.{obj.name}".encode()).hexdigest()

    try:
        if inspect.isfunction(obj) or inspect.isclass(obj):
            try:
                obj = inspect.unwrap(obj)
                source = inspect.getsource(obj)
                return hashlib.md5(source.encode()).hexdigest()
            except (OSError, TypeError):
                # Fallback to use the qualified name and source code for hashing functions/methods
                return hashlib.md5(
                    f"{obj.__module__}.{obj.__qualname__}".encode()
                ).hexdigest()
        elif isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            return hashlib.md5(
                f"{obj.__module__}.{obj.__qualname__}".encode()
            ).hexdigest()
        elif isinstance(obj, types.MethodType):
            return hashlib.md5(
                f"{obj.__func__.__module__}.{obj.__func__.__qualname__}".encode()
            ).hexdigest()
        elif isinstance(obj, (int, str, bool, type(None))):
            return hashlib.md5(str(obj).encode()).hexdigest()
        elif isinstance(obj, float):
            return hashlib.md5(
                f"{obj:.10f}".encode()
            ).hexdigest()  # Round to 10 decimal places
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return hashlib.md5(
                "|".join([get_object_hash(item) for item in obj]).encode()
            ).hexdigest()
        elif isinstance(obj, dict):
            # Convert each (key, value) tuple to a string before joining
            return hashlib.md5(
                "|".join(
                    [
                        f"{get_object_hash(k)}:{get_object_hash(v)}"
                        for k, v in obj.items()
                    ]
                ).encode()
            ).hexdigest()
        elif hasattr(obj, "__hash__") and callable(obj.__hash__):
            return hashlib.md5(str(hash(obj)).encode()).hexdigest()
        else:
            return hashlib.md5(str(obj).encode()).hexdigest()
    except Exception as e:
        logging.error(f"Error hashing object {obj}: {str(e)}")
        return hashlib.md5(str(type(obj)).encode()).hexdigest()


def is_pickleable(obj) -> bool:
    try:
        import dill

        dill.dumps(obj)
        return True
    except:
        return False


class ACUPickler:
    @staticmethod
    def deserialize_acu(unit):
        """
        Deserialize the object-like properties of a unit, including object-like and global-object-like.
        """
        from ivy.transpiler.translations.data.object_like import (
            BaseObjectLike,
        )
        from ivy.transpiler.translations.data.global_like import (
            GlobalObjectLike,
        )

        # Handle unit.ast_root
        unit.ast_root = (
            unit.ast_root
            if not isinstance(unit.ast_root, str)
            else gast.parse(unit.ast_root)
        )

        # Handle unit.object_like
        unit.object_like: BaseObjectLike = (  # type: ignore
            unit.object_like
            if isinstance(unit.object_like, BaseObjectLike)
            else BaseObjectLike.from_dict(unit.object_like)
        )

        # Handle unit.globals
        unit.globals: List[GlobalObjectLike] = [  # type: ignore
            (
                global_object_like
                if isinstance(global_object_like, GlobalObjectLike)
                else GlobalObjectLike.from_dict(global_object_like)
            )
            for global_object_like in unit.globals
        ]

        return unit

    @staticmethod
    def deserialize_cache(cache: Dict, cache_file: str = ""):
        from ivy.transpiler.utils.cache_utils import AtomicCacheUnit

        def deserialize_acus(
            acus: List[AtomicCacheUnit],
            acus_cache: Dict[str, AtomicCacheUnit],
        ) -> List[AtomicCacheUnit]:
            new_acus = []
            for acu in acus:
                acu_hash = get_object_hash(acu)
                if acu_hash not in acus_cache:
                    # Process the acu only if it hasn't been processed before
                    deserialized_acu = ACUPickler.deserialize_acu(acu)
                    acus_cache[acu_hash] = deserialized_acu
                else:
                    deserialized_acu = acus_cache[acu_hash]
                new_acus.append(deserialized_acu)
            return new_acus

        acus_cache = {}

        cache_file = cache_file + "::" if cache_file else cache_file
        for acus_object_hash, acus in tqdm(
            cache.items(), desc=f"{cache_file}AtomicCacheUnits"
        ):
            cache[acus_object_hash] = deserialize_acus(acus, acus_cache)

        acus_cache.clear()

        return cache

    @staticmethod
    def serialize_acu(unit):
        """
        Serialize the object-like properties of a unit, including object-like and global-object-like.
        """
        # Handle unit.ast_root
        # TODO: figure out why the transformed AST is non-serializable in case of TF code for some reason.
        from ivy.transpiler.utils import ast_utils

        unit.ast_root: str = (  # type: ignore
            unit.ast_root
            if isinstance(unit.ast_root, str)
            else ast_utils.ast_to_source_code(unit.ast_root)
        )

        # Handle unit.object_like
        unit.object_like: Dict = (  # type: ignore
            unit.object_like
            if isinstance(unit.object_like, dict)
            else unit.object_like.to_dict()
        )

        # Handle unit.globals
        unit.globals: List[Dict] = [  # type: ignore
            glob if isinstance(glob, dict) else glob.to_dict() for glob in unit.globals
        ]

        return unit

    @staticmethod
    def serialize_cache(cache: Dict, cache_file: str = "") -> Dict:
        from ivy.transpiler.utils.cache_utils import AtomicCacheUnit

        def serialize_acus(
            acus: List[AtomicCacheUnit],
            acus_cache: Dict[str, AtomicCacheUnit],
        ) -> List[AtomicCacheUnit]:
            new_acus = []
            for acu in acus:
                acu_hash = get_object_hash(acu)
                if acu_hash not in acus_cache:
                    # Process the acu only if it hasn't been processed before
                    serialized_acu = ACUPickler.serialize_acu(acu)
                    acus_cache[acu_hash] = serialized_acu
                else:
                    serialized_acu = acus_cache[acu_hash]
                new_acus.append(serialized_acu)
            return new_acus

        acus_cache = {}

        cache_file = cache_file + "::" if cache_file else cache_file
        for acus_object_hash, acus in tqdm(
            cache.items(), desc=f"{cache_file}AtomicCacheUnits"
        ):
            cache[acus_object_hash] = serialize_acus(acus, acus_cache)

        acus_cache.clear()

        return cache
