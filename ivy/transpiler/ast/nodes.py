class FromImportObj:
    """
    Represents an object imported from a module.

    Attributes:
        module (str): The name of the module from which the object is imported.
        obj (str): The name of the imported object.
        asname (str, optional): The alias used for the import, if any.
        canonical_name (str): The full canonical name of the imported object.
    """

    def __init__(self, *, module, obj, asname=None):
        self.module = module
        self.obj = obj
        self.asname = asname
        self.canonical_name = f"{self.module}.{self.obj}"

    def __repr__(self):
        return self.canonical_name


class ImportObj:
    """
    Represents an imported module.

    Attributes:
        module (str): The name of the imported module.
        asname (str, optional): The alias used for the import, if any.
        canonical_name (str): The full canonical name of the import.
    """

    def __init__(self, *, module, asname=None):
        self.module = module
        self.asname = asname
        self.canonical_name = f"{self.module}"

    def __repr__(self):
        return self.canonical_name


class InternalObj:
    """
    Represents an object defined within the root module.

    Attributes:
        name (str): The name of the object.
        canonical_name (str): The full canonical name of the object.
        type (str): The type of the object ('function' or 'class').
    """

    def __init__(self, *, name, module, canonical_name, type):
        self.name = name
        self.module = module
        self.canonical_name = canonical_name
        self.type = type

    def __repr__(self):
        return f"{self.type}: {self.canonical_name}"
