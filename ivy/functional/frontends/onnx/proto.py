class NodeProto:
    def __init__(self):
        self._fn = None
        self._fn_mod = None
        self._fn_name = None
        self.input = None
        self.output = None
        self.name = None

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)
