# import ivy


class IvyException(Exception):
    def __init__(self, message):
        # TODO: add backend str
        super().__init__(message)
