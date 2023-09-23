import os
from ivy.utils.backend.sub_backend_handler import find_available_sub_backends


sub_backends_loc = __file__.rpartition(os.path.sep)[0]

available_sub_backends = find_available_sub_backends(sub_backends_loc)
current_sub_backends = []
