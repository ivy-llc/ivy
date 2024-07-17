def tensorflow__check_in_nested_sequence(sequence, value=None, _type=None):
    if sequence is value or isinstance(sequence, _type):
        return True
    elif isinstance(sequence, (tuple, list)):
        if any(isinstance(_val, _type) or _val is value for _val in sequence):
            return True
        else:
            return any(
                tensorflow__check_in_nested_sequence(sub_sequence, value, _type)
                for sub_sequence in sequence
                if isinstance(sub_sequence, (tuple, list))
            )
