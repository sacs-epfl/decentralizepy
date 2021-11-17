def conditional_value(var, nul, default):
    if var != nul:
        return var
    else:
        return default


def remove_keys(d, keys_to_remove):
    return {key: d[key] for key in d if key not in keys_to_remove}
