def get_bool(key):
    if key == 'True':
        return True
    elif key=='False':
        return False # .get(key)
    else:
        value = bool(key)
    return value
