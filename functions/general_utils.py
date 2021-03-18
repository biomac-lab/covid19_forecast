def get_bool(key):
    if key == 'True':
        value = True
    elif key=='False':
        value = False # .get(key)
    else:
        value = bool(key)
    return value
