def label_to_int(string_label):
    if string_label == 'corchea': return 1
    elif string_label == 'fusa': return 2
    elif string_label == 'negra':
        return 3

    else:
        raise Exception('unkown class_label')


def int_to_label(string_label):
    if string_label == 1: return 'corchea'
    elif string_label == 2: return 'fusa'
    elif string_label == 3:
        return 'negra'
    else:
        raise Exception('unkown class_label')