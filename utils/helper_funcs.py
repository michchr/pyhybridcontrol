#Project help functions


def is_all_None(tup):
    return tup.count(None) == len(tup)

def is_any_None(tup):
    return None in tup

def num_not_None(tup):
    return len(tup)-tup.count(None)

def is_any_not_None(tup):
    return any(item != None for item in tup)