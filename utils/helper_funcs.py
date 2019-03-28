#Project help functions


def eq_all_val(*args, val=None):
    return args.count(val) == len(args)

def eq_any_val(*args, val=None):
    return val in args

def num_neq_val(*args, val=None):
    return len(args) - args.count(val)

def is_all_val(*args, val=None):
    return all([arg is val for arg in args])

def is_any_val(*args, val=None):
    return any([arg is val for arg in args])

def num_not_val(*args, val=None):
    return [arg is not val for arg in args].count(True)





def is_all_None(*args):
    return is_all_val(*args, val=None)

def is_any_None(*args):
    return is_any_val(*args, val=None)

def num_not_None(*args):
    return num_not_val(*args, val=None)

def eq_all_None(*args):
    return eq_all_val(*args, val=None)

def eq_any_None(*args):
    return eq_any_val(*args, val=None)

def num_neq_None(*args):
    return num_neq_val(*args, val=None)
