#Project help functions

def is_all_val(*args, val=None):
    return args.count(val) == len(args)

def is_any_val(*args, val=None):
    return val in args

def num_not_val(*args, val=None):
    return len(args) - args.count(val)

def is_all_None(*args):
    return is_all_val(*args, val=None)

def is_any_None(*args):
    return is_any_val(*args, val=None)

def num_not_None(*args):
    return num_not_val(*args, val=None)
