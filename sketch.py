import os
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'True'
import wrapt

def A(A):
    b = A
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        print (b)
        return wrapped(*args ,**kwargs)
    return wrapper

@A(1)
def test(a,b,c=1,*args, k=1, **kwargs):
    return 1

@A(3)
def test2(a,b,c=1,*args, k=1, **kwargs):
    return 1