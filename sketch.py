import cvxpy as cvx

v = cvx.Variable(50000, nonneg=True)
p = cvx.Problem(cvx.Minimize(cvx.sum(v)))
p.solve()

import re as regex

pat = regex.compile('([0-9]+)')


import logging
import sys

console_hand = logging.StreamHandler(sys.stdout)
console_hand.setLevel(logging.INFO)


a = {}


class Format(logging.Formatter):
    def formatMessage(self, record):
        import inspect
        import os
        f = inspect.currentframe()

        if f is not None:
            f = f.f_back
        qual_name = record.funcName
        record_filename = os.path.normcase(record.pathname)

        a['a'] = f

        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename != record_filename or f.f_lineno!=record.lineno or co.co_name!=record.funcName:
                a['c'] = f
                f = f.f_back
                continue
            else:
                f=f.f_back
                func = f.f_locals.get(record.funcName, None) or getattr(f.f_locals.get(f.f_code.co_names[-1]), record.funcName, None)
                if func:
                    qual_name = func.__qualname__
                break



        record.qual_name=qual_name
        return super(Format, self).formatMessage(record)

console_hand.setFormatter(Format(fmt='%(pathname)s:%(lineno)d:\n'
                                                '%(levelname)s:%(qual_name)s:%(message)s'))


logger = logging.getLogger(__name__)
logger.propagate = False
if logger.hasHandlers(): logger.handlers = []
logger.addHandler(console_hand)
logger.setLevel(logging.INFO)

class A:
    class B:
        def __init__(self):
            logger.info('hello')

    def __init__(self):
        self.b = self.B()

def hello():
    def hello2():
        logger.info('hello')
    return hello2()