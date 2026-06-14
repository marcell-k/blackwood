import sys


class _Stop:
    def __call__(self, code=0):
        sys.exit(code)


stop = _Stop()
