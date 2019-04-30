import functools

def memoized_as_tuple(f):
    cache = {}
    @functools.wraps(f)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        res = tuple(f(*args))
        cache[args] = res
        return res
    return wrapper
