import copy
import numpy as np


class FuncWrapper:

    def __init__(self, func):
        self.func = func

    def __mul__(self, rhs):
        return FuncWrapper(lambda *args: rhs * self.func(*args))

    def __truediv__(self, rhs):
        return FuncWrapper(lambda *args: self.func(*args) / rhs)

    def __add__(self, rhs):
        return FuncWrapper(lambda *args: rhs + self.func(*args))

    def __sub__(self, rhs):
        return FuncWrapper(lambda *args: self.func(*args) - rhs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return FuncWrapper(lambda *args: ufunc(self.func(*args)))

    def __call__(self, *args):
        return self.func(*args)


def gram_schmidt(objs, inner_product_func, args=()):
    """
    Gram–Schmidt orthonormalization.

    Parameters
    ----------
    objs : iterable of callables, arrays
        This can be a list of functions or a list of vectors that you
        would like to turn into an orthonormal basis.
    inner_product_func : callable
        This callable must take at minimum two arguments (two objects
        from the ``objs`` list) and return a scalar. If the input objs
        are vectors, this function could be as simple as `numpy.dot`.
        If the input objects are functions, this would be a way of
        evaluating the inner product integral of two functions from the
        ``objs`` list. In that case, the inner product function might
        take a third argument that specifies the measure function used
        to define the inner product, and some information about how to
        compute the numerical integral.
    args : iterable
        Passed in to ``inner_product_func`` after two objects from the
        ``objs`` list.

    Returns
    -------
    basis_objs : list
        A list of orthonormal objects (functions, vectors, etc. -
        whatever type was inputted).
    """

    # Input validation:
    if len(objs) == 0:
        raise ValueError("You must pass in at least one obj")

    obj_types = set([type(obj) for obj in objs])
    if len(obj_types) != 1:
        raise TypeError(
            f"All objs must have the same type (received: {obj_types})")

    wrap_objs = []
    for obj in objs:
        can_math = (
            hasattr(obj, "__truediv__") and
            hasattr(obj, "__sub__") and
            hasattr(obj, "__array_ufunc__")  # for np.sqrt
        )
        if can_math:  # can handle division and subtraction: good!
            wrap_objs.append(obj)
        elif hasattr(obj, "__call__"):  # callable
            wrap_objs.append(FuncWrapper(obj))
        else:  # not callable, can't math: bad!
            raise ValueError("TODO")

    basis_objs = []
    for obj in wrap_objs:
        this_bob = copy.copy(obj)
        for bobj in basis_objs:
            this_bob -= inner_product_func(this_bob, bobj, *args) * bobj
        this_bob /= np.sqrt(inner_product_func(this_bob, this_bob, *args))
        basis_objs.append(this_bob)
    return basis_objs
