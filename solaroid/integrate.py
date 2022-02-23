import numpy as np
from scipy.integrate._quadrature import tupleset
from scipy.special import logsumexp


def _basic_log_simpson(log_y, start, stop, x, dx, axis):
    nd = len(log_y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even-spaced Simpson's rule.
        # result = np.sum(y[slice0] + 4*y[slice1] + y[slice2], axis=axis)
        vals = []
        if log_y[slice0].size:
            vals.append(log_y[slice0])
        if log_y[slice1].size:
            vals.append(np.log(4) + log_y[slice1])
        if log_y[slice2].size:
            vals.append(log_y[slice2])
        result = np.array(-np.inf)
        if len(vals):
            result = logsumexp(vals, axis=0)

        if result.size:
            result = logsumexp(result, axis=axis)

        # result *= dx / 3.0
        result += np.log(dx) - np.log(3)
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0]
        h1 = h[sl1]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = h0 / h1

        # tmp = hsum/6.0 * (y[slice0] * (2 - 1.0/h0divh1) +
        #                   y[slice1] * (hsum * hsum / hprod) +
        #                   y[slice2] * (2 - h0divh1))
        tmp = (np.log(hsum) - np.log(6)) + logsumexp([
            log_y[slice0] + np.log(2 - 1.0/h0divh1),
            log_y[slice1] + np.log(hsum * hsum / hprod),
            log_y[slice2] + np.log(2 - h0divh1)
        ], axis=0)
        result = logsumexp(tmp, axis=axis)
    return result


def log_simpson(log_y, x=None, dx=1.0, axis=-1, even='avg'):
    """
    An implementation of Simpson's rule that takes log-integrand values and
    returns the (natural) log of the integral.
    """
    log_y = np.asarray(log_y)
    nd = len(log_y.shape)
    N = log_y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(log_y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as log_y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as log_y.")
    if N % 2 == 0:
        val = -np.inf
        result = -np.inf
        slice1 = (slice(None),)*nd
        slice2 = (slice(None),)*nd
        if even not in ['avg', 'last', 'first']:
            raise ValueError("Parameter 'even' must be "
                             "'avg', 'last', or 'first'.")
        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            slice1 = tupleset(slice1, axis, -1)
            slice2 = tupleset(slice2, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val = np.logaddexp(
                val,
                np.log(0.5*last_dx)
                + np.logaddexp(log_y[slice1], log_y[slice2])
            )
            result = _basic_log_simpson(log_y, 0, N-3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            slice1 = tupleset(slice1, axis, 0)
            slice2 = tupleset(slice2, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val = np.logaddexp(
                val,
                np.log(0.5*first_dx)
                + np.logaddexp(log_y[slice2], log_y[slice1])
            )
            result = np.logaddexp(
                result,
                _basic_log_simpson(log_y, 1, N-2, x, dx, axis)
            )
        if even == 'avg':
            val -= np.log(2.0)
            result -= np.log(2.0)
        result = np.logaddexp(result, val)
    else:
        result = _basic_log_simpson(log_y, 0, N-2, x, dx, axis)

    if returnshape:
        x = x.reshape(saveshape)

    return result
