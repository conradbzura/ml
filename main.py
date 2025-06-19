from math import log

from reservoirpy.nodes import Reservoir, Ridge
import numpy as np


OPEN = 0
HIGH = 1
LOW = 2
CLOSE = 3
VOLUME = 4


def prepare_dataset():
    data = np.loadtxt(
        "spy-ohlc.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5)
    )[-(365 * 4) - 5 :]

    close_to_open = (data[1:, OPEN] - data[:-1, CLOSE]) / data[:-1, CLOSE]
    close_to_open = (close_to_open - close_to_open.mean()) / close_to_open.std()

    open_to_high = abs(((data[:, HIGH] - data[:, OPEN]) / data[:, OPEN])[1:])

    open_to_low = abs(((data[:, LOW] - data[:, OPEN]) / data[:, OPEN])[1:])

    close = ((data[:, CLOSE] - data[:, OPEN]) / data[:, OPEN])[1:]
    mask = close > 0
    close[mask] = close[mask] / open_to_high[mask]
    mask = close < 0
    close[mask] = close[mask] / open_to_low[mask]

    volume = (data[1:, VOLUME] - data[1:, VOLUME].mean()) / data[1:, VOLUME].std()

    return np.concatenate(
        (
            close_to_open.reshape(-1, 1),
            open_to_high.reshape(-1, 1),
            open_to_low.reshape(-1, 1),
            close.reshape(-1, 1),
            volume.reshape(-1, 1),
        ),
        axis=1,
    )


def finalize_datum(x):
    x[x[:, HIGH] < 0, HIGH] = 0
    x[x[:, LOW] < 0, LOW] = 0
    x[x[:, CLOSE] < -1, CLOSE] = -1
    x[x[:, CLOSE] > 1, CLOSE] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def samples(x, n):
    for i in range(x.shape[0] - n - 1):
        yield x[i : i + n], x[i + 1 : i + n + 1], x[i + n + 1].reshape(1, -1)


def fit(x, y, *, reservoir: Reservoir, ridge: Ridge, warmup: int):
    return (reservoir >> ridge).fit(x, y, warmup=warmup, reset=True)


def predict(model, x):
    return model(x)


def search(data, window):
    kernel = np.ones(window + 1) / (window + 1)
    scores = np.convolve(data, kernel, mode="valid")
    index = np.array(np.where(scores == scores.min())).ravel()
    # slices = tuple(slice(i, i + w) for i, w in zip(index, window))
    return index, window


def interpolate(space, index, window, functions, shape=None):
    """
    This function "zooms in" on the specified hyperparameter space and returns the 
    interpolated subspace.

    space:
        The hyperparameter space. It is a sequence of numpy arrays
        with each array corresponding to a hyperparameter set. This function takes a subset
        of the hyperparameter space and interpolates the values using the function
        associated with the hyperparameter (specified as `functions`) such that the resulting
        subspace has the same shape as the original.

    index:
        The lower bound of the hyperparameter subspace relative to the existing space.
        It is a sequence of integers, each corresponding to an index of its associated
        hyperparameter set in the `space` sequence.

    window:
        The shape of the hyperparameter subspace to consider. Used to determine the upper
        bound of the subspace relative to the existing space. It is a sequence of
        integers, each corresponding to an offset against its associated index in the
        `index` sequence.

    shape:
        The shape of the subspace to return. Defaults to (window * 2) + 1.

    NOTE: Each input sequence should have the same length.
    """
    if shape is None:
        shape = [(w * 2) + 1 for w in window]
    subspace = []
    for d, i, w, f, s in zip(space, index, window, functions, shape):
        subspace.append(
            f(d[i], d[i + w], s)
            if isinstance(f, type(np.linspace))
            else f(log(d[i]), log(d[i + w]), s, base=10)
        )
    return subspace
