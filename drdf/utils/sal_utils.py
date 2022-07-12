import pdb

import numpy as np


def zero_crossings(
    y_axis,
    x_axis=None,
    window=49,
    direction=False,
    alignment="pos2neg",
    smooth_outputs=True,
    return_sign=False,
):
    """
    Algorithm to find zero crossings. Smoothens the curve and finds the
    zero-crossings by looking for a sign change.


    keyword arguments:
    y_axis -- A list containg the signal over which to find zero-crossings
    x_axis -- A x-axis whose values correspond to the 'y_axis' list and is used
        in the return to specify the postion of the zero-crossings. If omitted
        then the indice of the y_axis is used. (default: None)
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 49)

    return -- the x_axis value or the indice for each zero-crossing
    """
    # smooth the curve
    length = len(y_axis)
    if x_axis is None:
        x_axis = range(length)

    x_axis = np.asarray(x_axis)
    if smooth_outputs:
        y_axis = smooth(y_axis, window)[:length]

    if direction:
        zero_crossings = np.diff(np.sign(y_axis))
        diff_arr = np.diff(np.sign(y_axis))
        if alignment == "neg2pos":
            zero_crossings = np.where(
                np.logical_and(diff_arr > 0.5, np.logical_not(np.isnan(diff_arr)))
            )[
                0
            ]  ## only return zero corssings when distance function goes from positive to negative.
        elif alignment == "pos2neg":
            pdb.set_trace()
            zero_crossings = np.logical_and(
                diff_arr < -0.5, np.logical_not(np.isnan(diff_arr))
            )
            zero_crossings = np.where(zero_crossings)[
                0
            ]  ## only return zero crossing when distance function goes from negative to positive.
    else:
        diff_arr = np.diff(np.sign(y_axis))
        zero_crossings = np.where(
            np.logical_and(np.abs(diff_arr) > 0, np.logical_not(np.isnan(diff_arr)))
        )[0]

    # if len(zero_crossings) > 2:

    times = [x_axis[indice] for indice in zero_crossings]

    # check if zero-crossings are valid
    # diff = np.diff(times)
    # if diff.std() / diff.mean() > 0.1:
    #     raise ValueError(
    #         "smoothing window too small, false zero-crossings found"
    #     )
    if return_sign:
        return times, zero_crossings, diff_arr[times]
    return times, zero_crossings


def smooth(x, window_len=11, window="hanning"):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    # print(len(s))
    # s[np.isnan(s)] = 0
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def test_zero_crossing():
    i = 10000
    x = np.linspace(0, 3.7 * np.pi, i)
    y = 0.3 * np.sin(x)
    # y *= -1
    times, times_ind = zero_crossings(y, x)


if __name__ == "__main__":
    test_zero_crossing()
