import time
from typing import Optional


class Timer:
    r"""
    A simple timer to record time per iteration and ETA of training. ETA is
    estimated by moving window average with fixed window size.
    Args:
        start_from: Iteration from which counting should be started/resumed.
        total_iterations: Total number of iterations. ETA will not be tracked
            (will remain "N/A") if this is not provided.
        window_size: Window size to calculate ETA based on past few iterations.
    """

    def __init__(
        self,
        start_from: int = 1,
        total_iterations: Optional[int] = None,
        window_size: int = 20,
    ):
        # We decrement by 1 because `current_iter` changes increment during
        # an iteration (for example, will change from 0 -> 1 on iteration 1).
        self.current_iter = start_from - 1
        self.total_iters = total_iterations

        self._start_time = time.time()
        self._times = [0.0] * window_size

    def tic(self) -> None:
        r"""Start recording time: call at the beginning of iteration."""
        self._start_time = time.time()

    def toc(self) -> None:
        r"""Stop recording time: call at the end of iteration."""
        self._times.append(time.time() - self._start_time)
        self._times = self._times[1:]
        self.current_iter += 1

    def get_time(
        self,
    ) -> float:
        return self._times[-1]

    def __enter__(self):
        # ttysetattr etc goes here before opening and returning the file object
        self._start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        # Exception handling here
        self._times.append(time.time() - self._start_time)
        self._times = self._times[1:]
        self.current_iter += 1
