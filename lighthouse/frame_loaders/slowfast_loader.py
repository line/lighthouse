class SlowFastLoader:
    def __init__(
        self,
        framerate: float,
        size: int,
        device: str,
        centercrop: bool = True) -> None:
        self._framerate = framerate
        self._size = size
        self._device = device
        self._centercrop = centercrop