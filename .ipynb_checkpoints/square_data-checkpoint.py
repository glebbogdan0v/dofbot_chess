from dataclasses import dataclass

@dataclass
class SquareData:
    def __init__(self, center: tuple, top_side_len: int):
        self.center = center
        self.top_side_len = top_side_len