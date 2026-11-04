"""Virtual-line counting logic.

A LineCounter watches the centre point (centroid) of each tracked vehicle.
When a vehicle moves from one side of the line to the other, it is counted once.

Directions are named by the side of the line the vehicle came from:
  A_to_B : moved from side A to side B
  B_to_A : moved from side B to side A

For a line drawn left to right (the default), side A is ABOVE the line in the
image and side B is BELOW it, so A_to_B means "moving down the image".
Run src/preview_line.py to see the A and B sides drawn on a real frame.
"""


class LineCounter:
    def __init__(self, p1, p2):
        """p1, p2: the two end points of the counting line, in pixels (x, y)."""
        self.p1 = p1
        self.p2 = p2
        self.last_side = {}   # track_id -> last side seen (-1 = A, +1 = B)
        self.counted = set()  # track_ids already counted (each vehicle counts once)

    def _side_and_position(self, cx, cy):
        (x1, y1), (x2, y2) = self.p1, self.p2
        dx, dy = x2 - x1, y2 - y1
        cross = dx * (cy - y1) - dy * (cx - x1)          # sign tells which side
        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)  # 0..1 = along the segment
        side = 1 if cross > 0 else (-1 if cross < 0 else 0)
        return side, t

    def update(self, track_id, cx, cy):
        """Feed one tracked centroid. Returns 'A_to_B' / 'B_to_A' if the vehicle
        has just crossed the line (and was not counted before), otherwise None."""
        side, t = self._side_and_position(cx, cy)
        previous = self.last_side.get(track_id)
        if side != 0:
            self.last_side[track_id] = side

        if previous is None or side == 0 or side == previous:
            return None
        if track_id in self.counted:
            return None                      # jitter back and forth: do not count twice
        if not (0.0 <= t <= 1.0):
            return None                      # crossed the extended line, beyond the segment

        self.counted.add(track_id)
        return "A_to_B" if previous < 0 else "B_to_A"
