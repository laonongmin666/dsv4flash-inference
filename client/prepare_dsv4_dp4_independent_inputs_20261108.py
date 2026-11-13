"""Route frozen independent Completion rows without regenerating any token IDs."""
import re


def lane_rank(lane):
    if not isinstance(lane, str) or not re.fullmatch(r"lane-\d+", lane):
        raise ValueError("Expected a frozen lane-NNNNNN identity")
    return int(lane.removeprefix("lane-")) % 4
