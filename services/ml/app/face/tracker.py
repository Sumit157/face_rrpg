"""Frame-to-frame face tracking quality bookkeeping.

MediaPipe FaceMesh already tracks internally between frames, but the
pipeline still needs to know *how well* tracking held up over a video
(dropped frames, face loss) because that feeds directly into the
signal-quality and warnings the application must surface to the user.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrackingReport:
    total_frames: int = 0
    frames_with_face: int = 0
    longest_gap_frames: int = 0
    gap_events: int = 0
    _current_gap: int = field(default=0, repr=False)

    def update(self, face_found: bool) -> None:
        self.total_frames += 1
        if face_found:
            self.frames_with_face += 1
            if self._current_gap > 0:
                self.gap_events += 1
            self._current_gap = 0
        else:
            self._current_gap += 1
            self.longest_gap_frames = max(self.longest_gap_frames, self._current_gap)

    @property
    def detection_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.frames_with_face / self.total_frames

    def to_dict(self) -> dict:
        return {
            "totalFrames": self.total_frames,
            "framesWithFace": self.frames_with_face,
            "detectionRate": round(self.detection_rate, 4),
            "longestGapFrames": self.longest_gap_frames,
            "gapEvents": self.gap_events,
        }
