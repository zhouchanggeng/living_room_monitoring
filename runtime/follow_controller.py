from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]
FrameSize = Tuple[int, int]


class FollowMode(str, Enum):
    ADULT = "adult"
    CHILD = "child"
    AUTO = "auto"


class FollowState(str, Enum):
    IDLE = "idle"
    ACQUIRE = "acquire"
    TRACK = "track"
    HOLD = "hold"
    SEARCH = "search"


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    class_name: str
    bbox_xyxy: BBox
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class FollowConfig:
    follow_mode: FollowMode = FollowMode.AUTO
    fps: float = 25.0
    hold_seconds: float = 0.8
    lost_seconds: float = 1.2
    acquire_frames: int = 1
    low_pass_alpha: float = 0.35
    dead_zone_x_ratio: float = 0.03
    dead_zone_y_ratio: float = 0.03
    kp_x: float = 0.006
    kd_x: float = 0.002
    kp_y: float = 0.006
    kd_y: float = 0.002
    max_pan_speed: float = 1.0
    max_tilt_speed: float = 0.8
    search_pan_speed: float = 0.25
    current_track_bonus: float = 4.0
    class_priority_weight: float = 2.0
    confidence_weight: float = 1.0
    area_weight: float = 1.0
    center_weight: float = 1.0


@dataclass(frozen=True)
class FollowCommand:
    state: FollowState
    pan_speed: float
    tilt_speed: float
    direction_x: str = "stop"
    direction_y: str = "stop"
    normalized_error_xy: Tuple[float, float] = (0.0, 0.0)
    sim_delta_xy: Tuple[float, float] = (0.0, 0.0)
    target_track_id: Optional[int] = None
    target_class: Optional[str] = None
    target_bbox_xyxy: Optional[BBox] = None
    error_xy: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def stop(cls, state: FollowState) -> "FollowCommand":
        return cls(state=state, pan_speed=0.0, tilt_speed=0.0)


class FollowController:
    def __init__(self, config: FollowConfig):
        self.config = config
        self.state = FollowState.IDLE
        self.locked_track_id: Optional[int] = None
        self.locked_class: Optional[str] = None
        self.last_bbox: Optional[BBox] = None
        self.smoothed_center: Optional[Tuple[float, float]] = None
        self.last_error: Tuple[float, float] = (0.0, 0.0)
        self.missing_frames = 0
        self.acquire_count = 0
        self.search_direction = 1.0

    def reset(self) -> None:
        self.state = FollowState.IDLE
        self._clear_lock()

    def _clear_lock(self) -> None:
        self.locked_track_id = None
        self.locked_class = None
        self.last_bbox = None
        self.smoothed_center = None
        self.last_error = (0.0, 0.0)
        self.missing_frames = 0
        self.acquire_count = 0

    def update(self, tracks: Iterable[TrackedObject], frame_size: FrameSize) -> FollowCommand:
        candidates = self._filter_by_mode(list(tracks))
        locked = self._find_locked(candidates)
        target = locked or self._select_target(candidates, frame_size)

        if target is not None:
            return self._track_target(target, frame_size)

        if self.locked_track_id is not None:
            if self.missing_frames < self._lost_frames:
                return self._hold_or_search(frame_size)
            self._clear_lock()
            return self._search_command()

        if self.state == FollowState.SEARCH:
            return self._search_command()
        return FollowCommand.stop(FollowState.IDLE)

    @property
    def _hold_frames(self) -> int:
        return max(1, int(self.config.hold_seconds * self.config.fps))

    @property
    def _lost_frames(self) -> int:
        return max(self._hold_frames + 1, int(self.config.lost_seconds * self.config.fps))

    def _filter_by_mode(self, tracks: Sequence[TrackedObject]) -> list[TrackedObject]:
        if self.config.follow_mode == FollowMode.AUTO:
            return list(tracks)

        wanted = "Adult" if self.config.follow_mode == FollowMode.ADULT else "Kid"
        return [track for track in tracks if track.class_name.lower() == wanted.lower()]

    def _find_locked(self, tracks: Sequence[TrackedObject]) -> Optional[TrackedObject]:
        if self.locked_track_id is None:
            return None
        for track in tracks:
            if track.track_id == self.locked_track_id:
                return track
        return None

    def _select_target(
        self, tracks: Sequence[TrackedObject], frame_size: FrameSize
    ) -> Optional[TrackedObject]:
        if not tracks:
            return None
        return max(tracks, key=lambda track: self._score_track(track, frame_size))

    def _score_track(self, track: TrackedObject, frame_size: FrameSize) -> float:
        width, height = frame_size
        cx, cy = track.center
        area_norm = track.area / max(float(width * height), 1.0)
        dx = abs(cx - width * 0.5) / max(width * 0.5, 1.0)
        dy = abs(cy - height * 0.5) / max(height * 0.5, 1.0)
        center_score = 1.0 - min(1.0, (dx + dy) * 0.5)
        class_score = self._class_priority(track.class_name)
        lock_score = (
            self.config.current_track_bonus
            if track.track_id == self.locked_track_id
            else 0.0
        )
        return (
            self.config.class_priority_weight * class_score
            + self.config.confidence_weight * track.confidence
            + self.config.area_weight * area_norm
            + self.config.center_weight * center_score
            + lock_score
        )

    def _class_priority(self, class_name: str) -> float:
        if self.config.follow_mode == FollowMode.CHILD:
            return 1.0 if class_name.lower() == "kid" else 0.0
        if self.config.follow_mode == FollowMode.ADULT:
            return 1.0 if class_name.lower() == "adult" else 0.0
        return 1.0

    def _track_target(self, target: TrackedObject, frame_size: FrameSize) -> FollowCommand:
        was_locked = target.track_id == self.locked_track_id
        if not was_locked:
            self.acquire_count += 1
            self.state = FollowState.ACQUIRE
        else:
            self.acquire_count = max(self.acquire_count, self.config.acquire_frames)

        if self.acquire_count >= self.config.acquire_frames:
            self.locked_track_id = target.track_id
            self.locked_class = target.class_name
            self.state = FollowState.TRACK

        self.missing_frames = 0
        self.last_bbox = target.bbox_xyxy
        smoothed_center = self._smooth_center(target.center)
        pan_speed, tilt_speed, error = self._control_from_center(smoothed_center, frame_size)
        direction_x, direction_y = _directions_from_error(error)
        normalized_error = _normalized_error(error, frame_size)

        return FollowCommand(
            state=self.state,
            pan_speed=pan_speed,
            tilt_speed=tilt_speed,
            direction_x=direction_x,
            direction_y=direction_y,
            normalized_error_xy=normalized_error,
            sim_delta_xy=(pan_speed, tilt_speed),
            target_track_id=target.track_id,
            target_class=target.class_name,
            target_bbox_xyxy=target.bbox_xyxy,
            error_xy=error,
        )

    def _hold_or_search(self, frame_size: FrameSize) -> FollowCommand:
        self.missing_frames += 1
        if self.missing_frames <= self._hold_frames and self.smoothed_center is not None:
            pan_speed, tilt_speed, error = self._control_from_center(
                self.smoothed_center, frame_size
            )
            direction_x, direction_y = _directions_from_error(error)
            normalized_error = _normalized_error(error, frame_size)
            return FollowCommand(
                state=FollowState.HOLD,
                pan_speed=pan_speed,
                tilt_speed=tilt_speed,
                direction_x=direction_x,
                direction_y=direction_y,
                normalized_error_xy=normalized_error,
                sim_delta_xy=(pan_speed, tilt_speed),
                target_track_id=self.locked_track_id,
                target_class=self.locked_class,
                target_bbox_xyxy=self.last_bbox,
                error_xy=error,
            )

        if self.missing_frames < self._lost_frames:
            return FollowCommand(
                state=FollowState.HOLD,
                pan_speed=0.0,
                tilt_speed=0.0,
                target_track_id=self.locked_track_id,
                target_class=self.locked_class,
                target_bbox_xyxy=self.last_bbox,
            )

        self._clear_lock()
        return self._search_command()

    def _smooth_center(self, center: Tuple[float, float]) -> Tuple[float, float]:
        alpha = _clamp(self.config.low_pass_alpha, 0.0, 1.0)
        if self.smoothed_center is None:
            self.smoothed_center = center
        else:
            last_x, last_y = self.smoothed_center
            cx, cy = center
            self.smoothed_center = (
                alpha * cx + (1.0 - alpha) * last_x,
                alpha * cy + (1.0 - alpha) * last_y,
            )
        return self.smoothed_center

    def _control_from_center(
        self, center: Tuple[float, float], frame_size: FrameSize
    ) -> Tuple[float, float, Tuple[float, float]]:
        width, height = frame_size
        ex = center[0] - width * 0.5
        ey = center[1] - height * 0.5

        if abs(ex) < width * self.config.dead_zone_x_ratio:
            ex = 0.0
        if abs(ey) < height * self.config.dead_zone_y_ratio:
            ey = 0.0

        dex = ex - self.last_error[0]
        dey = ey - self.last_error[1]
        self.last_error = (ex, ey)

        pan_speed = self.config.kp_x * ex + self.config.kd_x * dex
        tilt_speed = self.config.kp_y * ey + self.config.kd_y * dey
        return (
            _clamp(pan_speed, -self.config.max_pan_speed, self.config.max_pan_speed),
            _clamp(tilt_speed, -self.config.max_tilt_speed, self.config.max_tilt_speed),
            (ex, ey),
        )

    def _search_command(self) -> FollowCommand:
        self.state = FollowState.SEARCH
        pan_speed = self.config.search_pan_speed * self.search_direction
        return FollowCommand(
            state=FollowState.SEARCH,
            pan_speed=pan_speed,
            tilt_speed=0.0,
            direction_x="right" if pan_speed > 0.0 else "left",
            direction_y="stop",
            sim_delta_xy=(pan_speed, 0.0),
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _directions_from_error(error: Tuple[float, float]) -> Tuple[str, str]:
    ex, ey = error
    if ex > 0.0:
        direction_x = "right"
    elif ex < 0.0:
        direction_x = "left"
    else:
        direction_x = "stop"

    if ey > 0.0:
        direction_y = "down"
    elif ey < 0.0:
        direction_y = "up"
    else:
        direction_y = "stop"
    return direction_x, direction_y


def _normalized_error(error: Tuple[float, float], frame_size: FrameSize) -> Tuple[float, float]:
    width, height = frame_size
    return (
        error[0] / max(width * 0.5, 1.0),
        error[1] / max(height * 0.5, 1.0),
    )
