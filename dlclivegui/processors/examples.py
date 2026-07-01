from __future__ import annotations

import logging
from collections import deque
from math import acos, atan2, copysign, degrees, pi, sqrt

import numpy as np

from dlclivegui.processors import BaseProcessorSocket, register_processor

logger = logging.getLogger(__name__)


class OneEuroFilter:  # pragma: no cover
    def __init__(self, t0, x0, dx0=None, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        if dx0 is None:
            dx0 = np.zeros_like(x0)
        self.dx_prev = dx0
        self.t_prev = t0

    @staticmethod
    def smoothing_factor(t_e, cutoff):
        r = 2 * pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def exponential_smoothing(alpha, x, x_prev):
        return alpha * x + (1 - alpha) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


@register_processor
class ExampleProcessorSocketCalculateMousePose(BaseProcessorSocket):  # pragma: no cover
    """
    DLC Processor with pose calculations (center, heading, head angle) and optional filtering.

    Calculates:
    - center: Weighted average of head keypoints
    - heading: Body orientation (degrees)
    - head_angle: Head rotation relative to body (radians)

    Broadcasts: [timestamp, center_x, center_y, heading, head_angle]
    """

    PROCESSOR_NAME = "Example Experiment Pose Processor"
    PROCESSOR_DESCRIPTION = "Calculates mouse center, heading, and head angle with optional One-Euro filtering"
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("127.0.0.1", 6000),
            "description": "Server address (host, port)",
        },
        "authkey": {
            "type": "bytes",
            "default": b"secret password",
            "description": "Authentication key for clients",
        },
        "use_perf_counter": {
            "type": "bool",
            "default": False,
            "description": "Use time.perf_counter() instead of time.time()",
        },
        "use_filter": {
            "type": "bool",
            "default": False,
            "description": "Apply One-Euro filter to calculated values",
        },
        "filter_kwargs": {
            "type": "dict",
            "default": {"min_cutoff": 1.0, "beta": 0.02, "d_cutoff": 1.0},
            "description": "One-Euro filter parameters (min_cutoff, beta, d_cutoff)",
        },
        "save_original": {
            "type": "bool",
            "default": False,
            "description": "Save raw pose arrays for analysis",
        },
    }

    def __init__(
        self,
        bind=("127.0.0.1", 6000),
        authkey=b"secret password",
        use_perf_counter=False,
        use_filter=False,
        filter_kwargs: dict | None = None,
        save_original=False,
    ):
        super().__init__(
            bind=bind,
            authkey=authkey,
            use_perf_counter=use_perf_counter,
            save_original=save_original,
        )

        self.center_x = deque()
        self.center_y = deque()
        self.heading_direction = deque()
        self.head_angle = deque()

        self.use_filter = use_filter
        self.filter_kwargs = filter_kwargs if filter_kwargs is not None else {}
        self.filters = None

    def _clear_data_queues(self):
        super()._clear_data_queues()
        self.center_x.clear()
        self.center_y.clear()
        self.heading_direction.clear()
        self.head_angle.clear()

    def _initialize_filters(self, vals):
        t0 = self.timing_func()
        self.filters = {
            "center_x": OneEuroFilter(t0, vals[0], **self.filter_kwargs),
            "center_y": OneEuroFilter(t0, vals[1], **self.filter_kwargs),
            "heading": OneEuroFilter(t0, vals[2], **self.filter_kwargs),
            "head_angle": OneEuroFilter(t0, vals[3], **self.filter_kwargs),
        }
        logger.debug(f"Initialized One-Euro filters with parameters: {self.filter_kwargs}")

    def process(self, pose, **kwargs):
        # Extract keypoints and confidence
        xy = pose[:, :2]
        conf = pose[:, 2]

        # Calculate weighted center from head keypoints
        head_xy = xy[[0, 1, 2, 3, 4, 5, 6, 26], :]
        head_conf = conf[[0, 1, 2, 3, 4, 5, 6, 26]]
        try:
            center = np.average(head_xy, axis=0, weights=head_conf)
        except ZeroDivisionError:
            center = np.zeros(2)

        # Calculate body axis (tail_base -> neck)
        body_axis = xy[7] - xy[13]
        body_axis /= sqrt(np.sum(body_axis**2))

        # Calculate head axis (neck -> nose)
        head_axis = xy[0] - xy[7]
        head_axis /= sqrt(np.sum(head_axis**2))

        # Calculate head angle relative to body
        cross = body_axis[0] * head_axis[1] - head_axis[0] * body_axis[1]
        sign = copysign(1, cross)  # Positive when looking left

        try:
            head_angle = acos(body_axis @ head_axis) * sign
        except ValueError:
            head_angle = 0

        # Calculate heading (body orientation)
        heading = degrees(atan2(body_axis[1], body_axis[0]))

        # Raw values (heading unwrapped for filtering)
        vals = [center[0], center[1], heading, head_angle]

        # Apply filtering if enabled
        curr_time = self.timing_func()
        if self.use_filter:
            if self.filters is None:
                self._initialize_filters(vals)

            vals = [
                self.filters["center_x"](curr_time, vals[0]),
                self.filters["center_y"](curr_time, vals[1]),
                self.filters["heading"](curr_time, vals[2]),
                self.filters["head_angle"](curr_time, vals[3]),
            ]

        # Wrap heading to [0, 360) after filtering
        vals[2] = vals[2] % 360
        # Update step counter
        self.curr_step = self.curr_step + 1

        # Store processed data (only if recording)
        if self.recording:
            if self.save_original and self.original_pose is not None:
                self.original_pose.append(pose.copy())
            self.center_x.append(vals[0])
            self.center_y.append(vals[1])
            self.heading_direction.append(vals[2])
            self.head_angle.append(vals[3])
            self.time_stamp.append(curr_time)
            self.step.append(self.curr_step)
            self.frame_time.append(kwargs.get("frame_time", -1))
            if "pose_time" in kwargs:
                self.pose_time.append(kwargs["pose_time"])

        payload = [curr_time, vals[0], vals[1], vals[2], vals[3]]
        self.broadcast(payload)
        return pose

    def get_data(self):
        save_dict = super().get_data()
        save_dict["x_pos"] = np.array(self.center_x)
        save_dict["y_pos"] = np.array(self.center_y)
        save_dict["heading_direction"] = np.array(self.heading_direction)
        save_dict["head_angle"] = np.array(self.head_angle)
        save_dict["use_filter"] = self.use_filter
        save_dict["filter_kwargs"] = self.filter_kwargs
        return save_dict


@register_processor
class ExampleProcessorSocketFilterKeypoints(BaseProcessorSocket):  # pragma: no cover
    PROCESSOR_NAME = "Mouse Pose with less keypoints"
    PROCESSOR_DESCRIPTION = "Calculates mouse center, heading, and head angle with optional One-Euro filtering"
    PROCESSOR_PARAMS = {
        "bind": {
            "type": "tuple",
            "default": ("127.0.0.1", 6000),
            "description": "Server address (host, port)",
        },
        "authkey": {
            "type": "bytes",
            "default": b"secret password",
            "description": "Authentication key for clients",
        },
        "use_perf_counter": {
            "type": "bool",
            "default": False,
            "description": "Use time.perf_counter() instead of time.time()",
        },
        "use_filter": {
            "type": "bool",
            "default": False,
            "description": "Apply One-Euro filter to calculated values",
        },
        "filter_kwargs": {
            "type": "dict",
            "default": {"min_cutoff": 1.0, "beta": 0.02, "d_cutoff": 1.0},
            "description": "One-Euro filter parameters (min_cutoff, beta, d_cutoff)",
        },
        "save_original": {
            "type": "bool",
            "default": True,
            "description": "Save raw pose arrays for analysis",
        },
    }

    def __init__(
        self,
        bind=("127.0.0.1", 6000),
        authkey=b"secret password",
        use_perf_counter=False,
        use_filter=False,
        filter_kwargs: dict | None = None,
        save_original=True,
        p_cutoff=0.4,
    ):
        super().__init__(
            bind=bind,
            authkey=authkey,
            use_perf_counter=use_perf_counter,
            save_original=save_original,
        )

        self.center_x = deque()
        self.center_y = deque()
        self.heading_direction = deque()
        self.head_angle = deque()

        self.p_cutoff = p_cutoff

        self.use_filter = use_filter
        self.filter_kwargs = filter_kwargs if filter_kwargs is not None else {}
        self.filters = None

    def _clear_data_queues(self):
        super()._clear_data_queues()
        self.center_x.clear()
        self.center_y.clear()
        self.heading_direction.clear()
        self.head_angle.clear()

    def _initialize_filters(self, vals):
        t0 = self.timing_func()
        self.filters = {
            "center_x": OneEuroFilter(t0, vals[0], **self.filter_kwargs),
            "center_y": OneEuroFilter(t0, vals[1], **self.filter_kwargs),
            "heading": OneEuroFilter(t0, vals[2], **self.filter_kwargs),
            "head_angle": OneEuroFilter(t0, vals[3], **self.filter_kwargs),
        }
        logger.debug(f"Initialized One-Euro filters with parameters: {self.filter_kwargs}")

    def process(self, pose, **kwargs):
        # Extract keypoints and confidence
        xy = pose[:, :2]
        conf = pose[:, 2]

        # Calculate weighted center from head keypoints
        head_xy = xy[[0, 1, 2, 3, 5, 6, 7], :]
        head_conf = conf[[0, 1, 2, 3, 5, 6, 7]]
        # set low confidence keypoints to zero weight
        head_conf = np.where(head_conf < self.p_cutoff, 0, head_conf)
        try:
            center = np.average(head_xy, axis=0, weights=head_conf)
        except ZeroDivisionError:
            # If all keypoints have zero weight, return without processing
            return pose

        neck = np.average(xy[[2, 3, 6, 7], :], axis=0, weights=conf[[2, 3, 6, 7]])

        # Calculate body axis (tail_base -> neck)
        body_axis = neck - xy[9]
        body_axis /= sqrt(np.sum(body_axis**2))

        # Calculate head axis (neck -> nose)
        head_axis = xy[0] - neck
        head_axis /= sqrt(np.sum(head_axis**2))

        # Calculate head angle relative to body
        cross = body_axis[0] * head_axis[1] - head_axis[0] * body_axis[1]
        sign = copysign(1, cross)  # Positive when looking left

        try:
            head_angle = acos(body_axis @ head_axis) * sign
        except ValueError:
            head_angle = 0

        # Calculate heading (body orientation)
        heading = degrees(atan2(body_axis[1], body_axis[0]))
        vals = [center[0], center[1], heading, head_angle]

        curr_time = self.timing_func()
        if self.use_filter:
            if self.filters is None:
                self._initialize_filters(vals)

            vals = [
                self.filters["center_x"](curr_time, vals[0]),
                self.filters["center_y"](curr_time, vals[1]),
                self.filters["heading"](curr_time, vals[2]),
                self.filters["head_angle"](curr_time, vals[3]),
            ]

        # Wrap heading to [0, 360) after filtering
        vals[2] = vals[2] % 360
        # Update step counter
        self.curr_step = self.curr_step + 1

        # Store processed data (only if recording)
        if self.recording:
            if self.save_original and self.original_pose is not None:
                self.original_pose.append(pose.copy())
            self.center_x.append(vals[0])
            self.center_y.append(vals[1])
            self.heading_direction.append(vals[2])
            self.head_angle.append(vals[3])
            self.time_stamp.append(curr_time)
            self.step.append(self.curr_step)
            self.frame_time.append(kwargs.get("frame_time", -1))
            if "pose_time" in kwargs:
                self.pose_time.append(kwargs["pose_time"])

        payload = [curr_time, vals[0], vals[1], vals[2], vals[3]]
        self.broadcast(payload)
        return pose

    def get_data(self):
        save_dict = super().get_data()
        save_dict["x_pos"] = np.array(self.center_x)
        save_dict["y_pos"] = np.array(self.center_y)
        save_dict["heading_direction"] = np.array(self.heading_direction)
        save_dict["head_angle"] = np.array(self.head_angle)
        save_dict["use_filter"] = self.use_filter
        save_dict["filter_kwargs"] = self.filter_kwargs
        return save_dict
