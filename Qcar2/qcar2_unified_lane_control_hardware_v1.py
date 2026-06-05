#!/usr/bin/env python3
"""
qcar2_unified_lane_control_hardware_v1.py

Hardware-safe QCar2 lane-following research runner.

Project folder target:
    ~/Documents/Quanser/examples/sdcs/qcar2/hardware/applications/GHYN/lane__following_research

Purpose
-------
This file is the first hardware-safe bridge between:

1) the working QCar2 manual-drive SSH workflow, and
2) the V6 virtual/unified controller architecture.

It keeps the useful V6 controller math:
    - PID
    - SMC
    - lightweight image-state NMPC
    - Adaptive-HyperLIF-inspired residual
    - gated residual wrapper

but removes the Windows/virtual assumptions and adds hardware protections:
    - no Windows Quanser paths
    - no virtual QCar prompt
    - perception-only mode
    - --armed required for motion
    - low-speed defaults
    - steering sign correction
    - watchdog
    - CSV logging into ./logs
    - safe Ctrl+C stop

Critical hardware sign convention
---------------------------------
Your manual-drive test showed that the physical left joystick steering was inverted:
    joystick left  -> QCar turns right
    joystick right -> QCar turns left

Therefore this hardware runner exposes:

    --steering-sign -1.0

The controllers compute steering in controller coordinates. The final hardware command is:

    hardware_steering = steering_sign * controller_steering

If the autonomous controller corrects in the wrong direction, switch:
    --steering-sign 1.0

First recommended tests
-----------------------

1) Perception only, no motion:
    python3 qcar2_unified_lane_control_hardware_v1.py \
        --mode perception \
        --duration 30 \
        --no-display

2) PID low-speed lane following:
    python3 qcar2_unified_lane_control_hardware_v1.py \
        --mode lane \
        --controller pid \
        --armed \
        --duration 20 \
        --no-display \
        --max-throttle 0.040 \
        --max-steering 0.30 \
        --steering-sign -1.0

3) SMC low-speed lane following:
    python3 qcar2_unified_lane_control_hardware_v1.py \
        --mode lane \
        --controller smc \
        --armed \
        --duration 20 \
        --no-display \
        --max-throttle 0.035 \
        --max-steering 0.30 \
        --steering-sign -1.0

4) NMPC low-speed lane following:
    python3 qcar2_unified_lane_control_hardware_v1.py \
        --mode lane \
        --controller nmpc_image \
        --armed \
        --duration 20 \
        --no-display \
        --max-throttle 0.030 \
        --max-steering 0.28 \
        --steering-sign -1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


# =============================================================================
# Local folders
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / "logs"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class LaneState:
    """Common state passed to every controller."""

    lane_error: float
    d_lane_error: float
    slope_xy: float
    previous_steering: float
    loop_dt: float
    lane_valid: bool = True
    yellow_pixel_fraction: float = 0.0
    lane_valid_reason: str = "ok"

    def vector(self) -> np.ndarray:
        return np.array(
            [
                float(self.lane_error),
                float(self.d_lane_error),
                float(self.slope_xy),
                float(self.previous_steering),
            ],
            dtype=float,
        )


@dataclass
class ControlOutput:
    raw_steering: float
    residual_steering: float = 0.0
    info: Optional[dict[str, Any]] = None


# =============================================================================
# Safety helpers
# =============================================================================

def default_leds(throttle: float, steering: float) -> np.ndarray:
    leds = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.uint8)

    if steering > 0.15:
        leds[0] = 1
        leds[2] = 1
    elif steering < -0.15:
        leds[1] = 1
        leds[3] = 1

    if throttle < 0:
        leds[5] = 1

    return leds


def safe_stop_car(car: Any) -> None:
    """Send several zero commands before termination."""
    if car is None:
        return

    try:
        for _ in range(4):
            car.read_write_std(
                throttle=0.0,
                steering=0.0,
                LEDs=default_leds(0.0, 0.0),
            )
            time.sleep(0.05)
    except Exception:
        pass


def safe_terminate(obj: Any) -> None:
    if obj is None:
        return
    try:
        obj.terminate()
    except Exception:
        pass


# =============================================================================
# Argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hardware-safe QCar2 lane-following runner for GHYN lane__following_research."
    )

    # Modes
    parser.add_argument("--mode", choices=["perception", "lane"], default="perception")
    parser.add_argument("--controller", choices=[
        "pid",
        "smc",
        "nmpc_image",
        "adaptive_hyperlif_gate",
        "pid_gated_adaptive_hyperlif_residual",
        "smc_gated_adaptive_hyperlif_residual",
        "nmpc_gated_adaptive_hyperlif_residual",
    ], default="pid")

    # Hardware safety
    parser.add_argument("--armed", action="store_true", help="Required for nonzero throttle in lane mode.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--watchdog-timeout", type=float, default=0.35)

    # Critical hardware sign correction
    parser.add_argument(
        "--steering-sign",
        type=float,
        default=-1.0,
        help="Final hardware steering sign. Use -1.0 if steering is inverted on this QCar.",
    )

    # Motion limits
    parser.add_argument("--max-steering", type=float, default=0.30)
    parser.add_argument("--base-throttle", type=float, default=0.020)
    parser.add_argument("--min-throttle", type=float, default=0.012)
    parser.add_argument("--max-throttle", type=float, default=0.040)
    parser.add_argument("--lost-lane-throttle", type=float, default=0.000)

    # Command filtering
    parser.add_argument("--steering-alpha", type=float, default=0.35)
    parser.add_argument("--steering-rate-limit", type=float, default=0.045)

    # PID
    parser.add_argument("--pid-kp", type=float, default=1.20)
    parser.add_argument("--pid-ki", type=float, default=0.00)
    parser.add_argument("--pid-kd", type=float, default=0.05)
    parser.add_argument("--pid-slope-gain", type=float, default=0.60)
    parser.add_argument("--pid-integral-limit", type=float, default=0.20)
    parser.add_argument("--derivative-limit", type=float, default=2.5)

    # SMC
    parser.add_argument("--smc-error-gain", type=float, default=0.95)
    parser.add_argument("--smc-slope-gain", type=float, default=0.45)
    parser.add_argument("--smc-switch-gain", type=float, default=0.08)
    parser.add_argument("--smc-lambda", type=float, default=1.20)
    parser.add_argument("--smc-boundary", type=float, default=0.45)
    parser.add_argument("--smc-surface-slope-gain", type=float, default=0.30)

    # NMPC
    parser.add_argument("--nmpc-horizon", type=int, default=8)
    parser.add_argument("--nmpc-candidates", type=int, default=9)
    parser.add_argument("--nmpc-beam", type=int, default=16)
    parser.add_argument("--nmpc-q-error", type=float, default=12.0)
    parser.add_argument("--nmpc-q-slope", type=float, default=0.35)
    parser.add_argument("--nmpc-r-steer", type=float, default=0.18)
    parser.add_argument("--nmpc-r-dsteer", type=float, default=0.65)
    parser.add_argument("--nmpc-model-am", type=float, default=0.25)
    parser.add_argument("--nmpc-model-adelta", type=float, default=1.8)
    parser.add_argument("--nmpc-model-bdelta", type=float, default=0.9)
    parser.add_argument("--nmpc-model-bm", type=float, default=1.2)
    parser.add_argument("--nmpc-rate-limit", type=float, default=0.055)

    # Adaptive HyperLIF-inspired family
    parser.add_argument("--family-kp", type=float, default=3.2)
    parser.add_argument("--family-kd", type=float, default=0.030)
    parser.add_argument("--family-slope-gain", type=float, default=0.75)
    parser.add_argument("--family-gate-alpha", type=float, default=0.92)
    parser.add_argument("--family-gate-beta", type=float, default=0.30)
    parser.add_argument("--family-gate-gain", type=float, default=0.40)
    parser.add_argument("--family-adapt-alpha", type=float, default=0.96)
    parser.add_argument("--family-adapt-beta", type=float, default=0.06)
    parser.add_argument("--family-adapt-gain", type=float, default=0.08)
    parser.add_argument("--family-state-limit", type=float, default=2.0)
    parser.add_argument("--family-output-limit", type=float, default=0.30)
    parser.add_argument("--family-residual-gain", type=float, default=0.08)
    parser.add_argument("--family-residual-limit", type=float, default=0.025)

    # Residual gate
    parser.add_argument("--residual-gating", action="store_true", default=True)
    parser.add_argument("--no-residual-gating", action="store_false", dest="residual_gating")
    parser.add_argument("--residual-gate-error-gain", type=float, default=4.0)
    parser.add_argument("--residual-gate-derivative-gain", type=float, default=0.25)
    parser.add_argument("--residual-gate-slope-gain", type=float, default=1.0)
    parser.add_argument("--residual-gate-pixel-reference", type=float, default=0.006)
    parser.add_argument("--residual-gate-pixel-gain", type=float, default=8.0)
    parser.add_argument("--residual-gate-min", type=float, default=0.0)
    parser.add_argument("--residual-gate-max", type=float, default=1.0)

    # Camera settings
    parser.add_argument("--image-width", type=int, default=1640)
    parser.add_argument("--image-height", type=int, default=820)

    # ROI
    parser.add_argument("--roi-row-start", type=int, default=500)
    parser.add_argument("--roi-row-end", type=int, default=815)
    parser.add_argument("--roi-col-start", type=int, default=80)
    parser.add_argument("--roi-col-end", type=int, default=1580)
    parser.add_argument("--lookahead-fraction", type=float, default=0.75)
    parser.add_argument("--desired-lane-x-fraction", type=float, default=0.50)

    # HSV yellow threshold
    parser.add_argument("--hsv-low-h", type=int, default=10)
    parser.add_argument("--hsv-low-s", type=int, default=45)
    parser.add_argument("--hsv-low-v", type=int, default=80)
    parser.add_argument("--hsv-high-h", type=int, default=45)
    parser.add_argument("--hsv-high-s", type=int, default=255)
    parser.add_argument("--hsv-high-v", type=int, default=255)

    # Lane validation
    parser.add_argument("--min-lane-pixels", type=float, default=0.00001)
    parser.add_argument("--max-lane-pixels", type=float, default=0.35)
    parser.add_argument("--min-fit-pixels", type=int, default=5)
    parser.add_argument("--hold-last-valid", type=float, default=0.50)
    parser.add_argument("--hold-decay", type=float, default=0.80)

    # Display/logging
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--log-prefix", type=str, default="qcar2_hardware_v1")

    return parser.parse_args()


# =============================================================================
# Perception
# =============================================================================

def fit_lane_x_from_y(
    binary_float: np.ndarray,
    *,
    min_fit_pixels: int,
) -> tuple[float, float, float, bool, str]:
    """
    Fit yellow lane as x = m*y + b inside ROI.
    """
    import cv2

    mask = (binary_float > 0.0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    yellow_pixel_fraction = float(np.mean(mask > 0.0))
    ys, xs = np.nonzero(mask)

    if len(xs) < min_fit_pixels:
        return math.nan, math.nan, yellow_pixel_fraction, False, "too_few_fit_pixels"

    slope_xy, intercept_xy = np.polyfit(ys.astype(float), xs.astype(float), 1)
    return float(slope_xy), float(intercept_xy), yellow_pixel_fraction, True, "ok"


def compute_lane_state_from_fit(
    *,
    slope_xy: float,
    intercept_xy: float,
    crop_width: int,
    crop_height: int,
    lookahead_fraction: float,
    desired_lane_x_fraction: float,
) -> tuple[float, float, float]:
    lookahead_y = int(lookahead_fraction * crop_height)
    lane_x = slope_xy * lookahead_y + intercept_xy
    desired_x = desired_lane_x_fraction * crop_width
    lane_error = (lane_x - desired_x) / crop_width
    return float(lane_x), float(desired_x), float(lane_error)


def get_lane_measurement(
    *,
    front: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import cv2
    from hal.utilities.image_processing import ImageProcessing

    crop = front[
        args.roi_row_start:args.roi_row_end,
        args.roi_col_start:args.roi_col_end,
    ].copy()

    lower_yellow = np.array([args.hsv_low_h, args.hsv_low_s, args.hsv_low_v], dtype=np.uint8)
    upper_yellow = np.array([args.hsv_high_h, args.hsv_high_s, args.hsv_high_v], dtype=np.uint8)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    binary = ImageProcessing.binary_thresholding(
        frame=hsv,
        lowerBounds=lower_yellow,
        upperBounds=upper_yellow,
    )
    binary_float = binary / 255.0

    slope_xy, intercept_xy, yellow_pixel_fraction, lane_valid, reason = fit_lane_x_from_y(
        binary_float,
        min_fit_pixels=args.min_fit_pixels,
    )

    if lane_valid and yellow_pixel_fraction < args.min_lane_pixels:
        lane_valid = False
        reason = "too_few_yellow_pixels"
    elif lane_valid and yellow_pixel_fraction > args.max_lane_pixels:
        lane_valid = False
        reason = "too_many_yellow_pixels"

    crop_height = args.roi_row_end - args.roi_row_start
    crop_width = args.roi_col_end - args.roi_col_start

    lane_x = math.nan
    desired_x = args.desired_lane_x_fraction * crop_width
    lane_error = math.nan

    if lane_valid:
        lane_x, desired_x, lane_error = compute_lane_state_from_fit(
            slope_xy=slope_xy,
            intercept_xy=intercept_xy,
            crop_width=crop_width,
            crop_height=crop_height,
            lookahead_fraction=args.lookahead_fraction,
            desired_lane_x_fraction=args.desired_lane_x_fraction,
        )

    return {
        "crop": crop,
        "binary": binary,
        "binary_float": binary_float,
        "lane_valid": bool(lane_valid),
        "lane_valid_reason": str(reason),
        "yellow_pixel_fraction": float(yellow_pixel_fraction),
        "slope_xy": float(slope_xy),
        "intercept_xy": float(intercept_xy),
        "lane_x": float(lane_x),
        "desired_x": float(desired_x),
        "lane_error": float(lane_error),
    }


# =============================================================================
# Controllers
# =============================================================================

class ControllerBase:
    name = "base"

    def step(self, state: LaneState) -> ControlOutput:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PIDLaneController(ControllerBase):
    name = "pid"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.integral_error = 0.0

    def step(self, state: LaneState) -> ControlOutput:
        de = float(np.clip(state.d_lane_error, -self.args.derivative_limit, self.args.derivative_limit))
        self.integral_error += state.lane_error * max(state.loop_dt, 1e-6)
        self.integral_error = float(
            np.clip(
                self.integral_error,
                -self.args.pid_integral_limit,
                self.args.pid_integral_limit,
            )
        )

        raw = (
            -self.args.pid_kp * state.lane_error
            -self.args.pid_ki * self.integral_error
            -self.args.pid_kd * de
            -self.args.pid_slope_gain * state.slope_xy
        )

        return ControlOutput(
            raw_steering=float(raw),
            info={"pid_integral": float(self.integral_error), "de_limited": float(de)},
        )


class SMCLaneController(ControllerBase):
    name = "smc"

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @staticmethod
    def sat(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    def step(self, state: LaneState) -> ControlOutput:
        de = float(np.clip(state.d_lane_error, -self.args.derivative_limit, self.args.derivative_limit))
        sigma = de + self.args.smc_lambda * state.lane_error + self.args.smc_surface_slope_gain * state.slope_xy
        boundary = max(float(self.args.smc_boundary), 1e-6)
        switching = self.sat(sigma / boundary)

        raw = (
            -self.args.smc_error_gain * state.lane_error
            -self.args.smc_slope_gain * state.slope_xy
            -self.args.smc_switch_gain * switching
        )

        return ControlOutput(
            raw_steering=float(raw),
            info={
                "smc_sigma": float(sigma),
                "smc_switching": float(switching),
                "de_limited": float(de),
            },
        )


class ImageNMPCController(ControllerBase):
    name = "nmpc_image"

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _rollout_cost(self, e0: float, m0: float, delta_prev: float, seq: list[float], dt: float) -> float:
        e = float(e0)
        m = float(m0)
        prev = float(delta_prev)
        cost = 0.0

        for delta in seq:
            ddelta = delta - prev
            cost += (
                self.args.nmpc_q_error * e * e
                + self.args.nmpc_q_slope * m * m
                + self.args.nmpc_r_steer * delta * delta
                + self.args.nmpc_r_dsteer * ddelta * ddelta
            )

            e_next = e + dt * (self.args.nmpc_model_am * m + self.args.nmpc_model_adelta * delta)
            m_next = m + dt * (self.args.nmpc_model_bdelta * delta - self.args.nmpc_model_bm * m)

            e, m, prev = float(e_next), float(m_next), float(delta)

        return float(cost)

    def step(self, state: LaneState) -> ControlOutput:
        dt = max(float(state.loop_dt), 1e-6)
        max_delta = float(self.args.max_steering)
        rate = max(float(self.args.nmpc_rate_limit), 1e-6)
        n_candidates = max(int(self.args.nmpc_candidates), 3)

        beams: list[tuple[float, list[float]]] = [(0.0, [])]

        for _ in range(int(self.args.nmpc_horizon)):
            new_beams: list[tuple[float, list[float]]] = []

            for _, seq in beams:
                prev = float(state.previous_steering if not seq else seq[-1])
                local_grid = np.linspace(prev - rate, prev + rate, n_candidates)
                local_grid = np.clip(local_grid, -max_delta, max_delta)
                local_grid = np.unique(np.round(local_grid, decimals=6))

                for delta in local_grid:
                    candidate = seq + [float(delta)]
                    cost = self._rollout_cost(
                        state.lane_error,
                        state.slope_xy,
                        state.previous_steering,
                        candidate,
                        dt,
                    )
                    new_beams.append((cost, candidate))

            if not new_beams:
                break

            new_beams.sort(key=lambda item: item[0])
            beams = new_beams[: int(self.args.nmpc_beam)]

        if not beams or not beams[0][1]:
            raw = -self.args.pid_kp * state.lane_error - self.args.pid_slope_gain * state.slope_xy
            return ControlOutput(raw_steering=float(raw), info={"nmpc_fallback": True})

        best_cost, best_seq = beams[0]
        raw = float(best_seq[0])

        return ControlOutput(
            raw_steering=raw,
            info={
                "nmpc_cost": float(best_cost),
                "nmpc_first_delta": float(raw),
                "nmpc_seq0": best_seq[:3],
                "nmpc_rate_limit": float(rate),
                "nmpc_candidates": int(n_candidates),
            },
        )


class AdaptiveHyperLIFGateController(ControllerBase):
    name = "adaptive_hyperlif_gate"

    def __init__(self, args: argparse.Namespace, *, residual_mode: bool = False):
        self.args = args
        self.residual_mode = residual_mode
        self.gate_state = 0.0
        self.adapt_state = 0.0
        self.last_output = 0.0

    def _base_pd(self, state: LaneState) -> float:
        de = float(np.clip(state.d_lane_error, -self.args.derivative_limit, self.args.derivative_limit))
        return float(
            -self.args.family_kp * state.lane_error
            -self.args.family_kd * de
            -self.args.family_slope_gain * state.slope_xy
        )

    def _full_output(self, state: LaneState) -> tuple[float, dict[str, float]]:
        de = float(np.clip(state.d_lane_error, -self.args.derivative_limit, self.args.derivative_limit))

        demand = abs(float(state.lane_error)) + 0.25 * abs(de) + 0.75 * abs(float(state.slope_xy))
        demand = float(np.clip(demand, 0.0, self.args.family_state_limit))

        self.gate_state = self.args.family_gate_alpha * self.gate_state + self.args.family_gate_beta * demand
        self.gate_state = float(np.clip(self.gate_state, 0.0, self.args.family_state_limit))

        base = self._base_pd(state)
        gated = (1.0 + self.args.family_gate_gain * self.gate_state) * base

        self.adapt_state = self.args.family_adapt_alpha * self.adapt_state + self.args.family_adapt_beta * self.last_output
        self.adapt_state = float(
            np.clip(self.adapt_state, -self.args.family_state_limit, self.args.family_state_limit)
        )

        output = gated - self.args.family_adapt_gain * self.adapt_state
        output = float(np.clip(output, -self.args.family_output_limit, self.args.family_output_limit))
        self.last_output = output

        info = {
            "family_demand": float(demand),
            "family_gate_state": float(self.gate_state),
            "family_adapt_state": float(self.adapt_state),
            "family_base": float(base),
            "family_gated": float(gated),
        }

        return output, info

    def step(self, state: LaneState) -> ControlOutput:
        full, info = self._full_output(state)

        if not self.residual_mode:
            return ControlOutput(raw_steering=full, residual_steering=0.0, info=info)

        base = self._base_pd(state)
        residual = self.args.family_residual_gain * (full - base)
        residual = float(np.clip(residual, -self.args.family_residual_limit, self.args.family_residual_limit))
        info["family_residual"] = float(residual)

        return ControlOutput(raw_steering=residual, residual_steering=residual, info=info)


class HybridResidualController(ControllerBase):
    def __init__(self, base: ControllerBase, residual: ControllerBase, args: argparse.Namespace, name: str):
        self.base = base
        self.residual = residual
        self.args = args
        self.name = name

    def _residual_gate(self, state: LaneState) -> tuple[float, dict[str, float]]:
        if not getattr(self.args, "residual_gating", True):
            return 1.0, {
                "gate_enabled": 0.0,
                "gate_error": 1.0,
                "gate_derivative": 1.0,
                "gate_slope": 1.0,
                "gate_pixel": 1.0,
            }

        if not bool(state.lane_valid):
            return 0.0, {
                "gate_enabled": 1.0,
                "gate_error": 0.0,
                "gate_derivative": 0.0,
                "gate_slope": 0.0,
                "gate_pixel": 0.0,
            }

        de = float(np.clip(state.d_lane_error, -self.args.derivative_limit, self.args.derivative_limit))

        ge = math.exp(-float(self.args.residual_gate_error_gain) * abs(float(state.lane_error)))
        gd = math.exp(-float(self.args.residual_gate_derivative_gain) * abs(de))
        gs = math.exp(-float(self.args.residual_gate_slope_gain) * abs(float(state.slope_xy)))

        pix_ref = max(float(self.args.residual_gate_pixel_reference), 1e-9)
        pix = max(float(state.yellow_pixel_fraction), 0.0)
        gp = 1.0 - math.exp(-float(self.args.residual_gate_pixel_gain) * pix / pix_ref)
        gp = float(np.clip(gp, 0.0, 1.0))

        gamma = ge * gd * gs * gp
        gamma = float(np.clip(gamma, float(self.args.residual_gate_min), float(self.args.residual_gate_max)))

        return gamma, {
            "gate_enabled": 1.0,
            "gate_error": float(ge),
            "gate_derivative": float(gd),
            "gate_slope": float(gs),
            "gate_pixel": float(gp),
            "gate_yellow_pixel_fraction": float(state.yellow_pixel_fraction),
        }

    def step(self, state: LaneState) -> ControlOutput:
        base_out = self.base.step(state)
        residual_out = self.residual.step(state)

        residual_raw = float(residual_out.residual_steering)
        gate, gate_info = self._residual_gate(state)
        residual = float(gate * residual_raw)

        raw = float(base_out.raw_steering + residual)

        info: dict[str, Any] = {
            "base_raw": float(base_out.raw_steering),
            "residual_raw": float(residual_raw),
            "residual_gate": float(gate),
            "residual": float(residual),
            "base_info": base_out.info or {},
            "residual_info": residual_out.info or {},
            "gate_info": gate_info,
        }

        return ControlOutput(raw_steering=raw, residual_steering=residual, info=info)

    def close(self) -> None:
        self.base.close()
        self.residual.close()


def build_controller(args: argparse.Namespace) -> ControllerBase:
    if args.controller == "pid":
        return PIDLaneController(args)

    if args.controller == "smc":
        return SMCLaneController(args)

    if args.controller == "nmpc_image":
        return ImageNMPCController(args)

    if args.controller == "adaptive_hyperlif_gate":
        return AdaptiveHyperLIFGateController(args, residual_mode=False)

    if args.controller == "pid_gated_adaptive_hyperlif_residual":
        return HybridResidualController(
            PIDLaneController(args),
            AdaptiveHyperLIFGateController(args, residual_mode=True),
            args,
            name=args.controller,
        )

    if args.controller == "smc_gated_adaptive_hyperlif_residual":
        return HybridResidualController(
            SMCLaneController(args),
            AdaptiveHyperLIFGateController(args, residual_mode=True),
            args,
            name=args.controller,
        )

    if args.controller == "nmpc_gated_adaptive_hyperlif_residual":
        return HybridResidualController(
            ImageNMPCController(args),
            AdaptiveHyperLIFGateController(args, residual_mode=True),
            args,
            name=args.controller,
        )

    raise ValueError(f"Unsupported controller: {args.controller}")


# =============================================================================
# Throttle and post-processing
# =============================================================================

def apply_steering_post_processing(
    *,
    raw_steering: float,
    previous_steering: float,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """
    Clip, low-pass filter, and rate-limit steering in controller coordinates.
    Hardware sign correction is NOT applied here.
    """
    clipped = float(np.clip(raw_steering, -args.max_steering, args.max_steering))

    alpha = float(np.clip(args.steering_alpha, 0.0, 0.999))
    filtered = alpha * float(previous_steering) + (1.0 - alpha) * clipped

    max_delta = max(float(args.steering_rate_limit), 1e-6)
    delta = float(np.clip(filtered - float(previous_steering), -max_delta, max_delta))
    command = float(previous_steering + delta)

    command = float(np.clip(command, -args.max_steering, args.max_steering))
    return command, delta


def compute_throttle(
    *,
    lane_valid: bool,
    lane_error: float,
    command_steering: float,
    args: argparse.Namespace,
) -> float:
    if not lane_valid:
        return float(args.lost_lane_throttle)

    # Simple conservative scheduling for hardware V1:
    # slower with larger steering and larger lane error.
    score = math.exp(-5.0 * abs(float(lane_error)) - 2.0 * abs(float(command_steering)))
    throttle = args.min_throttle + (args.max_throttle - args.min_throttle) * score
    throttle = min(throttle, args.base_throttle if args.base_throttle > 0 else throttle)
    return float(np.clip(throttle, 0.0, args.max_throttle))


# =============================================================================
# Visualization
# =============================================================================

def show_dashboard(
    *,
    front: np.ndarray,
    measurement: dict[str, Any],
    command_throttle: float,
    command_steering_controller: float,
    command_steering_hardware: float,
    control_mode: str,
    args: argparse.Namespace,
) -> bool:
    """
    Returns True if ESC was pressed.
    """
    import cv2

    overlay = front.copy()
    binary_float = measurement["binary_float"]

    roi = overlay[
        args.roi_row_start:args.roi_row_end,
        args.roi_col_start:args.roi_col_end,
    ]

    roi[:, :, 2] = roi[:, :, 2] + (255 - roi[:, :, 2]) * binary_float
    roi[:, :, 1] = roi[:, :, 1] * (1 - binary_float)
    roi[:, :, 0] = roi[:, :, 0] * (1 - binary_float)

    cv2.rectangle(
        overlay,
        (args.roi_col_start, args.roi_row_start),
        (args.roi_col_end - 1, args.roi_row_end - 1),
        (255, 255, 255),
        2,
    )

    color = (0, 255, 0) if measurement["lane_valid"] else (0, 0, 255)
    text1 = f"{args.mode} | {args.controller} | {measurement['lane_valid_reason']} | {control_mode}"
    text2 = (
        f"e={measurement['lane_error']:+.3f} "
        f"m={measurement['slope_xy']:+.3f} "
        f"thr={command_throttle:+.3f} "
        f"ctrl={command_steering_controller:+.3f} "
        f"hw={command_steering_hardware:+.3f}"
    )

    cv2.putText(overlay, text1, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, text2, (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    view = cv2.resize(overlay, (960, 480))
    mask = cv2.resize(measurement["binary"], (960, 220))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    dashboard = np.vstack([view, mask_bgr])

    cv2.imshow("QCar2 GHYN hardware V1", dashboard)
    key = cv2.waitKey(1) & 0xFF
    return key == 27


# =============================================================================
# Main hardware loop
# =============================================================================

def run_hardware(args: argparse.Namespace) -> None:
    import cv2
    from pal.products.qcar import QCar, QCarCameras

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{args.log_prefix}_{args.mode}_{args.controller}_{timestamp}.csv"

    sample_time = 1.0 / args.sample_rate
    controller = build_controller(args)

    stop_requested = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("\nStop requested. Sending zero command and closing hardware loop...")

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    qcar = None
    qcar_cameras = None

    previous_lane_error: Optional[float] = None
    previous_lane_error_time: Optional[float] = None
    previous_steering_controller = 0.0
    previous_good_steering_controller = 0.0
    previous_good_time = -math.inf
    previous_loop_time = time.time()
    last_camera_time = time.time()

    if args.mode == "lane" and not args.armed:
        print("\nWARNING: lane mode requested without --armed.")
        print("The QCar will run perception and logging only, with zero throttle/steering.")
        print("Add --armed only after the perception-only test is confirmed.\n")

    print("Starting QCar2 GHYN hardware V1.")
    print(f"Mode: {args.mode}")
    print(f"Controller: {args.controller}")
    print(f"Armed: {args.armed}")
    print(f"Steering sign: {args.steering_sign}")
    print(f"Log: {log_path}")
    print("Press Ctrl+C to stop safely.")

    fieldnames = [
        "time_s",
        "mode",
        "controller",
        "armed",
        "control_mode",
        "lane_valid",
        "lane_valid_reason",
        "yellow_pixel_fraction",
        "lane_slope_xy",
        "lane_intercept_xy",
        "lane_x",
        "desired_x",
        "lane_error",
        "d_lane_error",
        "raw_steering_controller",
        "residual_steering",
        "command_steering_controller",
        "command_steering_hardware",
        "command_throttle",
        "steering_rate_delta",
        "steering_sign",
        "battery_voltage",
        "motor_tach",
        "loop_dt",
        "watchdog_active",
        "state_json",
        "controller_info_json",
    ]

    try:
        qcar_cameras = QCarCameras(
            frameWidth=args.image_width,
            frameHeight=args.image_height,
            frameRate=args.sample_rate,
            enableFront=True,
        )
        qcar = QCar(readMode=1, frequency=args.sample_rate)

        with log_path.open("w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()

            start_time = time.time()

            while not stop_requested and time.time() - start_time < args.duration:
                loop_start = time.time()
                elapsed = loop_start - start_time
                loop_dt = max(loop_start - previous_loop_time, sample_time)
                previous_loop_time = loop_start

                qcar_cameras.readAll()
                last_camera_time = time.time()
                front = qcar_cameras.csiFront.imageData

                measurement = get_lane_measurement(front=front, args=args)

                lane_valid = bool(measurement["lane_valid"])
                lane_error = float(measurement["lane_error"])
                slope_xy = float(measurement["slope_xy"])
                d_lane_error = 0.0

                if lane_valid:
                    if previous_lane_error is None or previous_lane_error_time is None:
                        d_lane_error = 0.0
                    else:
                        dt_error = max(elapsed - previous_lane_error_time, 1e-6)
                        d_lane_error = (lane_error - previous_lane_error) / dt_error

                    previous_lane_error = lane_error
                    previous_lane_error_time = elapsed

                raw_steering_controller = 0.0
                residual_steering = 0.0
                controller_info: dict[str, Any] = {}
                steering_rate_delta = 0.0
                command_steering_controller = 0.0
                command_steering_hardware = 0.0
                command_throttle = 0.0
                control_mode = "perception_only"
                watchdog_active = False

                if args.mode == "lane" and lane_valid:
                    state = LaneState(
                        lane_error=float(lane_error),
                        d_lane_error=float(d_lane_error),
                        slope_xy=float(slope_xy),
                        previous_steering=float(previous_steering_controller),
                        loop_dt=float(loop_dt),
                        lane_valid=True,
                        yellow_pixel_fraction=float(measurement["yellow_pixel_fraction"]),
                        lane_valid_reason=str(measurement["lane_valid_reason"]),
                    )

                    control = controller.step(state)
                    raw_steering_controller = float(control.raw_steering)
                    residual_steering = float(control.residual_steering)
                    controller_info = control.info or {}

                    command_steering_controller, steering_rate_delta = apply_steering_post_processing(
                        raw_steering=raw_steering_controller,
                        previous_steering=previous_steering_controller,
                        args=args,
                    )

                    command_throttle = compute_throttle(
                        lane_valid=True,
                        lane_error=lane_error,
                        command_steering=command_steering_controller,
                        args=args,
                    )

                    previous_steering_controller = command_steering_controller
                    previous_good_steering_controller = command_steering_controller
                    previous_good_time = elapsed
                    control_mode = "lane_control"

                elif args.mode == "lane" and not lane_valid:
                    time_since_good = elapsed - previous_good_time
                    if time_since_good <= args.hold_last_valid:
                        command_steering_controller = float(
                            np.clip(
                                previous_good_steering_controller * args.hold_decay,
                                -args.max_steering,
                                args.max_steering,
                            )
                        )
                        previous_steering_controller = command_steering_controller
                        command_throttle = float(args.lost_lane_throttle)
                        control_mode = "hold_previous"
                    else:
                        command_steering_controller = 0.0
                        previous_steering_controller = 0.0
                        command_throttle = 0.0
                        control_mode = "lost_lane_stop"

                    state = LaneState(
                        lane_error=0.0,
                        d_lane_error=0.0,
                        slope_xy=0.0,
                        previous_steering=float(previous_steering_controller),
                        loop_dt=float(loop_dt),
                        lane_valid=False,
                        yellow_pixel_fraction=float(measurement["yellow_pixel_fraction"]),
                        lane_valid_reason=str(measurement["lane_valid_reason"]),
                    )
                else:
                    state = LaneState(
                        lane_error=0.0 if math.isnan(lane_error) else float(lane_error),
                        d_lane_error=0.0,
                        slope_xy=0.0 if math.isnan(slope_xy) else float(slope_xy),
                        previous_steering=float(previous_steering_controller),
                        loop_dt=float(loop_dt),
                        lane_valid=lane_valid,
                        yellow_pixel_fraction=float(measurement["yellow_pixel_fraction"]),
                        lane_valid_reason=str(measurement["lane_valid_reason"]),
                    )

                # Watchdog: if the loop timing stalls, stop.
                if (time.time() - last_camera_time) > args.watchdog_timeout:
                    watchdog_active = True

                # --armed is required for any physical motion.
                if not args.armed:
                    command_throttle = 0.0
                    command_steering_controller = 0.0
                    command_steering_hardware = 0.0
                    control_mode = "not_armed_zero_output"
                elif watchdog_active:
                    command_throttle = 0.0
                    command_steering_controller = 0.0
                    command_steering_hardware = 0.0
                    control_mode = "watchdog_stop"
                else:
                    command_steering_hardware = float(args.steering_sign * command_steering_controller)
                    command_steering_hardware = float(
                        np.clip(command_steering_hardware, -args.max_steering, args.max_steering)
                    )

                qcar.read_write_std(
                    throttle=float(command_throttle),
                    steering=float(command_steering_hardware),
                    LEDs=default_leds(command_throttle, command_steering_hardware),
                )

                battery_voltage = getattr(qcar, "batteryVoltage", math.nan)
                motor_tach = getattr(qcar, "motorTach", math.nan)

                writer.writerow({
                    "time_s": float(elapsed),
                    "mode": args.mode,
                    "controller": args.controller,
                    "armed": int(bool(args.armed)),
                    "control_mode": control_mode,
                    "lane_valid": int(lane_valid),
                    "lane_valid_reason": measurement["lane_valid_reason"],
                    "yellow_pixel_fraction": float(measurement["yellow_pixel_fraction"]),
                    "lane_slope_xy": float(measurement["slope_xy"]),
                    "lane_intercept_xy": float(measurement["intercept_xy"]),
                    "lane_x": float(measurement["lane_x"]),
                    "desired_x": float(measurement["desired_x"]),
                    "lane_error": float(measurement["lane_error"]),
                    "d_lane_error": float(d_lane_error),
                    "raw_steering_controller": float(raw_steering_controller),
                    "residual_steering": float(residual_steering),
                    "command_steering_controller": float(command_steering_controller),
                    "command_steering_hardware": float(command_steering_hardware),
                    "command_throttle": float(command_throttle),
                    "steering_rate_delta": float(steering_rate_delta),
                    "steering_sign": float(args.steering_sign),
                    "battery_voltage": float(battery_voltage),
                    "motor_tach": float(motor_tach),
                    "loop_dt": float(loop_dt),
                    "watchdog_active": int(bool(watchdog_active)),
                    "state_json": json.dumps(asdict(state)),
                    "controller_info_json": json.dumps(controller_info, default=str),
                })

                if not args.no_display:
                    esc = show_dashboard(
                        front=front,
                        measurement=measurement,
                        command_throttle=command_throttle,
                        command_steering_controller=command_steering_controller,
                        command_steering_hardware=command_steering_hardware,
                        control_mode=control_mode,
                        args=args,
                    )
                    if esc:
                        print("\nESC pressed.")
                        break

                print(
                    f"t={elapsed:5.1f}s | {control_mode:22s} | "
                    f"valid={int(lane_valid)} | "
                    f"e={measurement['lane_error']:+.3f} | "
                    f"m={measurement['slope_xy']:+.3f} | "
                    f"ctrl={command_steering_controller:+.3f} | "
                    f"hw={command_steering_hardware:+.3f} | "
                    f"thr={command_throttle:+.3f} | "
                    f"bat={battery_voltage:4.2f}",
                    end="\r",
                    flush=True,
                )

                sleep_time = sample_time - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        print("\nCommanding QCar2 to stop...")
        safe_stop_car(qcar)
        controller.close()
        safe_terminate(qcar_cameras)
        safe_terminate(qcar)

        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        print(f"Finished safely. Log saved to: {log_path}")


def main() -> None:
    args = parse_args()
    run_hardware(args)


if __name__ == "__main__":
    main()
