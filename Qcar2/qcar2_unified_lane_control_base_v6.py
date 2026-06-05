"""
qcar2_unified_lane_control_base_v6.py

Unified QCar2 lane-following base for continuity of the project.

Purpose
-------
This script consolidates the current frozen QCar2 scripts into one reusable base. V6 adds turn-aware speed/steering management so higher max speed and max steering can be used without globally destabilizing PID/SMC/NMPC:

1) Manual positioning mode.
2) Fixed yellow-lane perception pipeline.
3) Shared lane-state vector:
       z = [lane_error, d_lane_error, slope_xy, previous_steering]
4) Swappable controllers:
       pid
       smc
       nmpc_image
       nengo_lif_pd
       adaptive_hyperlif_gate
       pid_adaptive_hyperlif_residual
       smc_adaptive_hyperlif_residual
       nmpc_adaptive_hyperlif_residual
       full_nengo_lif
5) Shared safety, dashboard, CSV logging, adaptive throttle, high-speed stabilizer, and residual-confidence gating.

Core lane equations
-------------------
ROI:                     I_roi = I[r0:r1, c0:c1]
Yellow mask:             M(y,x) = 1 if HSV is inside yellow bounds, else 0
Lane fit:                x = m y + b
Lookahead:               y_L = alpha * H_roi,  x_L = m*y_L + b
Desired x:               x_d = beta * W_roi
Lane error:              e = (x_L - x_d)/W_roi
Derivative:              de = (e[k] - e[k-1])/dt
State vector:            z = [e, de, m, delta_prev]

Controller output
-----------------
All controllers return raw steering delta_raw. Afterward, the common post-processing is:

1) clipping
2) low-pass filtering
3) optional speed-scheduled steering reduction
4) optional steering slew-rate limit
5) throttle scheduling
6) qcar.read_write_std(throttle, steering, LEDs)

Notes
-----
- V6 is intentionally performance-oriented but turn-aware: max steering/speed remain available, while deceleration and steering-rate authority adapt during turns.
- NMPC here uses an image-state predictive model, not yet a full kinematic bicycle model.
- Full Nengo currently represents a PD-like steering law. Later, replace the decoded function or train it from logged data.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUANSER_LIBRARIES = Path(r"C:\Quanser\0_libraries\python")
RESULTS_DIR = PROJECT_ROOT / "results" / "qcar" / "logs"

# V4 preset intent:
# - SMC keeps the strongest V2 baseline gains.
# - NMPC keeps the corrected/high-performing V3 image-state model and weights.
# - PID uses the improved V3 performance gains with more damping than the original PD.
# - Residual hooks are available for both Adaptive HyperLIF and Nengo LIF.
# - Global steering/speed defaults are less conservative for real-application envelope testing.


# =============================================================================
# Environment / QCar selection
# =============================================================================


def configure_paths() -> None:
    """Configure Quanser Python paths and RT model directory."""
    os.environ.setdefault(
        "RTMODELS_DIR",
        r"C:\Quanser\0_libraries\resources\rt_models",
    )

    for path in (PROJECT_ROOT / "src", QUANSER_LIBRARIES):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.append(path_text)



def select_virtual_qcar(qcar_type: str) -> None:
    """Automatically answer Quanser's virtual QCar1/QCar2 prompt."""
    original_input = builtins.input

    def patched_input(prompt: str = "") -> str:
        if "virtual QCar1 or QCar2" in prompt:
            print(f"{prompt}{qcar_type}")
            return qcar_type
        return original_input(prompt)

    builtins.input = patched_input


# =============================================================================
# Dataclasses
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
    info: dict[str, Any] | None = None


# =============================================================================
# LED and safety helpers
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
    """Send zero throttle/steering twice before termination."""
    if car is None:
        return
    try:
        car.read_write_std(
            throttle=0.0,
            steering=0.0,
            LEDs=default_leds(0.0, 0.0),
        )
        time.sleep(0.05)
        car.read_write_std(
            throttle=0.0,
            steering=0.0,
            LEDs=default_leds(0.0, 0.0),
        )
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
        description="Unified QCar2 lane-following base V6: turn-aware performance baselines plus gated Adaptive HyperLIF and gated Nengo residual augmentation."
    )

    # Mode
    parser.add_argument("--mode", choices=["manual", "lane"], default="lane")
    parser.add_argument("--qcar-type", choices=["1", "2"], default="2")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sample-rate", type=float, default=60.0)

    # Manual drive
    parser.add_argument("--manual-max-throttle", type=float, default=0.08)
    parser.add_argument("--manual-max-steer", type=float, default=0.45)

    # Controller selection
    parser.add_argument(
        "--controller",
        choices=[
            "pid",
            "smc",
            "nmpc_image",
            "nengo_lif_pd",
            "adaptive_hyperlif_gate",
            "pid_adaptive_hyperlif_residual",
            "smc_adaptive_hyperlif_residual",
            "nmpc_adaptive_hyperlif_residual",
            "pid_nengo_lif_residual",
            "smc_nengo_lif_residual",
            "nmpc_nengo_lif_residual",
            "pid_gated_adaptive_hyperlif_residual",
            "smc_gated_adaptive_hyperlif_residual",
            "nmpc_gated_adaptive_hyperlif_residual",
            "pid_gated_nengo_lif_residual",
            "smc_gated_nengo_lif_residual",
            "nmpc_gated_nengo_lif_residual",
            "full_nengo_lif",
        ],
        default="pid",
    )

    # Common steering settings
    parser.add_argument("--max-steering", type=float, default=0.45)
    parser.add_argument("--filter-cutoff", type=float, default=10.0)
    parser.add_argument("--derivative-limit", type=float, default=3.0)

    # PID gains
    parser.add_argument("--pid-kp", type=float, default=1.9)
    parser.add_argument("--pid-ki", type=float, default=0.0)
    parser.add_argument("--pid-kd", type=float, default=0.10)
    parser.add_argument("--pid-slope-gain", type=float, default=1.15)
    parser.add_argument("--pid-integral-limit", type=float, default=0.25)

    # SMC gains
    parser.add_argument("--smc-error-gain", type=float, default=1.35)
    parser.add_argument("--smc-slope-gain", type=float, default=0.8)
    parser.add_argument("--smc-switch-gain", type=float, default=0.14)
    parser.add_argument("--smc-lambda", type=float, default=1.8)
    parser.add_argument("--smc-boundary", type=float, default=0.38)
    parser.add_argument("--smc-surface-slope-gain", type=float, default=0.5)

    # Image-state NMPC settings
    parser.add_argument("--nmpc-horizon", type=int, default=10)
    parser.add_argument("--nmpc-candidates", type=int, default=11)
    parser.add_argument("--nmpc-beam", type=int, default=20)
    parser.add_argument("--nmpc-q-error", type=float, default=16.0)
    parser.add_argument("--nmpc-q-slope", type=float, default=0.4)
    parser.add_argument("--nmpc-r-steer", type=float, default=0.14)
    parser.add_argument("--nmpc-r-dsteer", type=float, default=0.55)
    parser.add_argument("--nmpc-model-am", type=float, default=0.25)
    parser.add_argument("--nmpc-model-adelta", type=float, default=1.8)
    parser.add_argument("--nmpc-model-bdelta", type=float, default=0.9)
    parser.add_argument("--nmpc-model-bm", type=float, default=1.2)
    parser.add_argument("--nmpc-rate-limit", type=float, default=0.08)

    # Nengo settings
    parser.add_argument("--nengo-neurons", type=int, default=400)
    parser.add_argument("--nengo-radius", type=float, default=1.5)
    parser.add_argument("--nengo-synapse", type=float, default=0.01)
    parser.add_argument("--nengo-seed", type=int, default=7)
    parser.add_argument("--nengo-residual-gain", type=float, default=0.18)
    parser.add_argument("--nengo-residual-limit", type=float, default=0.05)

    # Adaptive HyperLIF-inspired gate settings
    parser.add_argument("--family-kp", type=float, default=4.5)
    parser.add_argument("--family-kd", type=float, default=0.035)
    parser.add_argument("--family-slope-gain", type=float, default=1.15)
    parser.add_argument("--family-gate-alpha", type=float, default=0.92)
    parser.add_argument("--family-gate-beta", type=float, default=0.35)
    parser.add_argument("--family-gate-gain", type=float, default=0.55)
    parser.add_argument("--family-adapt-alpha", type=float, default=0.96)
    parser.add_argument("--family-adapt-beta", type=float, default=0.08)
    parser.add_argument("--family-adapt-gain", type=float, default=0.12)
    parser.add_argument("--family-state-limit", type=float, default=2.0)
    parser.add_argument("--family-output-limit", type=float, default=0.45)
    parser.add_argument("--family-residual-gain", type=float, default=0.12)
    parser.add_argument("--family-residual-limit", type=float, default=0.04)

    # V5 residual confidence gating. Residuals help only when perception is reliable
    # and the lane state is inside a recoverable region. This keeps the classical
    # baseline in charge during lane-loss/recovery and lets the neural residual
    # refine tracking inside the valid lane-tracking region.
    parser.add_argument("--residual-gating", action="store_true", default=True)
    parser.add_argument("--no-residual-gating", action="store_false", dest="residual_gating")
    parser.add_argument("--residual-gate-error-gain", type=float, default=4.0)
    parser.add_argument("--residual-gate-derivative-gain", type=float, default=0.25)
    parser.add_argument("--residual-gate-slope-gain", type=float, default=1.0)
    parser.add_argument("--residual-gate-pixel-reference", type=float, default=0.006)
    parser.add_argument("--residual-gate-pixel-gain", type=float, default=8.0)
    parser.add_argument("--residual-gate-min", type=float, default=0.0)
    parser.add_argument("--residual-gate-max", type=float, default=1.0)

    # Lane validation / memory behavior
    parser.add_argument("--min-lane-pixels", type=float, default=0.00001)
    parser.add_argument("--max-lane-pixels", type=float, default=0.35)
    parser.add_argument("--min-fit-pixels", type=int, default=5)
    parser.add_argument("--hold-last-valid", type=float, default=1.0)
    parser.add_argument("--hold-decay", type=float, default=0.85)

    # Camera and ROI
    parser.add_argument("--image-width", type=int, default=1640)
    parser.add_argument("--image-height", type=int, default=820)
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

    # Throttle and speed scheduling
    parser.add_argument("--base-throttle", type=float, default=0.038)
    parser.add_argument("--lost-lane-throttle", type=float, default=0.012)
    parser.add_argument("--adaptive-speed", action="store_true", default=True)
    parser.add_argument("--no-adaptive-speed", action="store_false", dest="adaptive_speed")
    parser.add_argument("--min-throttle", type=float, default=0.025)
    parser.add_argument("--max-throttle", type=float, default=0.090)
    parser.add_argument("--speed-error-gain", type=float, default=10.0)
    parser.add_argument("--speed-steer-gain", type=float, default=2.0)
    parser.add_argument("--speed-slope-gain", type=float, default=4.0)
    parser.add_argument("--speed-derivative-gain", type=float, default=0.4)
    parser.add_argument("--speed-filter-alpha", type=float, default=0.90)
    parser.add_argument("--speed-error-soft-zone", type=float, default=0.025)

    # High-speed steering stabilizer
    parser.add_argument("--high-speed-stabilizer", action="store_true", default=True)
    parser.add_argument("--no-high-speed-stabilizer", action="store_false", dest="high_speed_stabilizer")
    parser.add_argument("--speed-steering-gain", type=float, default=2.4)
    parser.add_argument("--steering-rate-limit", type=float, default=0.060)
    parser.add_argument("--turn-aware-stabilizer", action="store_true", default=True)
    parser.add_argument("--no-turn-aware-stabilizer", action="store_false", dest="turn_aware_stabilizer")
    parser.add_argument("--turn-error-reference", type=float, default=0.12)
    parser.add_argument("--turn-slope-reference", type=float, default=0.12)
    parser.add_argument("--turn-rate-boost", type=float, default=0.85)
    parser.add_argument("--turn-steering-relief", type=float, default=0.55)
    parser.add_argument("--speed-accel-alpha", type=float, default=0.96)
    parser.add_argument("--speed-decel-alpha", type=float, default=0.45)

    # Display and logging
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--show-separate-windows", action="store_true")
    parser.add_argument("--show-binary-lane", action="store_true")
    parser.add_argument("--show-lane-crop", action="store_true")
    parser.add_argument("--dashboard-width", type=int, default=960)
    parser.add_argument("--dashboard-height", type=int, default=520)
    parser.add_argument("--log-prefix", type=str, default="qcar2_unified_lane_control")

    return parser.parse_args()


# =============================================================================
# Perception and lane-state equations
# =============================================================================


def fit_lane_x_from_y(
    binary_float: np.ndarray,
    *,
    min_fit_pixels: int,
) -> tuple[float, float, float, bool, str]:
    """
    Fit yellow lane as x = m*y + b inside ROI.

    Returns:
        slope_xy, intercept_xy, yellow_pixel_fraction, lane_valid, reason
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
    """Compute x_L, x_d, and normalized lane error e."""
    lookahead_y = int(lookahead_fraction * crop_height)
    lane_x = slope_xy * lookahead_y + intercept_xy
    desired_x = desired_lane_x_fraction * crop_width
    lane_error = (lane_x - desired_x) / crop_width
    return float(lane_x), float(desired_x), float(lane_error)


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
    """
    Image-based PID lane controller.

    I[k] = clip(I[k-1] + e[k]*dt, -Imax, Imax)
    delta = -Kp*e - Ki*I - Kd*de - Ks*m
    """

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
            info={"pid_integral": self.integral_error, "de_limited": de},
        )


class SMCLaneController(ControllerBase):
    """
    Image-based sliding mode controller.

    sigma = de + lambda*e + Kpsi*m
    delta = -Ke*e - Km*m - Ks*sat(sigma/phi)
    """

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
            info={"smc_sigma": float(sigma), "smc_switching": float(switching), "de_limited": de},
        )


class ImageNMPCController(ControllerBase):
    """
    Lightweight image-state NMPC.

    State model:
        e[k+1] = e[k] + dt*(a_m*m[k] + a_delta*delta[k])
        m[k+1] = m[k] + dt*(b_delta*delta[k] - b_m*m[k])

    v3 tuning note:
        Uploaded logs showed NMPC often selected steering with the opposite sign
        from the working PID/SMC correction and then lost the lane. Therefore the
        default image-model steering signs were flipped:
            a_delta: -1.8 -> +1.8
            b_delta: -0.9 -> +0.9
        This makes positive steering reduce negative lane error under the local
        image-state model, matching the observed QCar behavior in the logs.

    Cost:
        J = sum(qe*e^2 + qm*m^2 + r*delta^2 + rd*(delta-delta_prev)^2)

    This is a practical first NMPC layer for the current image-only code. Replace it later with
    a kinematic bicycle model when QCar pose/odometry is fused into the state.
    """

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

        # IMPORTANT FIX v2:
        # The first base used a global steering grid and then rejected commands
        # farther than nmpc_rate_limit from the previous steering. With defaults
        # max_steering=0.35, candidates=9, rate=0.04, the global grid spacing was
        # about 0.0875, so from previous_steering=0 almost only delta=0 survived.
        # That made NMPC look like steering was not working.
        #
        # Use a local grid around each beam's previous steering instead:
        #     delta_i in [delta_prev-rate, delta_prev+rate]
        # This preserves slew-rate constraints while allowing nonzero steering.

        beams: list[tuple[float, list[float]]] = [(0.0, [])]
        for _ in range(int(self.args.nmpc_horizon)):
            new_beams: list[tuple[float, list[float]]] = []
            for _, seq in beams:
                prev = float(state.previous_steering if not seq else seq[-1])
                local_grid = np.linspace(prev - rate, prev + rate, n_candidates)
                local_grid = np.clip(local_grid, -max_delta, max_delta)
                # Remove duplicates created by clipping near saturation.
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
                "nmpc_first_delta": raw,
                "nmpc_seq0": best_seq[:3],
                "nmpc_rate_limit": rate,
                "nmpc_candidates": n_candidates,
            },
        )


class NengoLIFController(ControllerBase):
    """
    Nengo LIF controller.

    Input:  z = [e, de, m, delta_prev]
    Output: full steering or residual steering depending on mode.
    """

    name = "nengo_lif"

    def __init__(self, args: argparse.Namespace, dt: float, *, residual_mode: bool = False):
        try:
            import nengo
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Nengo is not installed. Install it with: python -m pip install nengo"
            ) from exc

        self.args = args
        self.residual_mode = residual_mode
        self.state = np.zeros(4, dtype=float)

        with nengo.Network(seed=args.nengo_seed, label="QCar2 unified Nengo LIF") as model:
            input_node = nengo.Node(lambda _t: self.state, size_out=4, label="lane_state_input")
            ens = nengo.Ensemble(
                n_neurons=args.nengo_neurons,
                dimensions=4,
                radius=args.nengo_radius,
                neuron_type=nengo.LIF(),
                label="lane_state_ensemble",
            )
            output_node = nengo.Node(size_in=1, label="steering_output")
            nengo.Connection(input_node, ens, synapse=None)

            def decoded_function(x: np.ndarray) -> list[float]:
                e = x[0]
                de = x[1]
                m = x[2]
                if residual_mode:
                    # Small bounded residual-like correction.
                    return [-args.nengo_residual_gain * de]
                # Full PD-like steering map.
                return [-args.pid_kp * e - args.pid_kd * de - args.pid_slope_gain * m]

            nengo.Connection(ens, output_node, function=decoded_function, synapse=args.nengo_synapse)
            self.output_probe = nengo.Probe(output_node, synapse=None)

        self.nengo = nengo
        self.model = model
        self.sim = nengo.Simulator(model, dt=dt, progress_bar=False)

    def step(self, state: LaneState) -> ControlOutput:
        self.state[:] = state.vector()
        self.sim.step()
        y = 0.0 if len(self.sim.data[self.output_probe]) == 0 else float(self.sim.data[self.output_probe][-1][0])
        if self.residual_mode:
            y = float(np.clip(y, -self.args.nengo_residual_limit, self.args.nengo_residual_limit))
            return ControlOutput(raw_steering=y, residual_steering=y, info={"nengo_residual": y})
        return ControlOutput(raw_steering=y, info={"nengo_output": y})

    def close(self) -> None:
        try:
            self.sim.close()
        except Exception:
            pass


class AdaptiveHyperLIFGateController(ControllerBase):
    """
    Adaptive HyperLIF-inspired gate-state controller.

    demand = |e| + 0.25|de| + 0.75|m|
    g[k] = alpha_g*g[k-1] + beta_g*demand
    base = -Kp*e - Kd*de - Ks*m
    gated = (1 + Kg*g)*base
    a[k] = alpha_a*a[k-1] + beta_a*delta[k-1]
    delta = gated - Ka*a
    """

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
        info["family_residual"] = residual
        return ControlOutput(raw_steering=residual, residual_steering=residual, info=info)


class HybridResidualController(ControllerBase):
    """
    V5 gated hybrid residual controller.

    Baseline:
        delta_base = PID / SMC / NMPC

    Raw residual:
        r_raw = AdaptiveHyperLIF(state) or NengoLIF(state)

    Gated residual:
        r = gamma * r_raw

    Final:
        delta = delta_base + r

    The gate is high when the yellow-lane perception is reliable and the lane
    error/derivative/slope are inside a normal tracking region. It shrinks the
    residual near lane loss or highly transient recovery, so the baseline keeps
    authority when perception is uncertain.
    """

    def __init__(self, base: ControllerBase, residual: ControllerBase, args: argparse.Namespace, name: str):
        self.base = base
        self.residual = residual
        self.args = args
        self.name = name

    def _residual_gate(self, state: LaneState) -> tuple[float, dict[str, float]]:
        if not getattr(self.args, "residual_gating", True):
            return 1.0, {"gate_enabled": 0.0, "gate_error": 1.0, "gate_derivative": 1.0, "gate_slope": 1.0, "gate_pixel": 1.0}

        if not bool(state.lane_valid):
            return 0.0, {"gate_enabled": 1.0, "gate_error": 0.0, "gate_derivative": 0.0, "gate_slope": 0.0, "gate_pixel": 0.0}

        de = float(np.clip(state.d_lane_error, -self.args.derivative_limit, self.args.derivative_limit))
        ge = math.exp(-float(self.args.residual_gate_error_gain) * abs(float(state.lane_error)))
        gd = math.exp(-float(self.args.residual_gate_derivative_gain) * abs(de))
        gs = math.exp(-float(self.args.residual_gate_slope_gain) * abs(float(state.slope_xy)))

        # Pixel confidence rises smoothly from weak yellow evidence to normal lane evidence.
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
            "residual_raw": residual_raw,
            "residual_gate": float(gate),
            "residual": residual,
            "base_info": base_out.info or {},
            "residual_info": residual_out.info or {},
            "gate_info": gate_info,
        }
        return ControlOutput(raw_steering=raw, residual_steering=residual, info=info)

    def close(self) -> None:
        self.base.close()
        self.residual.close()


def build_controller(args: argparse.Namespace, dt: float) -> ControllerBase:
    if args.controller == "pid":
        return PIDLaneController(args)
    if args.controller == "smc":
        return SMCLaneController(args)
    if args.controller == "nmpc_image":
        return ImageNMPCController(args)
    if args.controller == "nengo_lif_pd":
        return NengoLIFController(args, dt=dt, residual_mode=False)
    if args.controller == "full_nengo_lif":
        return NengoLIFController(args, dt=dt, residual_mode=False)
    if args.controller == "adaptive_hyperlif_gate":
        return AdaptiveHyperLIFGateController(args, residual_mode=False)
    if args.controller in ("pid_adaptive_hyperlif_residual", "pid_gated_adaptive_hyperlif_residual"):
        return HybridResidualController(
            PIDLaneController(args),
            AdaptiveHyperLIFGateController(args, residual_mode=True),
            args,
            name=args.controller,
        )
    if args.controller in ("smc_adaptive_hyperlif_residual", "smc_gated_adaptive_hyperlif_residual"):
        return HybridResidualController(
            SMCLaneController(args),
            AdaptiveHyperLIFGateController(args, residual_mode=True),
            args,
            name=args.controller,
        )
    if args.controller in ("nmpc_adaptive_hyperlif_residual", "nmpc_gated_adaptive_hyperlif_residual"):
        return HybridResidualController(
            ImageNMPCController(args),
            AdaptiveHyperLIFGateController(args, residual_mode=True),
            args,
            name=args.controller,
        )
    if args.controller in ("pid_nengo_lif_residual", "pid_gated_nengo_lif_residual"):
        return HybridResidualController(
            PIDLaneController(args),
            NengoLIFController(args, dt=dt, residual_mode=True),
            args,
            name=args.controller,
        )
    if args.controller in ("smc_nengo_lif_residual", "smc_gated_nengo_lif_residual"):
        return HybridResidualController(
            SMCLaneController(args),
            NengoLIFController(args, dt=dt, residual_mode=True),
            args,
            name=args.controller,
        )
    if args.controller in ("nmpc_nengo_lif_residual", "nmpc_gated_nengo_lif_residual"):
        return HybridResidualController(
            ImageNMPCController(args),
            NengoLIFController(args, dt=dt, residual_mode=True),
            args,
            name=args.controller,
        )
    raise ValueError(f"Unsupported controller: {args.controller}")


# =============================================================================
# Throttle and post-processing
# =============================================================================


def compute_adaptive_throttle(
    *,
    lane_valid: bool,
    lane_error: float,
    d_lane_error: float,
    slope_xy: float,
    command_steering: float,
    previous_throttle: float,
    args: argparse.Namespace,
) -> float:
    if not args.adaptive_speed:
        return float(args.base_throttle * math.cos(command_steering))

    if not lane_valid or math.isnan(float(lane_error)):
        target = float(args.lost_lane_throttle)
    else:
        normalized_error = max(abs(float(lane_error)) - args.speed_error_soft_zone, 0.0)
        speed_score = math.exp(
            -args.speed_error_gain * normalized_error
            -args.speed_steer_gain * abs(float(command_steering))
            -args.speed_slope_gain * abs(float(slope_xy))
            -args.speed_derivative_gain * abs(float(d_lane_error))
        )
        target = args.min_throttle + (args.max_throttle - args.min_throttle) * speed_score

    target = float(np.clip(target, args.lost_lane_throttle, args.max_throttle))

    # V6: asymmetric throttle filtering. We still allow high top speed, but when
    # the lane state suddenly demands lower speed, the command decelerates much
    # faster than it accelerates. This fixes the V5 failure mode: high speed was
    # maintained too long into turns, so all controllers struggled near curves.
    if target < float(previous_throttle):
        alpha = float(np.clip(getattr(args, "speed_decel_alpha", args.speed_filter_alpha), 0.0, 0.999))
    else:
        alpha = float(np.clip(getattr(args, "speed_accel_alpha", args.speed_filter_alpha), 0.0, 0.999))

    filtered = alpha * float(previous_throttle) + (1.0 - alpha) * target
    return float(np.clip(filtered, args.lost_lane_throttle, args.max_throttle))



def compute_turn_demand(
    *,
    lane_error: float,
    slope_xy: float,
    args: argparse.Namespace,
) -> float:
    """
    V6 turn-demand index in [0, 1].

    It is high when the visual lane state indicates a curve or large lateral
    correction. The goal is not to reduce the available max steering/speed, but
    to change how quickly the post-processing lets the controller use that
    authority.
    """
    if not getattr(args, "turn_aware_stabilizer", True):
        return 0.0
    e_ref = max(float(getattr(args, "turn_error_reference", 0.12)), 1e-6)
    m_ref = max(float(getattr(args, "turn_slope_reference", 0.12)), 1e-6)
    turn = 0.5 * abs(float(lane_error)) / e_ref + 0.5 * abs(float(slope_xy)) / m_ref
    return float(np.clip(turn, 0.0, 1.0))


def apply_high_speed_stabilizer(
    *,
    raw_steering: float,
    previous_steering: float,
    previous_throttle: float,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    if not args.high_speed_stabilizer:
        return float(raw_steering), 1.0, float(raw_steering - previous_steering)

    throttle_mag = abs(float(previous_throttle))

    # These optional attributes are set by run_lane_mode immediately before the
    # stabilizer is called. This avoids changing the public function signature.
    turn_demand = float(np.clip(getattr(args, "_turn_demand", 0.0), 0.0, 1.0))

    # V6: in turns, relieve part of the speed-based steering attenuation and
    # allow faster steering-rate changes. This keeps straight-line behavior
    # smooth while preserving enough authority for curves.
    steering_relief = float(np.clip(getattr(args, "turn_steering_relief", 0.55), 0.0, 0.95))
    effective_speed_gain = float(args.speed_steering_gain) * (1.0 - steering_relief * turn_demand)
    speed_scale = 1.0 / (1.0 + effective_speed_gain * throttle_mag)
    scaled = float(raw_steering) * speed_scale

    rate_boost = max(float(getattr(args, "turn_rate_boost", 0.85)), 0.0)
    max_delta = max(float(args.steering_rate_limit) * (1.0 + rate_boost * turn_demand), 1e-6)
    delta = float(np.clip(scaled - float(previous_steering), -max_delta, max_delta))
    stabilized = float(previous_steering + delta)
    return stabilized, float(speed_scale), float(delta)


# =============================================================================
# Dashboard helpers
# =============================================================================


def as_bgr(frame: np.ndarray) -> np.ndarray:
    import cv2

    if frame is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    return np.zeros((240, 320, 3), dtype=np.uint8)



def resize_panel(frame: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
    import cv2

    panel = as_bgr(frame)
    panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel



def make_status_panel(*, width: int, height: int, lines: list[str], title: str = "status") -> np.ndarray:
    import cv2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y = 55
    for line in lines[:13]:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        y += 27
    return panel



def make_dashboard(
    *,
    overlay: np.ndarray,
    binary: np.ndarray,
    crop: np.ndarray,
    width: int,
    height: int,
    status_lines: list[str],
    footer_text: str,
) -> np.ndarray:
    import cv2

    panel_w = max(width // 2, 160)
    panel_h = max(height // 2, 120)
    p1 = resize_panel(overlay, panel_w, panel_h, "camera overlay")
    p2 = resize_panel(binary, panel_w, panel_h, "yellow lane mask")
    p3 = resize_panel(crop, panel_w, panel_h, "lane ROI crop")
    p4 = make_status_panel(width=panel_w, height=panel_h, lines=status_lines, title="status")
    dashboard = np.vstack((np.hstack((p1, p2)), np.hstack((p3, p4))))
    cv2.rectangle(dashboard, (0, dashboard.shape[0] - 30), (dashboard.shape[1], dashboard.shape[0]), (0, 0, 0), -1)
    cv2.putText(dashboard, footer_text, (8, dashboard.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return dashboard



def add_fit_overlay_x_from_y(
    overlay: np.ndarray,
    *,
    slope_xy: float,
    intercept_xy: float,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> np.ndarray:
    import cv2

    if math.isnan(slope_xy) or math.isnan(intercept_xy):
        return overlay

    crop_height = row_end - row_start
    crop_width = col_end - col_start
    y0, y1 = 0, crop_height - 1
    x0 = int(np.clip(slope_xy * y0 + intercept_xy, 0, crop_width - 1))
    x1 = int(np.clip(slope_xy * y1 + intercept_xy, 0, crop_width - 1))
    p0 = (col_start + x0, row_start + y0)
    p1 = (col_start + x1, row_start + y1)

    cv2.line(overlay, p0, p1, (0, 0, 255), 8)
    cv2.circle(overlay, p0, 9, (0, 0, 255), -1)
    cv2.circle(overlay, p1, 9, (0, 255, 0), -1)
    return overlay


# =============================================================================
# Manual mode
# =============================================================================


def run_manual_mode(args: argparse.Namespace) -> None:
    from pal.products.qcar import QCar
    from pal.utilities.keyboard import KeyboardDrive, QKeyboard

    sample_time = 1.0 / args.sample_rate
    stop_requested = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("\nStop requested. Sending zero command and closing manual drive...")

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    keyboard = None
    keyboard_drive = None
    car = None
    try:
        keyboard = QKeyboard()
        keyboard_drive = KeyboardDrive(
            mode=0,
            maxThrottle=args.manual_max_throttle,
            maxSteer=args.manual_max_steer,
        )
        car = QCar(readMode=1, frequency=args.sample_rate)
        start_time = time.time()
        print("QCar2 manual positioning mode")
        print("Hold SPACE to arm commands. Use WASD to drive. Ctrl+C exits safely.")

        while not stop_requested and time.time() - start_time < args.duration:
            loop_start = time.time()
            keyboard.update()
            if keyboard.states[keyboard.K_SPACE]:
                steering, throttle = keyboard_drive.update(keyboard)
            else:
                throttle = 0.0
                steering = 0.0

            throttle = float(np.clip(throttle, -args.manual_max_throttle, args.manual_max_throttle))
            steering = float(np.clip(steering, -args.manual_max_steer, args.manual_max_steer))
            car.read_write_std(throttle=throttle, steering=steering, LEDs=default_leds(throttle, steering))

            print(f"throttle={throttle:+.3f} | steering={steering:+.3f}", end="\r", flush=True)
            sleep_time = sample_time - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        print("\nCommanding QCar2 to stop...")
        safe_stop_car(car)
        safe_terminate(keyboard)
        safe_terminate(car)
        print("Manual mode finished safely.")


# =============================================================================
# Lane-following mode
# =============================================================================


def run_lane_mode(args: argparse.Namespace) -> None:
    import cv2
    from hal.utilities.image_processing import ImageProcessing
    from pal.products.qcar import QCar, QCarCameras
    from pal.utilities.math import Filter

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"{args.log_prefix}_{args.controller}_{timestamp}.csv"

    sample_time = 1.0 / args.sample_rate
    controller = build_controller(args, dt=sample_time)

    print("Starting unified QCar2 lane-following base.")
    print(f"Controller: {args.controller}")
    print(f"Sample time: {sample_time:.6f} s")
    print(f"Log: {log_path}")
    print("Press Ctrl+C or ESC to stop safely.")

    steering_filter = Filter().low_pass_first_order_variable(args.filter_cutoff, sample_time)
    next(steering_filter)

    lower_yellow = np.array([args.hsv_low_h, args.hsv_low_s, args.hsv_low_v], dtype=np.uint8)
    upper_yellow = np.array([args.hsv_high_h, args.hsv_high_s, args.hsv_high_v], dtype=np.uint8)
    leds = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.uint8)

    qcar_cameras = None
    qcar = None
    stop_requested = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("\nStop requested. Sending zero command and closing lane mode...")

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    try:
        qcar_cameras = QCarCameras(
            frameWidth=args.image_width,
            frameHeight=args.image_height,
            frameRate=args.sample_rate,
            enableFront=True,
        )
        qcar = QCar(readMode=1, frequency=args.sample_rate)

        fieldnames = [
            "time_s",
            "controller",
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
            "raw_steering",
            "residual_steering",
            "filtered_steering",
            "command_steering",
            "command_throttle",
            "steering_speed_scale",
            "steering_rate_delta",
            "loop_dt",
            "state_json",
            "controller_info_json",
        ]

        with log_path.open("w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()

            start_time = time.time()
            previous_good_steering = 0.0
            previous_good_time = -math.inf
            previous_lane_error: Optional[float] = None
            previous_lane_error_time: Optional[float] = None
            previous_command_throttle = float(args.base_throttle)
            loop_dt = sample_time

            while not stop_requested and time.time() - start_time < args.duration:
                loop_start = time.time()
                elapsed = loop_start - start_time

                qcar_cameras.readAll()
                front = qcar_cameras.csiFront.imageData
                crop = front[
                    args.roi_row_start : args.roi_row_end,
                    args.roi_col_start : args.roi_col_end,
                ].copy()

                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                binary = ImageProcessing.binary_thresholding(
                    frame=hsv,
                    lowerBounds=lower_yellow,
                    upperBounds=upper_yellow,
                )
                binary_float = binary / 255.0

                slope_xy, intercept_xy, yellow_pixel_fraction, lane_valid, lane_valid_reason = fit_lane_x_from_y(
                    binary_float,
                    min_fit_pixels=args.min_fit_pixels,
                )

                if lane_valid and yellow_pixel_fraction < args.min_lane_pixels:
                    lane_valid = False
                    lane_valid_reason = "too_few_yellow_pixels"
                elif lane_valid and yellow_pixel_fraction > args.max_lane_pixels:
                    lane_valid = False
                    lane_valid_reason = "too_many_yellow_pixels"

                crop_height = args.roi_row_end - args.roi_row_start
                crop_width = args.roi_col_end - args.roi_col_start

                lane_x = math.nan
                desired_x = args.desired_lane_x_fraction * crop_width
                lane_error = math.nan
                d_lane_error = 0.0
                raw_steering = math.nan
                residual_steering = 0.0
                filtered_steering = math.nan
                steering_speed_scale = 1.0
                steering_rate_delta = 0.0
                controller_info: dict[str, Any] = {}

                if lane_valid:
                    lane_x, desired_x, lane_error = compute_lane_state_from_fit(
                        slope_xy=slope_xy,
                        intercept_xy=intercept_xy,
                        crop_width=crop_width,
                        crop_height=crop_height,
                        lookahead_fraction=args.lookahead_fraction,
                        desired_lane_x_fraction=args.desired_lane_x_fraction,
                    )

                    if previous_lane_error is None or previous_lane_error_time is None:
                        d_lane_error = 0.0
                    else:
                        dt_error = max(elapsed - previous_lane_error_time, 1e-6)
                        d_lane_error = (lane_error - previous_lane_error) / dt_error

                    previous_lane_error = lane_error
                    previous_lane_error_time = elapsed

                    state = LaneState(
                        lane_error=float(lane_error),
                        d_lane_error=float(d_lane_error),
                        slope_xy=float(slope_xy),
                        previous_steering=float(previous_good_steering),
                        loop_dt=float(loop_dt),
                        lane_valid=True,
                        yellow_pixel_fraction=float(yellow_pixel_fraction),
                        lane_valid_reason=str(lane_valid_reason),
                    )

                    control = controller.step(state)
                    raw_steering = float(control.raw_steering)
                    residual_steering = float(control.residual_steering)
                    controller_info = control.info or {}

                    clipped = float(np.clip(raw_steering, -args.max_steering, args.max_steering))
                    filtered_steering = float(steering_filter.send((clipped, loop_dt)))
                    pre_stabilized = float(np.clip(filtered_steering, -args.max_steering, args.max_steering))

                    args._turn_demand = compute_turn_demand(
                        lane_error=float(lane_error),
                        slope_xy=float(slope_xy),
                        args=args,
                    )

                    command_steering, steering_speed_scale, steering_rate_delta = apply_high_speed_stabilizer(
                        raw_steering=pre_stabilized,
                        previous_steering=previous_good_steering,
                        previous_throttle=previous_command_throttle,
                        args=args,
                    )
                    controller_info["turn_demand"] = float(getattr(args, "_turn_demand", 0.0))
                    command_steering = float(np.clip(command_steering, -args.max_steering, args.max_steering))

                    command_throttle = compute_adaptive_throttle(
                        lane_valid=True,
                        lane_error=lane_error,
                        d_lane_error=d_lane_error,
                        slope_xy=slope_xy,
                        command_steering=command_steering,
                        previous_throttle=previous_command_throttle,
                        args=args,
                    )
                    previous_command_throttle = command_throttle
                    previous_good_steering = command_steering
                    previous_good_time = elapsed
                    control_mode = "lane_fit"
                else:
                    state = LaneState(
                        lane_error=0.0,
                        d_lane_error=0.0,
                        slope_xy=0.0,
                        previous_steering=float(previous_good_steering),
                        loop_dt=float(loop_dt),
                        lane_valid=False,
                    )
                    time_since_good = elapsed - previous_good_time
                    if time_since_good <= args.hold_last_valid:
                        command_steering = float(
                            np.clip(
                                previous_good_steering * args.hold_decay,
                                -args.max_steering,
                                args.max_steering,
                            )
                        )
                        control_mode = "hold_previous"
                    else:
                        command_steering = 0.0
                        control_mode = "lost_lane"

                    args._turn_demand = 0.0

                    command_throttle = compute_adaptive_throttle(
                        lane_valid=False,
                        lane_error=math.nan,
                        d_lane_error=0.0,
                        slope_xy=0.0,
                        command_steering=command_steering,
                        previous_throttle=previous_command_throttle,
                        args=args,
                    )
                    previous_command_throttle = command_throttle

                qcar.read_write_std(command_throttle, command_steering, leds)

                if not args.no_display:
                    overlay = front.copy()
                    roi = overlay[
                        args.roi_row_start : args.roi_row_end,
                        args.roi_col_start : args.roi_col_end,
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
                    status_color = (0, 255, 0) if lane_valid else (0, 0, 255)
                    cv2.putText(
                        overlay,
                        f"{args.controller} | {lane_valid_reason} | {control_mode}",
                        (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        status_color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay,
                        f"e={float(lane_error):+.3f} de={float(d_lane_error):+.3f} steer={command_steering:+.3f} thr={command_throttle:+.3f}",
                        (40, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        status_color,
                        2,
                        cv2.LINE_AA,
                    )
                    overlay = add_fit_overlay_x_from_y(
                        overlay.astype(np.uint8),
                        slope_xy=float(slope_xy) if lane_valid else math.nan,
                        intercept_xy=float(intercept_xy) if lane_valid else math.nan,
                        row_start=args.roi_row_start,
                        row_end=args.roi_row_end,
                        col_start=args.roi_col_start,
                        col_end=args.roi_col_end,
                    )

                    status_lines = [
                        f"controller: {args.controller}",
                        f"mode: {control_mode}",
                        f"lane valid: {bool(lane_valid)}",
                        f"reason: {lane_valid_reason}",
                        f"e: {float(lane_error):+.4f}" if not math.isnan(float(lane_error)) else "e: nan",
                        f"de: {float(d_lane_error):+.4f}",
                        f"slope: {float(slope_xy):+.4f}" if not math.isnan(float(slope_xy)) else "slope: nan",
                        f"raw: {float(raw_steering):+.4f}" if not math.isnan(float(raw_steering)) else "raw: nan",
                        f"resid: {float(residual_steering):+.4f}",
                        f"cmd steer: {command_steering:+.4f}",
                        f"cmd thr: {command_throttle:+.4f}",
                        f"speed scale: {steering_speed_scale:+.4f}",
                        f"yellow pix: {yellow_pixel_fraction:.6f}",
                    ]
                    dashboard = make_dashboard(
                        overlay=overlay,
                        binary=binary,
                        crop=crop,
                        width=args.dashboard_width,
                        height=args.dashboard_height,
                        status_lines=status_lines,
                        footer_text=f"{args.controller} | {control_mode} | e={float(lane_error):+.3f} | steer={command_steering:+.3f} | thr={command_throttle:+.3f}",
                    )
                    cv2.imshow("QCar2 unified lane control", dashboard)

                    if args.show_separate_windows:
                        cv2.imshow("QCar2 lane overlay", cv2.resize(overlay, (820, 410)))
                        if args.show_binary_lane:
                            cv2.imshow("Yellow lane mask", cv2.resize(binary, (820, 220)))
                        if args.show_lane_crop:
                            cv2.imshow("Lane ROI crop", cv2.resize(crop, (820, 220)))

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        raise KeyboardInterrupt

                writer.writerow(
                    {
                        "time_s": elapsed,
                        "controller": args.controller,
                        "control_mode": control_mode,
                        "lane_valid": int(lane_valid),
                        "lane_valid_reason": lane_valid_reason,
                        "yellow_pixel_fraction": yellow_pixel_fraction,
                        "lane_slope_xy": float(slope_xy),
                        "lane_intercept_xy": float(intercept_xy),
                        "lane_x": float(lane_x),
                        "desired_x": float(desired_x),
                        "lane_error": float(lane_error),
                        "d_lane_error": float(d_lane_error),
                        "raw_steering": float(raw_steering),
                        "residual_steering": float(residual_steering),
                        "filtered_steering": float(filtered_steering),
                        "command_steering": float(command_steering),
                        "command_throttle": float(command_throttle),
                        "steering_speed_scale": float(steering_speed_scale),
                        "steering_rate_delta": float(steering_rate_delta),
                        "loop_dt": float(loop_dt),
                        "state_json": json.dumps(asdict(state)),
                        "controller_info_json": json.dumps(controller_info, default=str),
                    }
                )

                loop_dt = max(time.time() - loop_start, sample_time)
                sleep_time = sample_time - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping safely.")
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
        print(f"Lane mode finished safely. Log saved to: {log_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    configure_paths()
    select_virtual_qcar(args.qcar_type)

    if args.mode == "manual":
        run_manual_mode(args)
    elif args.mode == "lane":
        run_lane_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
