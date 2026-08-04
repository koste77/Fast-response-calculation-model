from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from scipy.stats import qmc
import torch


torch.set_default_dtype(torch.float64)

SCRIPT_DIR = Path(__file__).resolve().parent
BP_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

DISPLAY_MODEL_NAME = "BE-PFNO"
RESPONSE_KEYS = ("u", "phi", "M", "Q")
RESPONSE_LABELS = {
    "u": "Deflection u",
    "phi": "Rotation phi",
    "M": "Moment M",
    "Q": "Shear Q",
}


@dataclass(frozen=True)
class SceneConfig:
    scene_id: str
    scene_label: str
    boundary_condition: str
    bc_type: str
    load_kind: str
    load_or_excitation: str
    parameter_range: str
    input_form: str
    ansatz: str
    relative_dir: Path
    training_script: str
    model_file: str
    logs_file: str
    lhs_dim: int

    @property
    def scene_dir(self) -> Path:
        return BP_ROOT / self.relative_dir

    @property
    def training_path(self) -> Path:
        return self.scene_dir / self.training_script

    @property
    def model_path(self) -> Path:
        return self.scene_dir / self.model_file

    @property
    def logs_path(self) -> Path:
        return self.scene_dir / self.logs_file


SCENES = [
    SceneConfig(
        scene_id="ss_linear",
        scene_label="Simply supported - linear load",
        boundary_condition="simply-supported",
        bc_type="simply-supported",
        load_kind="linear",
        load_or_excitation="Linear distributed load",
        parameter_range="q0,qL in [-0.3,-0.001]",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x(L-x)u_hat; M=x(L-x)M_hat",
        relative_dir=Path("simply supported/Simply supported Linear load"),
        training_script="BEP-PINO_Model_simply-supported_linear.py",
        model_file="BEP-PINO_Model_Linear_simply-supported.pth",
        logs_file="BEP-PINO_Logs_Linear_simply-supported.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="ss_quadratic",
        scene_label="Simply supported - quadratic load",
        boundary_condition="simply-supported",
        bc_type="simply-supported",
        load_kind="quadratic",
        load_or_excitation="Symmetric quadratic distributed load",
        parameter_range="qe in [-0.3,-0.001], dq in [0.01,0.15]",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x(L-x)u_hat; M=x(L-x)M_hat",
        relative_dir=Path("simply supported/Simply supported Quadratic load"),
        training_script="BEP-PINO_Model_simply-supported_quadratic.py",
        model_file="BEP-PINO_Model_Quadratic_2Param_simply-supported.pth",
        logs_file="BEP-PINO_Logs_Quadratic_2Param_simply-supported.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="ss_concentrated",
        scene_label="Simply supported - Gaussian local load",
        boundary_condition="simply-supported",
        bc_type="simply-supported",
        load_kind="concentrated",
        load_or_excitation="Gaussian local load",
        parameter_range="P in [-5.0,-0.5], xp in [0.2,0.8], sigma=0.04",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x(L-x)u_hat; M=x(L-x)M_hat",
        relative_dir=Path("simply supported/Simply supported Concentrated load"),
        training_script="BEP-PINO_Model_simply-supported_concentrated.py",
        model_file="BEP-PINO_Model_Concentrated_simply-supported.pth",
        logs_file="BEP-PINO_Logs_Concentrated_simply-supported.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="ff_linear",
        scene_label="Fixed-fixed - linear load",
        boundary_condition="fixed-fixed",
        bc_type="fixed-fixed",
        load_kind="linear",
        load_or_excitation="Linear distributed load",
        parameter_range="q0,qL in [-0.3,-0.001]",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x^2(L-x)^2u_hat; phi=x(L-x)phi_hat",
        relative_dir=Path("Fixed-fixed/Fixed-fixed linear load"),
        training_script="BEP-PINO_Model_fixed-fixed_linear.py",
        model_file="BEP-PINO_Model_Linear_fixed-fixed.pth",
        logs_file="BEP-PINO_Logs_Linear_fixed-fixed.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="ff_quadratic",
        scene_label="Fixed-fixed - quadratic load",
        boundary_condition="fixed-fixed",
        bc_type="fixed-fixed",
        load_kind="quadratic",
        load_or_excitation="Symmetric quadratic distributed load",
        parameter_range="qe in [-0.3,-0.001], dq in [0.01,0.15]",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x^2(L-x)^2u_hat; phi=x(L-x)phi_hat",
        relative_dir=Path("Fixed-fixed/Fixed-fixed quadratic load"),
        training_script="BEP-PINO_Model_fixed-fixed_quadratic.py",
        model_file="BEP-PINO_Model_Quadratic_2Param_fixed-fixed.pth",
        logs_file="BEP-PINO_Logs_Quadratic_2Param_fixed-fixed.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="ff_concentrated",
        scene_label="Fixed-fixed - Gaussian local load",
        boundary_condition="fixed-fixed",
        bc_type="fixed-fixed",
        load_kind="concentrated",
        load_or_excitation="Gaussian local load",
        parameter_range="P in [-5.0,-0.5], xp in [0.2,0.8], sigma=0.04",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x^2(L-x)^2u_hat; phi=x(L-x)phi_hat",
        relative_dir=Path("Fixed-fixed/Fixed-fixed Concentrated load"),
        training_script="BEP-PINO_Model_fixed-fixed_concentrated.py",
        model_file="BEP-PINO_Model_Concentrated_fixed-fixed.pth",
        logs_file="BEP-PINO_Logs_Concentrated_fixed-fixed.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="cf_linear",
        scene_label="Cantilever - linear load",
        boundary_condition="cantilever",
        bc_type="cantilever",
        load_kind="linear",
        load_or_excitation="Linear distributed load",
        parameter_range="q0,qL in [-0.3,-0.001]",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x^2u_hat; phi=x phi_hat; M=(L-x)M_hat; Q=(L-x)Q_hat",
        relative_dir=Path("Cantilever/Cantilever linear load"),
        training_script="BEP-PINO_Model_cantilever_linear.py",
        model_file="BEP-PINO_Model_Linear_cantilever.pth",
        logs_file="BEP-PINO_Logs_Linear_cantilever.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="cf_quadratic",
        scene_label="Cantilever - quadratic load",
        boundary_condition="cantilever",
        bc_type="cantilever",
        load_kind="quadratic",
        load_or_excitation="Symmetric quadratic distributed load",
        parameter_range="qe in [-0.3,-0.001], dq in [0.01,0.15]",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x^2u_hat; phi=x phi_hat; M=(L-x)M_hat; Q=(L-x)Q_hat",
        relative_dir=Path("Cantilever/Cantilever quadratic load"),
        training_script="BEP-PINO_Model_cantilever_quadratic.py",
        model_file="BEP-PINO_Model_Quadratic_2Param_cantilever.pth",
        logs_file="BEP-PINO_Logs_Quadratic_2Param_cantilever.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="cf_concentrated",
        scene_label="Cantilever - Gaussian local load",
        boundary_condition="cantilever",
        bc_type="cantilever",
        load_kind="concentrated",
        load_or_excitation="Gaussian local load",
        parameter_range="P in [-5.0,-0.5], xp in [0.2,0.8], sigma=0.04",
        input_form="51 sensors of q(x) + EI,L",
        ansatz="u=x^2u_hat; phi=x phi_hat; M=(L-x)M_hat; Q=(L-x)Q_hat",
        relative_dir=Path("Cantilever/Cantilever Concentrated load"),
        training_script="BEP-PINO_Model_cantilever_concentrated.py",
        model_file="BEP-PINO_Model_Concentrated_cantilever.pth",
        logs_file="BEP-PINO_Logs_Concentrated_cantilever.pth",
        lhs_dim=2,
    ),
    SceneConfig(
        scene_id="settlement",
        scene_label="Fixed-fixed - left settlement",
        boundary_condition="clamped-clamped-settlement",
        bc_type="clamped-clamped-settlement",
        load_kind="settlement",
        load_or_excitation="Left support settlement",
        parameter_range="s in [-0.1,0.0]",
        input_form="1 settlement value + EI,L",
        ansatz="xi=x/L; u=s(1-3xi^2+2xi^3)+x^2(L-x)^2u_hat; phi=s(-6xi+6xi^2)/L+x(L-x)phi_hat",
        relative_dir=Path("Fixed-fixed left settlement"),
        training_script="BEP-PINO_Model_fixed-fixed_settlement.py",
        model_file="BEP-PINO_Model_Settlement_fixed-fixed.pth",
        logs_file="BEP-PINO_Logs_Settlement_fixed-fixed.pth",
        lhs_dim=1,
    ),
    SceneConfig(
        scene_id="rotation",
        scene_label="Fixed-fixed - left rotation",
        boundary_condition="clamped-clamped-rotation",
        bc_type="clamped-clamped-rotation",
        load_kind="rotation",
        load_or_excitation="Left support rotation",
        parameter_range="theta in [-0.1,0.1]",
        input_form="1 rotation value + EI,L",
        ansatz="xi=x/L; u=theta L(xi-2xi^2+xi^3)+x^2(L-x)^2u_hat; phi=theta(1-4xi+3xi^2)+x(L-x)phi_hat",
        relative_dir=Path("Fixed-fixed left rotation"),
        training_script="BEP-PINO_Model_fixed-fixed_rotation.py",
        model_file="BEP-PINO_Model_Rotation_fixed-fixed.pth",
        logs_file="BEP-PINO_Logs_Rotation_fixed-fixed.pth",
        lhs_dim=1,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-boundary BE-PFNO LHS-100 generalization results."
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--n-domain", type=int, default=100)
    parser.add_argument("--num-nodes-ref", type=int, default=1000)
    parser.add_argument("--gauss-order", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timing-warmup", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_training_module(scene: SceneConfig) -> Any:
    spec = importlib.util.spec_from_file_location(f"{scene.scene_id}_training", scene.training_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training script: {scene.training_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_scene_parameters(scene: SceneConfig, sample_count: int, seed: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    lhs = qmc.LatinHypercube(d=scene.lhs_dim, seed=seed).random(sample_count)
    params: list[dict[str, float]] = []
    for row in lhs:
        if scene.load_kind == "linear":
            params.append({
                "q0": -0.3 + row[0] * (-0.001 + 0.3),
                "qL": -0.3 + row[1] * (-0.001 + 0.3),
            })
        elif scene.load_kind == "quadratic":
            params.append({
                "qe": -0.3 + row[0] * (-0.001 + 0.3),
                "dq": 0.01 + row[1] * (0.15 - 0.01),
            })
        elif scene.load_kind == "concentrated":
            params.append({
                "P": -5.0 + row[0] * (-0.5 + 5.0),
                "xp": 0.2 + row[1] * (0.8 - 0.2),
                "sigma": 0.04,
            })
        elif scene.load_kind == "settlement":
            params.append({"s": -0.1 + row[0] * 0.1})
        elif scene.load_kind == "rotation":
            params.append({"theta": -0.1 + row[0] * 0.2})
        else:
            raise ValueError(f"Unsupported load kind: {scene.load_kind}")
    return lhs, params


def representative_parameter_sets(scene: SceneConfig) -> list[dict[str, float]]:
    if scene.load_kind == "linear":
        return [
            {"q0": -0.15, "qL": -0.05},
            {"q0": -0.05, "qL": -0.20},
            {"q0": -0.15, "qL": -0.15},
        ]
    if scene.load_kind == "quadratic":
        return [
            {"qe": -0.20, "dq": 0.05},
            {"qe": -0.05, "dq": 0.10},
            {"qe": -0.12, "dq": 0.08},
        ]
    if scene.load_kind == "concentrated":
        return [
            {"P": -3.0, "xp": 0.3, "sigma": 0.04},
            {"P": -2.0, "xp": 0.5, "sigma": 0.04},
            {"P": -2.5, "xp": 0.7, "sigma": 0.04},
        ]
    if scene.load_kind == "settlement":
        return [
            {"s": -0.08},
            {"s": -0.05},
            {"s": -0.02},
        ]
    if scene.load_kind == "rotation":
        return [
            {"theta": -0.08},
            {"theta": 0.04},
            {"theta": 0.08},
        ]
    raise ValueError(f"Unsupported load kind: {scene.load_kind}")


def load_values(scene: SceneConfig, x: np.ndarray, params: dict[str, float], L: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if scene.load_kind == "linear":
        return params["q0"] + (params["qL"] - params["q0"]) * x / L
    if scene.load_kind == "quadratic":
        c0 = params["qe"]
        c2 = 4.0 * params["dq"] / L**2
        c1 = -4.0 * params["dq"] / L
        return c2 * x**2 + c1 * x + c0
    if scene.load_kind == "concentrated":
        return params["P"] * np.exp(-((x - params["xp"]) ** 2) / (2.0 * params["sigma"] ** 2))
    if scene.load_kind == "settlement":
        return np.full_like(x, params["s"], dtype=np.float64)
    if scene.load_kind == "rotation":
        return np.full_like(x, params["theta"], dtype=np.float64)
    raise ValueError(f"Unsupported load kind: {scene.load_kind}")


def q_domain_values(scene: SceneConfig, x: np.ndarray, params: dict[str, float], L: float) -> np.ndarray:
    if scene.load_kind in {"settlement", "rotation"}:
        return np.zeros_like(x, dtype=np.float64)
    return load_values(scene, x, params, L)


def parameter_label(scene: SceneConfig, params: dict[str, float]) -> str:
    if scene.load_kind == "linear":
        return f"q0={params['q0']:.4f}, qL={params['qL']:.4f}"
    if scene.load_kind == "quadratic":
        return f"qe={params['qe']:.4f}, dq={params['dq']:.4f}"
    if scene.load_kind == "concentrated":
        return f"P={params['P']:.4f}, xp={params['xp']:.4f}, sigma={params['sigma']:.2f}"
    if scene.load_kind == "settlement":
        return f"s={params['s']:.4f}"
    if scene.load_kind == "rotation":
        return f"theta={params['theta']:.4f}"
    return json.dumps(params, ensure_ascii=False)


def display_path(path: Path) -> str:
    return str(path).replace("BP-PFNO", DISPLAY_MODEL_NAME)


def build_model_inputs(
    scene: SceneConfig,
    params_list: list[dict[str, float]],
    m_sensors: int,
    n_domain: int,
    L_val: float,
    EI_val: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    x_sensors = np.linspace(0.0, L_val, m_sensors, dtype=np.float64)
    x_grid = np.linspace(0.0, L_val, n_domain, dtype=np.float64)
    sensors = np.zeros((len(params_list), m_sensors), dtype=np.float64)
    q_domain = np.zeros((len(params_list), n_domain), dtype=np.float64)

    for idx, params in enumerate(params_list):
        sensors[idx, :] = load_values(scene, x_sensors, params, L_val)
        q_domain[idx, :] = q_domain_values(scene, x_grid, params, L_val)

    branch_input = torch.cat(
        [
            torch.tensor(sensors, dtype=torch.float64),
            torch.tensor([[EI_val, L_val]], dtype=torch.float64).expand(len(params_list), -1),
        ],
        dim=1,
    ).to(device)
    x_batch = torch.tensor(x_grid, dtype=torch.float64).view(1, n_domain, 1).expand(len(params_list), -1, -1).to(device)
    q_tensor = torch.tensor(q_domain, dtype=torch.float64).unsqueeze(-1).to(device)
    return branch_input, x_batch, q_tensor, x_sensors, x_grid


def hermite_shape_functions(s: np.ndarray, Le: float) -> np.ndarray:
    r = s / Le
    return np.column_stack([
        1.0 - 3.0 * r**2 + 2.0 * r**3,
        s * (1.0 - r) ** 2,
        3.0 * r**2 - 2.0 * r**3,
        s * (r**2 - r),
    ])


def beam_element_stiffness(EI: float, Le: float) -> np.ndarray:
    return (EI / Le**3) * np.array([
        [12.0, 6.0 * Le, -12.0, 6.0 * Le],
        [6.0 * Le, 4.0 * Le**2, -6.0 * Le, 2.0 * Le**2],
        [-12.0, -6.0 * Le, 12.0, -6.0 * Le],
        [6.0 * Le, 2.0 * Le**2, -6.0 * Le, 4.0 * Le**2],
    ], dtype=np.float64)


class HermiteBeamSolver:
    def __init__(self, bc_type: str, L: float, EI: float, num_nodes: int, gauss_order: int):
        if num_nodes < 3:
            raise ValueError("num_nodes must be at least 3")
        self.bc_type = bc_type
        self.L = L
        self.EI = EI
        self.num_nodes = num_nodes
        self.n_elem = num_nodes - 1
        self.total_dof = 2 * num_nodes
        self.x_ref = np.linspace(0.0, L, num_nodes, dtype=np.float64)
        self.Le = L / self.n_elem
        self.ke = beam_element_stiffness(EI, self.Le)
        self.gauss_xi, self.gauss_w = np.polynomial.legendre.leggauss(gauss_order)
        self.gauss_s = 0.5 * self.Le * (self.gauss_xi + 1.0)
        self.gauss_N = hermite_shape_functions(self.gauss_s, self.Le)

        K = lil_matrix((self.total_dof, self.total_dof), dtype=np.float64)
        for e in range(self.n_elem):
            dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
            for a in range(4):
                for b in range(4):
                    K[dof[a], dof[b]] += self.ke[a, b]
        self.K = K.tocsc()
        self.fixed_dofs = self._fixed_dofs()
        self.free_dofs = np.setdiff1d(np.arange(self.total_dof), self.fixed_dofs)
        self.K_free_fixed = self.K[self.free_dofs, :][:, self.fixed_dofs]
        self.solve_free = factorized(self.K[self.free_dofs, :][:, self.free_dofs].tocsc())

    def _fixed_dofs(self) -> np.ndarray:
        if self.bc_type == "simply-supported":
            return np.array([0, self.total_dof - 2])
        if self.bc_type == "cantilever":
            return np.array([0, 1])
        if self.bc_type in {"fixed-fixed", "clamped-clamped-settlement", "clamped-clamped-rotation"}:
            return np.array([0, 1, self.total_dof - 2, self.total_dof - 1])
        raise ValueError(f"Unsupported boundary type: {self.bc_type}")

    def _fixed_values(self, params: dict[str, float]) -> np.ndarray:
        if self.bc_type == "simply-supported":
            return np.array([0.0, 0.0], dtype=np.float64)
        if self.bc_type == "cantilever":
            return np.array([0.0, 0.0], dtype=np.float64)
        if self.bc_type == "fixed-fixed":
            return np.zeros(4, dtype=np.float64)
        if self.bc_type == "clamped-clamped-settlement":
            return np.array([params["s"], 0.0, 0.0, 0.0], dtype=np.float64)
        if self.bc_type == "clamped-clamped-rotation":
            return np.array([0.0, params["theta"], 0.0, 0.0], dtype=np.float64)
        raise ValueError(f"Unsupported boundary type: {self.bc_type}")

    def _consistent_element_load(
        self,
        x_left: float,
        q_func: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        x_g = x_left + self.gauss_s
        q_g = np.asarray(q_func(x_g), dtype=np.float64)
        weights = self.gauss_w * q_g * (self.Le / 2.0)
        return weights @ self.gauss_N

    def solve_on_grid(
        self,
        q_func: Callable[[np.ndarray], np.ndarray],
        x_target: np.ndarray,
        params: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        F = np.zeros(self.total_dof, dtype=np.float64)
        element_loads = np.zeros((self.n_elem, 4), dtype=np.float64)

        for e in range(self.n_elem):
            x_left = self.x_ref[e]
            fe = self._consistent_element_load(x_left, q_func)
            element_loads[e, :] = fe
            dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
            F[dof] += fe

        fixed_vals = self._fixed_values(params)
        U = np.zeros(self.total_dof, dtype=np.float64)
        U[self.fixed_dofs] = fixed_vals
        rhs = F[self.free_dofs] - self.K_free_fixed @ fixed_vals
        U[self.free_dofs] = self.solve_free(rhs)

        u_ref = U[0::2]
        phi_ref = U[1::2]
        M_sum = np.zeros(self.num_nodes, dtype=np.float64)
        Q_sum = np.zeros(self.num_nodes, dtype=np.float64)
        count = np.zeros(self.num_nodes, dtype=np.float64)

        for e in range(self.n_elem):
            dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
            p_e = self.ke @ U[dof] - element_loads[e]
            Q_left, M_left = p_e[0], -p_e[1]
            Q_right, M_right = -p_e[2], p_e[3]
            Q_sum[e] += Q_left
            M_sum[e] += M_left
            count[e] += 1.0
            Q_sum[e + 1] += Q_right
            M_sum[e + 1] += M_right
            count[e + 1] += 1.0

        M_ref = M_sum / np.maximum(count, 1.0)
        Q_ref = Q_sum / np.maximum(count, 1.0)
        x_target = np.asarray(x_target, dtype=np.float64).reshape(-1)
        return (
            np.interp(x_target, self.x_ref, u_ref),
            np.interp(x_target, self.x_ref, phi_ref),
            np.interp(x_target, self.x_ref, M_ref),
            np.interp(x_target, self.x_ref, Q_ref),
        )


def make_q_func(scene: SceneConfig, params: dict[str, float], L: float) -> Callable[[np.ndarray], np.ndarray]:
    def q_func(x: np.ndarray) -> np.ndarray:
        return q_domain_values(scene, x, params, L)

    return q_func


def prediction_metrics(pred: np.ndarray, fem: np.ndarray, x_grid: np.ndarray) -> dict[str, float]:
    abs_error = np.abs(pred - fem)
    idx = int(np.argmax(abs_error))
    ss_res = float(np.sum((fem - pred) ** 2))
    ss_tot = float(np.sum((fem - np.mean(fem)) ** 2) + 1e-12)
    return {
        "mae": float(np.mean(abs_error)),
        "rel_l2": float(np.linalg.norm(pred - fem) / (np.linalg.norm(fem) + 1e-12)),
        "r2": float(1.0 - ss_res / ss_tot),
        "max_abs_error": float(abs_error[idx]),
        "x_at_max_error": float(x_grid[idx]),
        "pred_at_max_error": float(pred[idx]),
        "fem_at_max_error": float(fem[idx]),
    }


def classify_error_region(scene: SceneConfig, params: dict[str, float], x_loc: float, L: float) -> str:
    tol = 0.06 * L
    if scene.load_kind == "concentrated" and abs(x_loc - params["xp"]) <= tol:
        return "near load center"
    if abs(x_loc) <= tol:
        return "near left boundary"
    if abs(x_loc - L) <= tol:
        if scene.bc_type == "cantilever":
            return "near free end"
        return "near right boundary"
    return "interior"


def boundary_violation(
    scene: SceneConfig,
    fields: dict[str, np.ndarray],
    params: dict[str, float],
) -> dict[str, float]:
    u = fields["u"]
    phi = fields["phi"]
    M = fields["M"]
    Q = fields["Q"]
    checks: dict[str, float]
    if scene.bc_type == "simply-supported":
        checks = {
            "abs_u_0": abs(float(u[0])),
            "abs_u_L": abs(float(u[-1])),
            "abs_M_0": abs(float(M[0])),
            "abs_M_L": abs(float(M[-1])),
        }
    elif scene.bc_type == "fixed-fixed":
        checks = {
            "abs_u_0": abs(float(u[0])),
            "abs_phi_0": abs(float(phi[0])),
            "abs_u_L": abs(float(u[-1])),
            "abs_phi_L": abs(float(phi[-1])),
        }
    elif scene.bc_type == "cantilever":
        checks = {
            "abs_u_0": abs(float(u[0])),
            "abs_phi_0": abs(float(phi[0])),
            "abs_M_L": abs(float(M[-1])),
            "abs_Q_L": abs(float(Q[-1])),
        }
    elif scene.bc_type == "clamped-clamped-settlement":
        checks = {
            "abs_u0_minus_s": abs(float(u[0] - params["s"])),
            "abs_phi_0": abs(float(phi[0])),
            "abs_u_L": abs(float(u[-1])),
            "abs_phi_L": abs(float(phi[-1])),
        }
    elif scene.bc_type == "clamped-clamped-rotation":
        checks = {
            "abs_u_0": abs(float(u[0])),
            "abs_phi0_minus_theta": abs(float(phi[0] - params["theta"])),
            "abs_u_L": abs(float(u[-1])),
            "abs_phi_L": abs(float(phi[-1])),
        }
    else:
        raise ValueError(f"Unsupported boundary type: {scene.bc_type}")
    checks["max_boundary_violation"] = max(checks.values())
    return checks


def last_log_value(logs: dict[str, Any], key: str) -> float:
    values = logs.get(key, [])
    if values is None or len(values) == 0:
        return math.nan
    return float(np.asarray(values, dtype=float)[-1])


def measure_inference_time(
    model: torch.nn.Module,
    branch_input: torch.Tensor,
    x_batch: torch.Tensor,
    L_val: float,
    bc_type: str,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.predict_with_ansatz(branch_input, x_batch, L=L_val, bc_type=bc_type)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict_with_ansatz(branch_input, x_batch, L=L_val, bc_type=bc_type)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / (repeats * branch_input.shape[0]))


def build_representative_payloads(
    scene: SceneConfig,
    model: torch.nn.Module,
    solver: HermiteBeamSolver,
    m_sensors: int,
    n_domain: int,
    L_val: float,
    EI_val: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    params_list = representative_parameter_sets(scene)
    branch_input, x_batch, _, _, x_grid = build_model_inputs(
        scene, params_list, m_sensors, n_domain, L_val, EI_val, device
    )
    with torch.no_grad():
        pred_tensors = model.predict_with_ansatz(branch_input, x_batch, L=L_val, bc_type=scene.bc_type)
    pred_arrays = {
        key: pred_tensors[idx].detach().cpu().numpy().squeeze(-1)
        for idx, key in enumerate(RESPONSE_KEYS)
    }
    payloads: list[dict[str, Any]] = []
    for case_index, params in enumerate(params_list, start=1):
        fem_tuple = solver.solve_on_grid(make_q_func(scene, params, L_val), x_grid, params)
        fem_fields = {key: fem_tuple[idx] for idx, key in enumerate(RESPONSE_KEYS)}
        payloads.append({
            "model_label": DISPLAY_MODEL_NAME,
            "scene_id": scene.scene_id,
            "scene_label": scene.scene_label,
            "boundary_condition": scene.boundary_condition,
            "load_or_excitation": scene.load_or_excitation,
            "representative_case_index": case_index,
            "parameter_label": parameter_label(scene, params),
            "parameters": params,
            "x": x_grid.tolist(),
            "input": load_values(scene, x_grid, params, L_val).tolist(),
            "pred": {
                key: np.asarray(pred_arrays[key][case_index - 1], dtype=np.float64).tolist()
                for key in RESPONSE_KEYS
            },
            "fem": {key: np.asarray(fem_fields[key], dtype=np.float64).tolist() for key in RESPONSE_KEYS},
        })
    return payloads


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: float) -> float | str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    representative_curves: list[dict[str, Any]] = []

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    for scene_index, scene in enumerate(SCENES):
        print(f"[{scene_index + 1:02d}/{len(SCENES)}] Evaluating {scene.scene_label} ...")
        for required_path in (scene.training_path, scene.model_path, scene.logs_path):
            if not required_path.exists():
                raise FileNotFoundError(required_path)

        module = load_training_module(scene)
        model_data = torch.load(scene.model_path, weights_only=False, map_location=device)
        logs_data = torch.load(scene.logs_path, weights_only=False, map_location="cpu")
        L_val = float(model_data["L_val"])
        EI_val = float(model_data["EI_val"])
        m_sensors = int(model_data["m_sensors"])

        model = module.FNO_Combined(
            m_sensors=m_sensors,
            modes=int(model_data["modes"]),
            width=int(model_data["width"]),
            hidden_dim=int(model_data.get("hidden_dim", 128)),
        ).to(device)
        model.load_state_dict(model_data["model_state_dict"])
        model.eval()

        _lhs, params_list = sample_scene_parameters(scene, args.samples, args.seed + scene_index)
        branch_input, x_batch, q_tensor, x_sensors, x_grid = build_model_inputs(
            scene, params_list, m_sensors, args.n_domain, L_val, EI_val, device
        )

        with torch.no_grad():
            pred_tensors = model.predict_with_ansatz(branch_input, x_batch, L=L_val, bc_type=scene.bc_type)
        preds = {
            key: pred_tensors[idx].detach().cpu().numpy().squeeze(-1)
            for idx, key in enumerate(RESPONSE_KEYS)
        }

        physics_loss, loss_geo, loss_const, loss_eq1, loss_eq2 = module.compute_physics_loss_combined(
            model, branch_input, x_batch, q_tensor, EI_val, L_val, scene.bc_type
        )
        inference_ms = measure_inference_time(
            model,
            branch_input,
            x_batch,
            L_val,
            scene.bc_type,
            device,
            args.timing_warmup,
            args.timing_repeats,
        )

        solver = HermiteBeamSolver(scene.bc_type, L_val, EI_val, args.num_nodes_ref, args.gauss_order)
        boundary_values: list[float] = []
        rel_l2_by_response: dict[str, list[float]] = {key: [] for key in RESPONSE_KEYS}

        for sample_idx, params in enumerate(params_list):
            fem_tuple = solver.solve_on_grid(make_q_func(scene, params, L_val), x_grid, params)
            fem_fields = {key: fem_tuple[idx] for idx, key in enumerate(RESPONSE_KEYS)}
            pred_fields = {key: preds[key][sample_idx] for key in RESPONSE_KEYS}
            bc_checks = boundary_violation(scene, pred_fields, params)
            boundary_values.append(float(bc_checks["max_boundary_violation"]))

            for key in RESPONSE_KEYS:
                metrics = prediction_metrics(pred_fields[key], fem_fields[key], x_grid)
                rel_l2_by_response[key].append(metrics["rel_l2"])
                detail_rows.append({
                    "model_label": DISPLAY_MODEL_NAME,
                    "scene_id": scene.scene_id,
                    "scene_label": scene.scene_label,
                    "boundary_condition": scene.boundary_condition,
                    "load_or_excitation": scene.load_or_excitation,
                    "load_kind": scene.load_kind,
                    "sample_index": sample_idx,
                    "parameter_label": parameter_label(scene, params),
                    "response": key,
                    "response_label": RESPONSE_LABELS[key],
                    "mae": metrics["mae"],
                    "rel_l2": metrics["rel_l2"],
                    "r2": metrics["r2"],
                    "max_abs_error": metrics["max_abs_error"],
                    "x_at_max_error": metrics["x_at_max_error"],
                    "error_region": classify_error_region(scene, params, metrics["x_at_max_error"], L_val),
                    "pred_at_max_error": metrics["pred_at_max_error"],
                    "fem_at_max_error": metrics["fem_at_max_error"],
                    "max_boundary_violation": bc_checks["max_boundary_violation"],
                })

        all_rel = np.concatenate([np.asarray(rel_l2_by_response[key], dtype=np.float64) for key in RESPONSE_KEYS])
        summary_row: dict[str, Any] = {
            "model_label": DISPLAY_MODEL_NAME,
            "scene_id": scene.scene_id,
            "scene_label": scene.scene_label,
            "boundary_condition": scene.boundary_condition,
            "load_or_excitation": scene.load_or_excitation,
            "load_kind": scene.load_kind,
            "sample_count": args.samples,
            "mean_rel_l2_all": float(np.mean(all_rel)),
            "std_rel_l2_all": float(np.std(all_rel, ddof=1)),
            "max_rel_l2_all": float(np.max(all_rel)),
            "mean_boundary_violation": float(np.mean(boundary_values)),
            "max_boundary_violation": float(np.max(boundary_values)),
        }
        for key in RESPONSE_KEYS:
            values = np.asarray(rel_l2_by_response[key], dtype=np.float64)
            summary_row[f"{key}_mean_rel_l2"] = float(np.mean(values))
            summary_row[f"{key}_std_rel_l2"] = float(np.std(values, ddof=1))
            summary_row[f"{key}_max_rel_l2"] = float(np.max(values))
        summary_rows.append(summary_row)

        physics_rows.append({
            "model_label": DISPLAY_MODEL_NAME,
            "scene_id": scene.scene_id,
            "scene_label": scene.scene_label,
            "boundary_condition": scene.boundary_condition,
            "load_or_excitation": scene.load_or_excitation,
            "sample_count": args.samples,
            "residual_geo_norm": float(loss_geo.detach().cpu().item()),
            "residual_const_norm": float(loss_const.detach().cpu().item()),
            "residual_eq1_norm": float(loss_eq1.detach().cpu().item()),
            "residual_eq2_norm": float(loss_eq2.detach().cpu().item()),
            "total_physics_loss_norm": float(physics_loss.detach().cpu().item()),
            "mean_boundary_violation": float(np.mean(boundary_values)),
            "max_boundary_violation": float(np.max(boundary_values)),
            "inference_ms_per_sample": inference_ms,
            "device": str(device),
            "adam_final_loss": finite_float(last_log_value(logs_data, "loss_history")),
            "lbfgs_final_loss": finite_float(last_log_value(logs_data, "lbfgs_loss_history")),
        })

        setup_rows.append({
            "model_label": DISPLAY_MODEL_NAME,
            "scene_id": scene.scene_id,
            "scene_label": scene.scene_label,
            "boundary_condition": scene.boundary_condition,
            "load_or_excitation": scene.load_or_excitation,
            "parameter_range": scene.parameter_range,
            "input_form": scene.input_form,
            "lhs_test_samples": args.samples,
            "boundary_ansatz": scene.ansatz,
            "model_file_path": display_path(scene.model_path),
            "logs_file_path": display_path(scene.logs_path),
            "m_sensors": m_sensors,
            "L_val": L_val,
            "EI_val": EI_val,
        })

        representative_curves.extend(build_representative_payloads(
            scene, model, solver, m_sensors, args.n_domain, L_val, EI_val, device
        ))

    write_csv(
        output_dir / "section4_scene_setup.csv",
        setup_rows,
        [
            "model_label", "scene_id", "scene_label", "boundary_condition", "load_or_excitation",
            "parameter_range", "input_form", "lhs_test_samples", "boundary_ansatz",
            "model_file_path", "logs_file_path", "m_sensors", "L_val", "EI_val",
        ],
    )
    write_csv(
        output_dir / "section4_lhs100_detail_metrics.csv",
        detail_rows,
        [
            "model_label", "scene_id", "scene_label", "boundary_condition", "load_or_excitation",
            "load_kind", "sample_index", "parameter_label", "response", "response_label",
            "mae", "rel_l2", "r2", "max_abs_error", "x_at_max_error", "error_region",
            "pred_at_max_error", "fem_at_max_error", "max_boundary_violation",
        ],
    )
    write_csv(
        output_dir / "section4_lhs100_summary.csv",
        summary_rows,
        [
            "model_label", "scene_id", "scene_label", "boundary_condition", "load_or_excitation",
            "load_kind", "sample_count", "mean_rel_l2_all", "std_rel_l2_all", "max_rel_l2_all",
            "u_mean_rel_l2", "u_std_rel_l2", "u_max_rel_l2",
            "phi_mean_rel_l2", "phi_std_rel_l2", "phi_max_rel_l2",
            "M_mean_rel_l2", "M_std_rel_l2", "M_max_rel_l2",
            "Q_mean_rel_l2", "Q_std_rel_l2", "Q_max_rel_l2",
            "mean_boundary_violation", "max_boundary_violation",
        ],
    )
    write_csv(
        output_dir / "section4_physics_efficiency.csv",
        physics_rows,
        [
            "model_label", "scene_id", "scene_label", "boundary_condition", "load_or_excitation",
            "sample_count", "residual_geo_norm", "residual_const_norm", "residual_eq1_norm",
            "residual_eq2_norm", "total_physics_loss_norm", "mean_boundary_violation",
            "max_boundary_violation", "inference_ms_per_sample", "device",
            "adam_final_loss", "lbfgs_final_loss",
        ],
    )

    metadata = {
        "model_label": DISPLAY_MODEL_NAME,
        "method_note": (
            "Each boundary-load problem uses its own trained parametric operator. "
            "The LHS-100 test validates framework applicability and in-domain generalization, "
            "not direct cross-boundary prediction with one shared weight set."
        ),
        "samples_per_scene": args.samples,
        "seed": args.seed,
        "n_domain": args.n_domain,
        "num_nodes_ref": args.num_nodes_ref,
        "gauss_order": args.gauss_order,
        "device": str(device),
        "scene_count": len(SCENES),
        "response_order": RESPONSE_KEYS,
    }
    (output_dir / "section4_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "section4_tables.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "scene_setup": setup_rows,
                "lhs100_summary": summary_rows,
                "physics_efficiency": physics_rows,
                "detail_metrics": detail_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "section4_representative_curves.json").write_text(
        json.dumps(representative_curves, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"Finished {len(SCENES)} scenes x {args.samples} LHS samples.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
