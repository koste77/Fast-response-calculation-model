from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARISON_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "comparison_multiscenario_fem_comparison"

RESPONSE_KEYS = ("u", "phi", "M", "Q")
RESPONSE_TITLES = {
    "u": "Deflection $u$",
    "phi": "Rotation $\\varphi$",
    "M": "Moment $M$",
    "Q": "Shear $Q$",
}
CASE_COLORS = ("#4583B6", "#B02425", "#218D42")
TEST_LOADS = ((-3.0, 0.3), (-2.0, 0.5), (-2.5, 0.7))


@dataclass(frozen=True)
class TorchModelConfig:
    slug: str
    label: str
    folder: str
    module_file: str
    model_file: str
    log_file: str
    class_name: str


@dataclass(frozen=True)
class PinnModelConfig:
    slug: str
    label: str
    folder: str
    module_file: str
    ckpt_prefix: str
    log_file: str


@dataclass(frozen=True)
class SingleCasePinnModelConfig:
    slug: str
    label: str
    folder: str
    module_file: str
    ckpt_prefix_base: str


TORCH_MODEL_CONFIGS = (
    TorchModelConfig(
        slug="bep_pfno",
        label="BEP-PFNO",
        folder="BEP-PFNO",
        module_file="BEP-PFNO_Model_fixed-fixed_concentrated.py",
        model_file="BEP-PFNO_Model_fixed-fixed concentrated.pth",
        log_file="BEP-PFNO_Logs_fixed-fixed concentrated.pth",
        class_name="BEP_PFNO_Combined",
    ),
    TorchModelConfig(
        slug="me_bep_pino",
        label="ME-BEP-PINO",
        folder="ME-BEP-PINO",
        module_file="ME-BEP-PINO_Parallel_Concentrated_fixed-fixed.py",
        model_file="ME-BEP-PINO_Parallel_Model_fixed-fixed_Concentrated.pth",
        log_file="ME-BEP-PINO_Parallel_Logs_fixed-fixed_Concentrated.pth",
        class_name="ME_BEP_PINO_Parallel",
    ),
    TorchModelConfig(
        slug="ho_bep_pino",
        label="HO-BEP-PINO",
        folder="HO-BEP-PINO",
        module_file="HO-BEP-PINO_Model_fixed-fixed_concentrated.py",
        model_file="HO-BEP-PINO_Model_fixed-fixed_Concentrated.pth",
        log_file="HO-BEP-PINO_Logs_fixed-fixed_Concentrated.pth",
        class_name="HO_BEP_PINO_Combined",
    ),
    TorchModelConfig(
        slug="bep_pdon",
        label="BEP-PDON",
        folder="BEP-PDON",
        module_file="BEP-PDON Fixed-fixed concentrated.PY",
        model_file="BEP-PDON_Model_Fixed-fixed concentrated.pth",
        log_file="BEP-PDON_Fixed-fixed concentrated.pth",
        class_name="BEP_PDON_Combined",
    ),
)

SINGLE_CASE_PINN_CONFIG = SingleCasePinnModelConfig(
    slug="sc_ml_pinn",
    label="SC-ml-PINN",
    folder="SC-ml-PINN",
    module_file="SC-ml-PINN adam LBFGS EVA.py",
    ckpt_prefix_base="SC-ml-PINN_Combined",
)

PINN_MODEL_CONFIG = PinnModelConfig(
    slug="p_ml_pinn",
    label="P-ml-PINN",
    folder="P-ml-PINN",
    module_file="P-ml-PINN adam LBFGS EVA.py",
    ckpt_prefix="P-ml-PINN adam LBFGS",
    log_file="P-ml-PINN adam LBFGS_logs.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FEM comparison figures for fixed-fixed comparison models "
            "with the same visual style as BP-PFNO section4_multiscenario_fem_comparison."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--num-fem-nodes", type=int, default=1000)
    parser.add_argument("--gauss-order", type=int, default=8)
    parser.add_argument("--skip-pinn", action="store_true", help="Skip the TensorFlow PINN models.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 500,
        "savefig.dpi": 500,
        "font.size": 10,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def import_module_from_path(path: Path, module_name: str, definitions_only: bool = False) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        loader = importlib.machinery.SourceFileLoader(module_name, str(path))
        spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if definitions_only:
        source = path.read_text(encoding="utf-8", errors="ignore")
        main_marker = "if __name__ == '__main__':"
        marker_index = source.find(main_marker)
        if marker_index >= 0:
            source = source[:marker_index]
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)
        return module
    spec.loader.exec_module(module)
    return module


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_log_summary(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {"log_path": str(path), "status": "missing"}
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)

        def last_npz(key: str) -> float:
            if key not in data:
                return float("nan")
            arr = np.asarray(data[key], dtype=float).reshape(-1)
            return float(arr[-1]) if arr.size else float("nan")

        return {
            "log_path": str(path),
            "status": "loaded",
            "adam_final_loss": last_npz("loss_log"),
            "lbfgs_final_loss": last_npz("loss_lbfgs_log"),
            "physics_final_loss": last_npz("loss_f_log"),
        }

    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)

    def last_torch(key: str) -> float:
        values = data.get(key, [])
        if values is None or len(values) == 0:
            return float("nan")
        return float(np.asarray(values, dtype=float).reshape(-1)[-1])

    return {
        "log_path": str(path),
        "status": "loaded",
        "adam_final_loss": last_torch("loss_history"),
        "lbfgs_final_loss": last_torch("lbfgs_loss_history"),
        "raw_final_loss": last_torch("raw_total_loss_history"),
    }


def gaussian_localized_load(x: np.ndarray | float, P: float, xp: float, sigma: float) -> np.ndarray | float:
    return P * np.exp(-((x - xp) ** 2) / (2.0 * sigma ** 2))


def hermite_shape_functions(s: float, Le: float) -> np.ndarray:
    r = s / Le
    return np.array([
        1.0 - 3.0 * r ** 2 + 2.0 * r ** 3,
        s * (1.0 - r) ** 2,
        3.0 * r ** 2 - 2.0 * r ** 3,
        s * (r ** 2 - r),
    ])


def beam_element_stiffness(EI: float, Le: float) -> np.ndarray:
    return (EI / Le ** 3) * np.array([
        [12.0,       6.0 * Le,   -12.0,       6.0 * Le],
        [6.0 * Le,   4.0 * Le**2, -6.0 * Le,   2.0 * Le**2],
        [-12.0,     -6.0 * Le,    12.0,      -6.0 * Le],
        [6.0 * Le,   2.0 * Le**2, -6.0 * Le,   4.0 * Le**2],
    ])


def consistent_element_load_gaussian(
    x_left: float,
    Le: float,
    P: float,
    xp: float,
    sigma: float,
    gauss_order: int,
) -> np.ndarray:
    xi, wi = np.polynomial.legendre.leggauss(gauss_order)
    fe = np.zeros(4, dtype=np.float64)
    for xi_i, wi_i in zip(xi, wi):
        s = 0.5 * Le * (xi_i + 1.0)
        x_g = x_left + s
        q_g = gaussian_localized_load(x_g, P, xp, sigma)
        fe += wi_i * hermite_shape_functions(s, Le) * q_g * (Le / 2.0)
    return fe


def solve_beam_fem_fixed_gaussian(
    P: float,
    xp: float,
    sigma: float = 0.04,
    L: float = 1.0,
    EI: float = 1.0,
    num_nodes: int = 1000,
    gauss_order: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    n_elem = num_nodes - 1
    total_dof = num_nodes * 2
    x_ref = np.linspace(0.0, L, num_nodes)
    Le = L / n_elem
    ke = beam_element_stiffness(EI, Le)

    K = np.zeros((total_dof, total_dof), dtype=np.float64)
    F = np.zeros(total_dof, dtype=np.float64)
    element_loads: list[np.ndarray] = []

    for e in range(n_elem):
        x_left = x_ref[e]
        fe = consistent_element_load_gaussian(x_left, Le, P, xp, sigma, gauss_order)
        element_loads.append(fe)
        dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        K[np.ix_(dof, dof)] += ke
        F[dof] += fe

    fixed_dofs = np.array([0, 1, total_dof - 2, total_dof - 1])
    free_dofs = np.setdiff1d(np.arange(total_dof), fixed_dofs)

    U = np.zeros(total_dof, dtype=np.float64)
    U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])

    u_ref = U[0::2]
    phi_ref = U[1::2]
    M_sum = np.zeros(num_nodes, dtype=np.float64)
    Q_sum = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)

    for e in range(n_elem):
        dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        p_e = ke @ U[dof] - element_loads[e]

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
    return x_ref, u_ref, phi_ref, M_ref, Q_ref


def solve_beam_fem_fixed_gaussian_on_grid(
    P: float,
    xp: float,
    x_target: np.ndarray,
    sigma: float,
    L: float,
    EI: float,
    num_nodes_ref: int,
    gauss_order: int,
) -> dict[str, np.ndarray]:
    x_ref, u_ref, phi_ref, M_ref, Q_ref = solve_beam_fem_fixed_gaussian(
        P=P,
        xp=xp,
        sigma=sigma,
        L=L,
        EI=EI,
        num_nodes=num_nodes_ref,
        gauss_order=gauss_order,
    )
    x_target = np.asarray(x_target, dtype=np.float64).reshape(-1)
    return {
        "u": np.interp(x_target, x_ref, u_ref),
        "phi": np.interp(x_target, x_ref, phi_ref),
        "M": np.interp(x_target, x_ref, M_ref),
        "Q": np.interp(x_target, x_ref, Q_ref),
    }


def build_branch_input(P: float, xp: float, sigma: float, L: float, EI: float, m_sensors: int, device: str) -> Any:
    import torch

    x_sensors = np.linspace(0.0, L, m_sensors)
    q_sensor_data = gaussian_localized_load(x_sensors, P=P, xp=xp, sigma=sigma)
    return torch.cat([
        torch.tensor([q_sensor_data], dtype=torch.float64, device=device),
        torch.tensor([[EI, L]], dtype=torch.float64, device=device),
    ], dim=1)


def make_torch_predictor(config: TorchModelConfig, device: str) -> tuple[Callable[..., dict[str, np.ndarray]], dict[str, Any]]:
    import torch

    model_dir = COMPARISON_ROOT / config.folder
    module = import_module_from_path(
        model_dir / config.module_file,
        f"comparison_{config.slug}_module",
        definitions_only=True,
    )
    model_data = torch.load(model_dir / config.model_file, map_location=device, weights_only=False)

    model_class = getattr(module, config.class_name)
    init_kwargs = {
        "m_sensors": int(model_data["m_sensors"]),
        "modes": int(model_data.get("modes", 16)),
        "width": int(model_data.get("width", 64)),
        "hidden_dim": int(model_data.get("hidden_dim", 128)),
    }
    if config.slug == "be_pdon":
        init_kwargs = {
            "m_sensors": int(model_data["m_sensors"]),
            "hidden_dim": int(model_data.get("hidden_dim", 128)),
            "num_layers": int(model_data.get("num_layers", 4)),
        }

    model = model_class(**init_kwargs).to(device)
    state_dict = {k.replace("module.", ""): v for k, v in model_data["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{config.label} weight load mismatch. missing keys={missing}; unexpected keys={unexpected}"
        )
    model.eval()

    L_val = float(model_data["L_val"])
    EI_val = float(model_data["EI_val"])
    m_sensors = int(model_data["m_sensors"])

    def predict(P: float, xp: float, x_grid: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
        branch_input = build_branch_input(P, xp, sigma, L_val, EI_val, m_sensors, device)
        x_tensor = torch.tensor(x_grid.reshape(1, -1, 1), dtype=torch.float64, device=device)
        with torch.no_grad():
            outputs = model.predict_with_ansatz(branch_input, x_tensor, L=L_val, bc_type="fixed-fixed")
        return {
            response: values.detach().cpu().numpy().reshape(-1)
            for response, values in zip(RESPONSE_KEYS, outputs)
        }

    metadata = {
        "model_file": str(model_dir / config.model_file),
        "module_file": str(model_dir / config.module_file),
        "log_summary": load_log_summary(model_dir / config.log_file),
        "L_val": L_val,
        "EI_val": EI_val,
        "m_sensors": m_sensors,
        "sigma_val": 0.04,
    }
    return predict, metadata


def single_case_pinn_prefix(config: SingleCasePinnModelConfig, P: float, xp: float) -> str:
    return f"{config.ckpt_prefix_base}_P{P}_xp{xp}"


def make_single_case_pinn_predictor(
    config: SingleCasePinnModelConfig,
) -> tuple[Callable[..., dict[str, np.ndarray]], dict[str, Any]]:
    model_dir = COMPARISON_ROOT / config.folder
    module = import_module_from_path(
        model_dir / config.module_file,
        "comparison_sc_ml_pinn_module",
        definitions_only=True,
    )

    L_val = 1.0
    EI_val = 1.0
    sigma_val = 0.04
    layers = [[1] + 3 * [10] + [1] for _ in range(4)]
    X_fake = np.linspace(0.0, L_val, 2).reshape(-1, 1).astype(np.float32)
    q_fake = np.zeros_like(X_fake)

    models: dict[tuple[float, float], Any] = {}
    case_files: list[dict[str, Any]] = []
    for P, xp in TEST_LOADS:
        prefix = single_case_pinn_prefix(config, P, xp)
        log_path = model_dir / f"{prefix}_logs.npz"
        ckpt_path = model_dir / f"{prefix}.ckpt"
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        if not ckpt_path.with_suffix(".ckpt.index").exists():
            raise FileNotFoundError(ckpt_path.with_suffix(".ckpt.index"))

        logs = np.load(log_path, allow_pickle=True)
        X_train_current = logs["X_train"].astype(np.float32)

        module.tf.reset_default_graph()
        model = module.PINN_model(
            layers,
            X_train_current,
            q_fake,
            X_fake,
            X_fake,
            X_fake,
            X_fake,
            X_fake,
            X_fake,
        )
        model.saver.restore(model.sess, str(ckpt_path))
        models[(float(P), float(xp))] = model
        case_files.append({
            "P": P,
            "xp": xp,
            "model_file": str(ckpt_path),
            "log_summary": load_log_summary(log_path),
        })

    def predict(P: float, xp: float, x_grid: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
        del sigma
        key = (float(P), float(xp))
        if key not in models:
            raise KeyError(f"No SC-ml-PINN checkpoint loaded for P={P}, xp={xp}.")
        x_star = np.asarray(x_grid, dtype=np.float32).reshape(-1, 1)
        model = models[key]
        return {
            "u": model.predict_u(x_star).reshape(-1),
            "phi": model.predict_a(x_star).reshape(-1),
            # Keep the visual sign convention used by the saved SC-ml-PINN EVA plotting code.
            "M": -model.predict_M(x_star).reshape(-1),
            "Q": -model.predict_Q(x_star).reshape(-1),
        }

    metadata = {
        "model_file": "three single-case TensorFlow checkpoints",
        "module_file": str(model_dir / config.module_file),
        "case_files": case_files,
        "L_val": L_val,
        "EI_val": EI_val,
        "sigma_val": sigma_val,
    }
    return predict, metadata


def make_pinn_predictor(config: PinnModelConfig) -> tuple[Callable[..., dict[str, np.ndarray]], dict[str, Any]]:
    model_dir = COMPARISON_ROOT / config.folder
    module = import_module_from_path(
        model_dir / config.module_file,
        "comparison_p_ml_pinn_module",
        definitions_only=True,
    )
    logs = np.load(model_dir / config.log_file, allow_pickle=True)

    sigma_val = float(logs["sigma_val"])
    L_val = float(logs["L_val"])
    EI_val = float(logs["EI_val"])
    P_min_val = float(logs["P_min_val"])
    P_max_val = float(logs["P_max_val"])
    xp_min_val = float(logs["xp_min_val"])
    xp_max_val = float(logs["xp_max_val"])
    layers = logs["layers"].tolist()

    module.tf.reset_default_graph()
    X_fake = np.zeros((2, 3), dtype=np.float32)
    q_fake = np.zeros((2, 1), dtype=np.float32)
    model = module.PINN_model(
        layers,
        X_fake,
        q_fake,
        X_fake,
        q_fake,
        X_fake,
        q_fake,
        X_fake,
        q_fake,
        EI=EI_val,
        L=L_val,
        P_min=P_min_val,
        P_max=P_max_val,
        xp_min=xp_min_val,
        xp_max=xp_max_val,
    )
    model.saver.restore(model.sess, str(model_dir / f"{config.ckpt_prefix}.ckpt"))

    def predict(P: float, xp: float, x_grid: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
        del sigma
        x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)
        return {
            "u": model.predict_u(x_grid, P, xp).reshape(-1),
            "phi": model.predict_a(x_grid, P, xp).reshape(-1),
            # Keep the visual sign convention used by the saved P-ml-PINN EVA plotting code.
            "M": -model.predict_M(x_grid, P, xp).reshape(-1),
            "Q": -model.predict_Q(x_grid, P, xp).reshape(-1),
        }

    metadata = {
        "model_file": str(model_dir / f"{config.ckpt_prefix}.ckpt"),
        "module_file": str(model_dir / config.module_file),
        "log_summary": load_log_summary(model_dir / config.log_file),
        "L_val": L_val,
        "EI_val": EI_val,
        "sigma_val": sigma_val,
        "P_range": [P_min_val, P_max_val],
        "xp_range": [xp_min_val, xp_max_val],
    }
    return predict, metadata


def rel_l2(pred: np.ndarray, fem: np.ndarray) -> float:
    return float(np.linalg.norm(pred - fem) / (np.linalg.norm(fem) + 1e-12))


def generate_curves(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []

    x_grid = np.linspace(0.0, 1.0, args.num_points)
    model_entries: list[tuple[str, str, Callable[..., dict[str, np.ndarray]], dict[str, Any]]] = []

    for config in TORCH_MODEL_CONFIGS:
        predict, metadata = make_torch_predictor(config, args.device)
        model_entries.append((config.slug, config.label, predict, metadata))

    if not args.skip_pinn:
        predict, metadata = make_single_case_pinn_predictor(SINGLE_CASE_PINN_CONFIG)
        model_entries.append((SINGLE_CASE_PINN_CONFIG.slug, SINGLE_CASE_PINN_CONFIG.label, predict, metadata))
        predict, metadata = make_pinn_predictor(PINN_MODEL_CONFIG)
        model_entries.append((PINN_MODEL_CONFIG.slug, PINN_MODEL_CONFIG.label, predict, metadata))

    fem_cache: dict[tuple[float, float, float, float, float], dict[str, np.ndarray]] = {}
    for slug, label, predict, metadata in model_entries:
        L_val = to_float(metadata.get("L_val"), 1.0)
        EI_val = to_float(metadata.get("EI_val"), 1.0)
        sigma_val = to_float(metadata.get("sigma_val"), 0.04)
        x_grid = np.linspace(0.0, L_val, args.num_points)

        model_manifest.append({
            "slug": slug,
            "label": label,
            **metadata,
        })

        for case_index, (P, xp) in enumerate(TEST_LOADS, start=1):
            pred = predict(P, xp, x_grid, sigma_val)
            fem_key = (float(P), float(xp), sigma_val, L_val, EI_val)
            if fem_key not in fem_cache:
                fem_cache[fem_key] = solve_beam_fem_fixed_gaussian_on_grid(
                    P=P,
                    xp=xp,
                    x_target=x_grid,
                    sigma=sigma_val,
                    L=L_val,
                    EI=EI_val,
                    num_nodes_ref=args.num_fem_nodes,
                    gauss_order=args.gauss_order,
                )
            fem = fem_cache[fem_key]

            curves.append({
                "model_slug": slug,
                "model_label": label,
                "scene_id": slug,
                "scene_label": label,
                "boundary_condition": "fixed-fixed",
                "load_or_excitation": "Gaussian local load",
                "representative_case_index": case_index,
                "parameter_label": f"P={P:.1f}, xp={xp:.1f}, sigma={sigma_val:.2f}",
                "x": x_grid.tolist(),
                "pred": {key: np.asarray(pred[key], dtype=float).reshape(-1).tolist() for key in RESPONSE_KEYS},
                "fem": {key: np.asarray(fem[key], dtype=float).reshape(-1).tolist() for key in RESPONSE_KEYS},
            })

            for response in RESPONSE_KEYS:
                metrics.append({
                    "model_slug": slug,
                    "model_label": label,
                    "case_index": case_index,
                    "P": P,
                    "xp": xp,
                    "response": response,
                    "rel_l2": rel_l2(np.asarray(pred[response]), np.asarray(fem[response])),
                })

    return curves, metrics, model_manifest


def group_curves_by_model(curves: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in curves:
        slug = item["model_slug"]
        if slug not in grouped:
            grouped[slug] = []
            order.append(slug)
        grouped[slug].append(item)
    for slug in order:
        grouped[slug].sort(key=lambda item: int(item.get("representative_case_index", 1)))
    return [grouped[slug] for slug in order]


def apply_readable_ylim(ax: plt.Axes, *arrays: np.ndarray, include_zero: bool = False) -> None:
    values = np.concatenate([np.ravel(array) for array in arrays])
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.concatenate([values, np.array([0.0])])
    if values.size == 0:
        return
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin
    abs_scale = float(np.max(np.abs(values)))
    min_span = max(0.12 * abs_scale, 1e-6)
    if span < min_span:
        center = 0.5 * (ymin + ymax)
        half_span = 0.5 * min_span
        ax.set_ylim(center - half_span, center + half_span)


def draw_beam_boundary_axis(ax: plt.Axes, boundary_condition: str, color: str = "black") -> None:
    beam_left = 0.0
    beam_right = 1.0
    x_margin = 0.05
    ax.hlines(0.0, beam_left, beam_right, color=color, linewidth=2.4, zorder=8)
    if boundary_condition == "simply-supported":
        ax.plot([beam_left, beam_right], [0.0, 0.0], linestyle="", marker="^",
                markersize=7.0, color=color, zorder=20, clip_on=False)
    elif boundary_condition == "cantilever":
        ax.plot([beam_left], [0.0], linestyle="", marker="s",
                markersize=6.5, color=color, zorder=20, clip_on=False)
    else:
        ax.plot([beam_left, beam_right], [0.0, 0.0], linestyle="", marker="s",
                markersize=6.5, color=color, zorder=20, clip_on=False)
    ax.set_xlim(beam_left - x_margin, beam_right + x_margin)
    ax.set_xticks(np.linspace(beam_left, beam_right, 6))


def plot_multimodel_comparison(curves: list[dict[str, Any]], output_dir: Path) -> Path:
    model_groups = group_curves_by_model(curves)
    if not model_groups:
        raise ValueError("No representative curves are available for plotting.")

    fig, axes = plt.subplots(
        len(model_groups),
        len(RESPONSE_KEYS),
        figsize=(3.2 * len(RESPONSE_KEYS), 1.45 * len(model_groups)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    for row, model_group in enumerate(model_groups):
        scene_item = model_group[0]
        for col, response in enumerate(RESPONSE_KEYS):
            ax = axes[row, col]
            ylim_series: list[np.ndarray] = []
            for case_pos, item in enumerate(model_group):
                x = np.asarray(item["x"], dtype=float)
                fem = np.asarray(item["fem"][response], dtype=float)
                pred = np.asarray(item["pred"][response], dtype=float)
                color = CASE_COLORS[case_pos % len(CASE_COLORS)]
                case_index = int(item.get("representative_case_index", case_pos + 1))
                show_legend_label = row == 0 and col == 0
                ax.plot(
                    x, fem, color=color, alpha=0.35, linewidth=3.2,
                    label=f"Case {case_index} FEM" if show_legend_label else None,
                )
                ax.plot(
                    x, pred, color=color, linewidth=1.3, linestyle="--",
                    label=f"Case {case_index} Prediction" if show_legend_label else None,
                )
                ylim_series.extend([fem, pred])
            apply_readable_ylim(ax, *ylim_series, include_zero=True)
            draw_beam_boundary_axis(ax, scene_item["boundary_condition"])
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
            if col == 0:
                ax.set_ylabel(scene_item["scene_label"], fontsize=14)
            if row == 0:
                ax.set_title(RESPONSE_TITLES[response], fontsize=14)
            if row == len(model_groups) - 1:
                ax.set_xlabel("Beam Position ($x$)", fontsize=14)
            if response == "M":
                ax.invert_yaxis()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.025))
    fig.suptitle("Representative comparison-model response comparison", y=1.055, fontsize=14)
    path = output_dir / "comparison_models_multiscenario_fem_comparison.png"
    save_figure(fig, path)
    return path


def plot_single_model_comparisons(curves: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for model_group in group_curves_by_model(curves):
        model_label = model_group[0]["model_label"]
        model_slug = model_group[0]["model_slug"]
        fig, axes = plt.subplots(
            1,
            len(RESPONSE_KEYS),
            figsize=(3.2 * len(RESPONSE_KEYS), 1.2),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        for col, response in enumerate(RESPONSE_KEYS):
            ax = axes[0, col]
            ylim_series: list[np.ndarray] = []
            for case_pos, item in enumerate(model_group):
                x = np.asarray(item["x"], dtype=float)
                fem = np.asarray(item["fem"][response], dtype=float)
                pred = np.asarray(item["pred"][response], dtype=float)
                color = CASE_COLORS[case_pos % len(CASE_COLORS)]
                case_index = int(item.get("representative_case_index", case_pos + 1))
                show_legend_label = col == 0
                ax.plot(
                    x, fem, color=color, alpha=0.35, linewidth=3.2,
                    label=f"Case {case_index} FEM" if show_legend_label else None,
                )
                ax.plot(
                    x, pred, color=color, linewidth=1.3, linestyle="--",
                    label=f"Case {case_index} {model_label}" if show_legend_label else None,
                )
                ylim_series.extend([fem, pred])
            apply_readable_ylim(ax, *ylim_series, include_zero=True)
            draw_beam_boundary_axis(ax, model_group[0]["boundary_condition"])
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
            ax.set_title(RESPONSE_TITLES[response], fontsize=14)
            ax.set_xlabel("Beam Position ($x$)", fontsize=14)
            if response == "M":
                ax.invert_yaxis()

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.06))
        fig.suptitle(f"{model_label} representative FEM comparison", y=1.13, fontsize=14)
        path = output_dir / f"{model_slug}_multiscenario_fem_comparison.png"
        save_figure(fig, path)
        paths.append(path)
    return paths


def write_metrics_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["model_slug", "model_label", "case_index", "P", "xp", "response", "rel_l2"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    curves, metrics, model_manifest = generate_curves(args)
    curves_path = output_dir / "comparison_model_representative_curves.json"
    curves_path.write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics_csv(metrics, output_dir / "comparison_model_rel_l2_metrics.csv")

    generated = [plot_multimodel_comparison(curves, output_dir)]
    generated.extend(plot_single_model_comparisons(curves, output_dir))

    manifest = {
        "boundary_condition": "fixed-fixed",
        "load_or_excitation": "Gaussian local load",
        "test_loads": [{"P": P, "xp": xp} for P, xp in TEST_LOADS],
        "style_reference": str((SCRIPT_DIR.parent.parent / "BP-PFNO" / "多场景泛化验证" / "plot_multiscenario_results.py").resolve()),
        "models": model_manifest,
        "curves": str(curves_path.relative_to(output_dir)),
        "figures": [str(path.relative_to(output_dir)) for path in generated],
        "metrics": "comparison_model_rel_l2_metrics.csv",
    }
    (output_dir / "comparison_model_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generated comparison-model FEM figures:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
