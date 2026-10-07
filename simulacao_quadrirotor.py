"""Reproduz as simulacoes, metricas e figuras apresentadas no TCC.

O modelo e linearizado em torno do voo pairado. A subida inicial e modelada
sem contato com o solo, efeito solo ou dinamica nao linear completa.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Parametros fisicos adotados de Luukkonen (2011).
G, M = 9.81, 0.468
IXX, IYY, IZZ = 4.856e-3, 4.856e-3, 8.801e-3
L, KF, KM = 0.225, 2.980e-6, 1.140e-7

# Hipoteses dos ensaios numericos.
DT = 0.002
OMEGA_MAX = 900.0
ANGLE_MAX = np.deg2rad(12.0)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figs"


def state_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Matrizes do modelo linearizado no ponto de voo pairado."""
    a = np.zeros((12, 12))
    a[0, 1], a[1, 8] = 1.0, G
    a[2, 3], a[3, 6] = 1.0, -G
    a[4, 5] = a[6, 7] = a[8, 9] = a[10, 11] = 1.0
    b = np.zeros((12, 4))
    b[5, 0], b[7, 1] = 1.0 / M, 1.0 / IXX
    b[9, 2], b[11, 3] = 1.0 / IYY, 1.0 / IZZ
    return a, b


A, B = state_matrices()

# u = G_mistura @ [omega_1^2, ..., omega_4^2].
MIXER = np.array(
    [
        [KF, KF, KF, KF],
        [0.0, -KF * L, 0.0, KF * L],
        [-KF * L, 0.0, KF * L, 0.0],
        [-KM, KM, -KM, KM],
    ]
)
MIXER_INV = np.linalg.inv(MIXER)


@dataclass(frozen=True)
class PoleDesign:
    """Especificacoes e ganhos obtidos por alocacao de tres polos."""

    zeta: float
    settling_time: float
    third_pole_ratio: float
    plant_gain: float

    @property
    def wn(self) -> float:
        return 4.0 / (self.zeta * self.settling_time)

    @property
    def third_rate(self) -> float:
        return self.third_pole_ratio * self.zeta * self.wn

    @property
    def gains(self) -> tuple[float, float, float]:
        wn, a, k = self.wn, self.third_rate, self.plant_gain
        kd = (2.0 * self.zeta * wn + a) / k
        kp = (wn**2 + 2.0 * a * self.zeta * wn) / k
        ki = a * wn**2 / k
        return kp, ki, kd

    @property
    def poles(self) -> np.ndarray:
        sigma = -self.zeta * self.wn
        wd = self.wn * np.sqrt(1.0 - self.zeta**2)
        return np.array([sigma + 1j * wd, sigma - 1j * wd, -self.third_rate])


DESIGNS = {
    "altitude": PoleDesign(0.90, 2.50, 5.0, 1.0 / M),
    "attitude": PoleDesign(0.90, 0.80, 5.0, 1.0 / IXX),
    "yaw": PoleDesign(0.90, 1.50, 5.0, 1.0 / IZZ),
    "position": PoleDesign(0.90, 4.00, 5.0, G),
}


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    beta: float = 1.0
    integral_limit: float = np.inf
    integral: float = 0.0

    def output(self, reference: float, measurement: float, rate: float) -> float:
        """PID 2-GDL com derivada na medida, evitando salto derivativo."""
        return (
            self.kp * (self.beta * reference - measurement)
            + self.ki * self.integral
            - self.kd * rate
        )

    def update(self, error: float, requested: float, applied: float, dt: float) -> None:
        """Integracao condicional: anti-windup por clamping."""
        saturation_error = requested - applied
        if abs(saturation_error) < 1e-10 or error * saturation_error < 0.0:
            self.integral = float(
                np.clip(
                    self.integral + error * dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )


def make_controllers() -> dict[str, PID]:
    weights = {
        "z": 0.35,
        "phi": 0.55,
        "theta": 0.55,
        "psi": 0.35,
        "x": 0.35,
        "y": 0.35,
    }
    associations = {
        "z": DESIGNS["altitude"],
        "phi": DESIGNS["attitude"],
        "theta": DESIGNS["attitude"],
        "psi": DESIGNS["yaw"],
        "x": DESIGNS["position"],
        "y": DESIGNS["position"],
    }
    return {
        name: PID(*design.gains, beta=weights[name], integral_limit=20.0)
        for name, design in associations.items()
    }


def allocate_actuators(requested: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    raw_squared = MIXER_INV @ requested
    squared = np.clip(raw_squared, 0.0, OMEGA_MAX**2)
    saturated = bool(np.any(np.abs(squared - raw_squared) > 1e-9))
    return MIXER @ squared, np.sqrt(squared), saturated


ReferenceFunction = Callable[[float], np.ndarray]


def simulate(
    duration: float,
    initial_state: np.ndarray,
    reference: ReferenceFunction,
    dt: float = DT,
) -> dict[str, np.ndarray]:
    controllers = make_controllers()
    initial_reference = np.asarray(reference(0.0), dtype=float)
    for name, reference_value in zip(("x", "y", "z", "psi"), initial_reference):
        controller = controllers[name]
        controller.integral = (
            controller.kp * (1.0 - controller.beta) * reference_value
            / controller.ki
        )

    time = np.arange(0.0, duration + dt, dt)
    state = np.zeros((12, time.size))
    references = np.zeros((4, time.size))
    commands = np.zeros((6, time.size))
    controls = np.zeros((4, time.size))
    rotors = np.zeros((4, time.size))
    saturated = np.zeros(time.size, dtype=bool)
    state[:, 0] = initial_state

    for index in range(time.size - 1):
        value = state[:, index]
        ref = np.asarray(reference(float(time[index])), dtype=float)
        x_ref, y_ref, z_ref, psi_ref = ref
        references[:, index] = ref

        theta_raw = controllers["x"].output(x_ref, value[0], value[1])
        theta_ref = float(np.clip(theta_raw, -ANGLE_MAX, ANGLE_MAX))
        controllers["x"].update(x_ref - value[0], theta_raw, theta_ref, dt)

        minus_phi_raw = controllers["y"].output(y_ref, value[2], value[3])
        minus_phi_ref = float(np.clip(minus_phi_raw, -ANGLE_MAX, ANGLE_MAX))
        phi_ref = -minus_phi_ref
        controllers["y"].update(y_ref - value[2], minus_phi_raw, minus_phi_ref, dt)

        du1_raw = controllers["z"].output(z_ref, value[4], value[5])
        u2_raw = controllers["phi"].output(phi_ref, value[6], value[7])
        u3_raw = controllers["theta"].output(theta_ref, value[8], value[9])
        u4_raw = controllers["psi"].output(psi_ref, value[10], value[11])
        requested = np.array([M * G + du1_raw, u2_raw, u3_raw, u4_raw])
        applied, omega, is_saturated = allocate_actuators(requested)
        applied_linear = np.array([applied[0] - M * G, *applied[1:]])
        requested_linear = np.array([du1_raw, u2_raw, u3_raw, u4_raw])

        errors = np.array(
            [
                z_ref - value[4],
                phi_ref - value[6],
                theta_ref - value[8],
                psi_ref - value[10],
            ]
        )
        inner = (
            controllers["z"],
            controllers["phi"],
            controllers["theta"],
            controllers["psi"],
        )
        for controller, error, req, app in zip(
            inner, errors, requested_linear, applied_linear
        ):
            controller.update(float(error), float(req), float(app), dt)

        def dynamics(current: np.ndarray) -> np.ndarray:
            return A @ current + B @ applied_linear

        k1 = dynamics(value)
        k2 = dynamics(value + 0.5 * dt * k1)
        k3 = dynamics(value + 0.5 * dt * k2)
        k4 = dynamics(value + dt * k3)
        state[:, index + 1] = value + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        commands[:, index] = [x_ref, y_ref, z_ref, psi_ref, phi_ref, theta_ref]
        controls[:, index], rotors[:, index] = applied_linear, omega
        saturated[index] = is_saturated

    references[:, -1] = reference(float(time[-1]))
    for series in (commands, controls, rotors):
        series[:, -1] = series[:, -2]
    saturated[-1] = saturated[-2]
    return {
        "t": time,
        "state": state,
        "reference": references,
        "command": commands,
        "control": controls,
        "rotor": rotors,
        "saturated": saturated,
    }


def smooth_transition(t: float, start: float, end: float) -> float:
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    tau = (t - start) / (end - start)
    return 3.0 * tau**2 - 2.0 * tau**3


def climb_reference(t: float) -> np.ndarray:
    """Subida modelada ate 5 m, sem representar contato com o solo."""
    return np.array([0.0, 0.0, 5.0 * smooth_transition(t, 0.5, 4.5), 0.0])


def position_reference(t: float) -> np.ndarray:
    factor = smooth_transition(t, 1.0, 5.0)
    return np.array([2.0 * factor, 1.5 * factor, 5.0, 0.0])


def yaw_reference(t: float) -> np.ndarray:
    factor = smooth_transition(t, 1.0, 3.0)
    return np.array([0.0, 0.0, 5.0, np.deg2rad(15.0) * factor])


def hover_reference(_: float) -> np.ndarray:
    return np.array([0.0, 0.0, 5.0, 0.0])


def tracking_metrics(
    time: np.ndarray,
    response: np.ndarray,
    final_value: float,
    hold_start: float,
) -> dict[str, float]:
    mask = time >= hold_start
    local_time, local_response = time[mask] - hold_start, response[mask]
    scale = abs(final_value) if abs(final_value) > 1e-12 else 1.0
    if final_value >= 0.0:
        overshoot = max(0.0, (float(np.max(local_response)) - final_value) / scale * 100)
    else:
        overshoot = max(0.0, (final_value - float(np.min(local_response))) / scale * 100)
    outside = np.flatnonzero(np.abs(local_response - final_value) > 0.02 * scale)
    settling = 0.0
    if outside.size:
        next_index = outside[-1] + 1
        settling = float(local_time[next_index]) if next_index < local_time.size else float("nan")
    return {
        "overshoot": overshoot,
        "settling": settling,
        "terminal_mean_error": float(final_value - np.mean(local_response[-500:])),
    }


def controllability_rank() -> int:
    matrix = np.hstack([np.linalg.matrix_power(A, power) @ B for power in range(12)])
    return int(np.linalg.matrix_rank(matrix))


def run_scenarios() -> dict[str, dict[str, np.ndarray]]:
    hover = np.zeros(12)
    hover[4] = 5.0
    perturbed = hover.copy()
    perturbed[[0, 2, 4]] = [0.25, -0.20, 5.15]
    perturbed[[6, 8, 10]] = np.deg2rad([3.0, -3.0, 2.0])
    return {
        "climb": simulate(14.0, np.zeros(12), climb_reference),
        "position": simulate(14.0, hover, position_reference),
        "yaw": simulate(8.0, hover, yaw_reference),
        "recovery": simulate(10.0, perturbed, hover_reference),
    }


def configure_plotting() -> None:
    plt.rcParams.update(
        {"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.28}
    )


def save_figures(
    runs: dict[str, dict[str, np.ndarray]], output_dir: Path, dpi: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    run = runs["climb"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].plot(run["t"], run["reference"][2], "k--", label="Referência")
    axes[0].plot(run["t"], run["state"][4], label="Altitude")
    axes[0].set_ylabel("Altitude z [m]")
    axes[0].legend()
    for rotor in range(4):
        axes[1].plot(run["t"], run["rotor"][rotor], label=f"Rotor {rotor + 1}")
    axes[1].axhline(OMEGA_MAX, color="k", ls="--", lw=1, label="Limite")
    axes[1].set(xlabel="Tempo [s]", ylabel="Velocidade [rad/s]")
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "resultado_decolagem_pairado.png", dpi=dpi)
    plt.close(fig)

    run = runs["position"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    axes[0, 0].plot(run["t"], run["reference"][0], "k--", label="Referência")
    axes[0, 0].plot(run["t"], run["state"][0], label="Posição")
    axes[0, 0].set_ylabel("x [m]")
    axes[0, 0].legend()
    axes[0, 1].plot(run["t"], run["reference"][1], "k--", label="Referência")
    axes[0, 1].plot(run["t"], run["state"][2], label="Posição")
    axes[0, 1].set_ylabel("y [m]")
    axes[0, 1].legend()
    axes[1, 0].plot(run["t"], np.rad2deg(run["command"][5]), "k--", label=r"$\theta_d$")
    axes[1, 0].plot(run["t"], np.rad2deg(run["state"][8]), label=r"$\theta$")
    axes[1, 0].set(xlabel="Tempo [s]", ylabel="Arfagem [graus]")
    axes[1, 0].legend()
    axes[1, 1].plot(run["t"], np.rad2deg(run["command"][4]), "k--", label=r"$\phi_d$")
    axes[1, 1].plot(run["t"], np.rad2deg(run["state"][6]), label=r"$\phi$")
    axes[1, 1].set(xlabel="Tempo [s]", ylabel="Rolagem [graus]")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "resultado_posicao_horizontal.png", dpi=dpi)
    plt.close(fig)

    run = runs["yaw"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].plot(run["t"], np.rad2deg(run["reference"][3]), "k--", label="Referência")
    axes[0].plot(run["t"], np.rad2deg(run["state"][10]), label="Guinada")
    axes[0].set_ylabel("Guinada [graus]")
    axes[0].legend()
    for rotor in range(4):
        axes[1].plot(run["t"], run["rotor"][rotor], label=f"Rotor {rotor + 1}")
    axes[1].axhline(OMEGA_MAX, color="k", ls="--", lw=1, label="Limite")
    axes[1].set(xlabel="Tempo [s]", ylabel="Velocidade [rad/s]")
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "resultado_guinada.png", dpi=dpi)
    plt.close(fig)

    run = runs["recovery"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].plot(run["t"], run["state"][0], label="x")
    axes[0].plot(run["t"], run["state"][2], label="y")
    axes[0].plot(run["t"], run["state"][4] - 5.0, label="z - 5 m")
    axes[0].set_ylabel("Desvio de posição [m]")
    axes[0].legend(ncol=3)
    axes[1].plot(run["t"], np.rad2deg(run["state"][6]), label="Rolagem")
    axes[1].plot(run["t"], np.rad2deg(run["state"][8]), label="Arfagem")
    axes[1].plot(run["t"], np.rad2deg(run["state"][10]), label="Guinada")
    axes[1].set(xlabel="Tempo [s]", ylabel="Ângulo [graus]")
    axes[1].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "resultado_recuperacao_pairado.png", dpi=dpi)
    plt.close(fig)


def summarize(runs: dict[str, dict[str, np.ndarray]]) -> list[dict[str, float | str]]:
    cases = [
        ("Altitude z", "climb", 4, 5.0, 4.5, "m", 1.0),
        ("Posicao x", "position", 0, 2.0, 5.0, "m", 1.0),
        ("Posicao y", "position", 2, 1.5, 5.0, "m", 1.0),
        ("Guinada psi", "yaw", 10, np.deg2rad(15.0), 3.0, "grau", 180.0 / np.pi),
    ]
    rows: list[dict[str, float | str]] = []
    for label, scenario, index, final_value, hold_start, error_unit, error_scale in cases:
        run = runs[scenario]
        metric = tracking_metrics(run["t"], run["state"][index], final_value, hold_start)
        rows.append(
            {
                "canal": label,
                "tempo_acomodacao_pos_transicao_s": metric["settling"],
                "sobressinal_pos_transicao_percentual": metric["overshoot"],
                "erro_medio_terminal": metric["terminal_mean_error"] * error_scale,
                "unidade_erro_medio_terminal": error_unit,
                "rotor_max_rad_s": float(np.max(run["rotor"])),
                "saturacao_percentual": float(np.mean(run["saturated"]) * 100.0),
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa os quatro cenarios numericos apresentados no TCC."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretorio para figuras e CSV (padrao: pasta figs do projeto).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolucao das figuras PNG (padrao: 300).",
    )
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi deve ser um inteiro positivo")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    runs = run_scenarios()
    rows = summarize(runs)
    save_figures(runs, output_dir, args.dpi)
    with (output_dir / "metricas_simulacao.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Posto da matriz de controlabilidade: {controllability_rank()}")
    for name, design in DESIGNS.items():
        print(f"{name}: Kp, Ki, Kd = {design.gains}; polos = {design.poles}")
    for row in rows:
        print(
            f"{row['canal']}: "
            f"ts_pos={row['tempo_acomodacao_pos_transicao_s']:.3f} s; "
            f"OS_pos={row['sobressinal_pos_transicao_percentual']:.3f}%; "
            f"erro_medio_terminal={row['erro_medio_terminal']:.6e} "
            f"{row['unidade_erro_medio_terminal']}; "
            f"omega_max={row['rotor_max_rad_s']:.2f} rad/s; "
            f"saturacao={row['saturacao_percentual']:.3f}%"
        )
    recovery = runs["recovery"]
    position_error = recovery["state"][[0, 2, 4], -1] - [0.0, 0.0, 5.0]
    angle_error = recovery["state"][[6, 8, 10], -1]
    print("Maior erro final na recuperacao [m]:", np.max(np.abs(position_error)))
    print("Maior erro angular final [graus]:", np.rad2deg(np.max(np.abs(angle_error))))
    print(
        "Amostras com saturacao na recuperacao:",
        int(np.sum(recovery["saturated"])),
        "de",
        recovery["t"].size,
    )
    print("Arquivos gravados em:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
