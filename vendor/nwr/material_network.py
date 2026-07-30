"""Train and export the neural material network used by the render shader.

The network is a small multilayer perceptron, ``32 -> 16 (ReLU) -> 3``, trained
to reproduce the procedural material response. Distilling the analytic shading
into weights gives a genuine learned material stage with a measurable error,
and keeps the runtime free of any inference framework: the weights are uploaded
as a single-channel float texture and evaluated directly in GLSL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from bds_format import ANCHORS, DTYPE, VECTOR_DIMENSIONS

HIDDEN_UNITS: Final = 16
OUTPUT_CHANNELS: Final = 3
WEIGHT_ROWS: Final = HIDDEN_UNITS + OUTPUT_CHANNELS
WEIGHT_COLUMNS: Final = VECTOR_DIMENSIONS + 1
DEFAULT_WEIGHTS_PATH: Final = Path(__file__).resolve().parent / "material_weights.npy"


def procedural_material(states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Reference shading target, mirroring ``procedural_material`` in GLSL."""
    samples = np.atleast_2d(np.asarray(states, dtype=np.float32))
    albedo = np.maximum(samples[:, 8:11], 0.0)
    opacity = np.clip(samples[:, 11:12], 0.0, 1.0)
    emission = np.maximum(samples[:, 14:15], 0.0)
    energy = np.maximum(samples[:, 7:8], 0.0)
    intent = np.linalg.norm(samples[:, 16:24], axis=1, keepdims=True)
    glow = emission * 1.8 + energy * 0.65 + intent * 0.35
    tint = albedo * 0.75 + np.asarray([0.12, 0.72, 1.0], dtype=np.float32) * 0.25
    return (albedo * (0.2 + opacity * 0.8) + tint * glow).astype(np.float32)


def sample_states(
    count: int,
    *,
    generator: np.random.Generator,
    clean_fraction: float = 0.3,
    noise_scale: float = 0.05,
) -> npt.NDArray[np.float32]:
    """Draw plausible cell states spanning anchors, blends, and noise.

    A portion of the samples are exact anchors with no noise. Vacuum is by far
    the most common cell in practice, so the network must reproduce it exactly
    or the rendered black level lifts into grey.
    """
    anchors = np.asarray(list(ANCHORS.values()), dtype=np.float32)
    indices = generator.integers(0, anchors.shape[0], size=count)
    base = anchors[indices]
    scale = generator.uniform(0.0, 1.25, size=(count, 1)).astype(np.float32)
    blend_indices = generator.integers(0, anchors.shape[0], size=count)
    blend_weight = generator.uniform(0.0, 0.5, size=(count, 1)).astype(np.float32)
    mixed = base * (1.0 - blend_weight) + anchors[blend_indices] * blend_weight
    noise = generator.normal(0.0, noise_scale, size=(count, VECTOR_DIMENSIONS))
    states = mixed * scale + noise.astype(np.float32)

    clean = generator.random(count) < clean_fraction
    states[clean] = base[clean]
    return np.clip(states, -1.5, 1.5).astype(np.float32)


def train(
    *,
    samples: int = 60_000,
    epochs: int = 400,
    batch_size: int = 512,
    learning_rate: float = 3e-3,
    seed: int = 20260729,
    verbose: bool = False,
) -> tuple[npt.NDArray[np.float32], dict[str, float]]:
    """Fit the material network and return its weights plus error metrics."""
    generator = np.random.default_rng(seed)
    inputs = sample_states(samples, generator=generator)
    targets = procedural_material(inputs)
    split = int(samples * 0.9)
    train_x, validate_x = inputs[:split], inputs[split:]
    train_y, validate_y = targets[:split], targets[split:]

    scale = np.sqrt(2.0 / VECTOR_DIMENSIONS)
    hidden_weights = (
        generator.normal(0.0, scale, size=(VECTOR_DIMENSIONS, HIDDEN_UNITS))
    ).astype(np.float32)
    hidden_bias = np.zeros(HIDDEN_UNITS, dtype=np.float32)
    output_weights = (
        generator.normal(0.0, np.sqrt(2.0 / HIDDEN_UNITS), size=(HIDDEN_UNITS, OUTPUT_CHANNELS))
    ).astype(np.float32)
    output_bias = np.zeros(OUTPUT_CHANNELS, dtype=np.float32)

    parameters = [hidden_weights, hidden_bias, output_weights, output_bias]
    moments = [np.zeros_like(item) for item in parameters]
    velocities = [np.zeros_like(item) for item in parameters]
    step = 0

    for epoch in range(epochs):
        order = generator.permutation(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            batch = order[start : start + batch_size]
            x = train_x[batch]
            y = train_y[batch]

            pre_activation = x @ parameters[0] + parameters[1]
            hidden = np.maximum(pre_activation, 0.0)
            prediction = hidden @ parameters[2] + parameters[3]

            residual = (prediction - y) * (2.0 / x.shape[0])
            gradients = [
                None,
                None,
                hidden.T @ residual,
                residual.sum(axis=0),
            ]
            hidden_gradient = residual @ parameters[2].T
            hidden_gradient[pre_activation <= 0.0] = 0.0
            gradients[0] = x.T @ hidden_gradient
            gradients[1] = hidden_gradient.sum(axis=0)

            step += 1
            for index, gradient in enumerate(gradients):
                moments[index] = 0.9 * moments[index] + 0.1 * gradient
                velocities[index] = 0.999 * velocities[index] + 0.001 * gradient**2
                corrected_moment = moments[index] / (1.0 - 0.9**step)
                corrected_velocity = velocities[index] / (1.0 - 0.999**step)
                parameters[index] = (
                    parameters[index]
                    - learning_rate * corrected_moment / (np.sqrt(corrected_velocity) + 1e-8)
                ).astype(np.float32)

        if verbose and (epoch + 1) % max(epochs // 10, 1) == 0:
            loss = _mean_squared_error(parameters, validate_x, validate_y)
            print(f"epoch {epoch + 1}/{epochs} validation mse {loss:.6f}")

    metrics = {
        "train_mse": _mean_squared_error(parameters, train_x, train_y),
        "validation_mse": _mean_squared_error(parameters, validate_x, validate_y),
        "validation_max_error": _max_error(parameters, validate_x, validate_y),
    }
    return pack_weights(*parameters), metrics


def pack_weights(
    hidden_weights: npt.NDArray[np.float32],
    hidden_bias: npt.NDArray[np.float32],
    output_weights: npt.NDArray[np.float32],
    output_bias: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Lay the parameters out as the texture the fragment shader samples."""
    table = np.zeros((WEIGHT_ROWS, WEIGHT_COLUMNS), dtype=DTYPE)
    table[:HIDDEN_UNITS, :VECTOR_DIMENSIONS] = hidden_weights.T
    table[:HIDDEN_UNITS, VECTOR_DIMENSIONS] = hidden_bias
    table[HIDDEN_UNITS:, :HIDDEN_UNITS] = output_weights.T
    table[HIDDEN_UNITS:, HIDDEN_UNITS] = output_bias
    return table


def unpack_weights(
    table: npt.NDArray[np.float32],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    if table.shape != (WEIGHT_ROWS, WEIGHT_COLUMNS):
        raise ValueError(
            f"Weight table must have shape {(WEIGHT_ROWS, WEIGHT_COLUMNS)}"
        )
    return (
        table[:HIDDEN_UNITS, :VECTOR_DIMENSIONS].T.copy(),
        table[:HIDDEN_UNITS, VECTOR_DIMENSIONS].copy(),
        table[HIDDEN_UNITS:, :HIDDEN_UNITS].T.copy(),
        table[HIDDEN_UNITS:, HIDDEN_UNITS].copy(),
    )


def evaluate(
    table: npt.NDArray[np.float32],
    states: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Run the packed network exactly as the shader does."""
    hidden_weights, hidden_bias, output_weights, output_bias = unpack_weights(table)
    samples = np.atleast_2d(np.asarray(states, dtype=np.float32))
    hidden = np.maximum(samples @ hidden_weights + hidden_bias, 0.0)
    return np.maximum(hidden @ output_weights + output_bias, 0.0)


def save_material_weights(
    path: str | Path,
    table: npt.NDArray[np.float32],
) -> None:
    if table.shape != (WEIGHT_ROWS, WEIGHT_COLUMNS):
        raise ValueError(f"Expected shape {(WEIGHT_ROWS, WEIGHT_COLUMNS)}")
    if not np.isfinite(table).all():
        raise ValueError("Weights must be finite")
    np.save(Path(path), np.ascontiguousarray(table, dtype=DTYPE))


def load_material_weights(path: str | Path) -> npt.NDArray[np.float32]:
    table = np.load(Path(path))
    if table.shape != (WEIGHT_ROWS, WEIGHT_COLUMNS):
        raise ValueError(
            f"{path} has shape {table.shape}; expected {(WEIGHT_ROWS, WEIGHT_COLUMNS)}"
        )
    if not np.isfinite(table).all():
        raise ValueError(f"{path} contains non-finite weights")
    return np.ascontiguousarray(table, dtype=DTYPE)


def _mean_squared_error(
    parameters: list[npt.NDArray[np.float32]],
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
) -> float:
    hidden = np.maximum(x @ parameters[0] + parameters[1], 0.0)
    prediction = hidden @ parameters[2] + parameters[3]
    return float(np.mean((prediction - y) ** 2))


def _max_error(
    parameters: list[npt.NDArray[np.float32]],
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
) -> float:
    hidden = np.maximum(x @ parameters[0] + parameters[1], 0.0)
    prediction = hidden @ parameters[2] + parameters[3]
    return float(np.abs(prediction - y).max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the material network.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    table, metrics = train(
        samples=arguments.samples,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        verbose=not arguments.quiet,
    )
    save_material_weights(arguments.output, table)
    print(f"Saved {arguments.output}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
