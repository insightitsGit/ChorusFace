"""Model A — vowel×emotion → 9D target (D12 / F9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chorusface.vowel.priors import all_prior_targets, clamp_9d, prior_9d
from chorusface.vowel.schema import EMOTION_INDEX, GA16_INDEX, GROUP_DIM


def one_hot_22(tag: str, emotion: str) -> NDArray[np.float64]:
    x = np.zeros(22, dtype=np.float64)
    ti = GA16_INDEX.get((tag or "AX").upper(), GA16_INDEX["AX"])
    ei = EMOTION_INDEX.get((emotion or "NEUTRAL").upper(), 0)
    x[ti] = 1.0
    x[16 + ei] = 1.0
    return x


class ModelA:
    """Small MLP 22→64→64→9 trained with numpy (no torch required)."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.05, size=(22, 64))
        self.b1 = np.zeros(64)
        self.W2 = rng.normal(0, 0.05, size=(64, 64))
        self.b2 = np.zeros(64)
        self.W3 = rng.normal(0, 0.05, size=(64, GROUP_DIM))
        self.b3 = np.zeros(GROUP_DIM)
        self.trained = False

    @staticmethod
    def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(x, 0.0)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2) + h1  # residual
        y = h2 @ self.W3 + self.b3
        return clamp_9d(y)

    def predict(self, tag: str, emotion: str) -> NDArray[np.float64]:
        if not self.trained:
            return prior_9d(tag, emotion)
        return self.forward(one_hot_22(tag, emotion))

    def fit(
        self,
        x: NDArray[np.float64] | None = None,
        y: NDArray[np.float64] | None = None,
        epochs: int = 400,
        lr: float = 0.05,
        seed: int = 0,
    ) -> dict[str, float]:
        if x is None or y is None:
            x, y = all_prior_targets()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = x.shape[0]
        history_loss = 0.0
        for _ in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            for i in perm:
                xi = x[i]
                yi = y[i]
                h1_pre = xi @ self.W1 + self.b1
                h1 = self._relu(h1_pre)
                h2_pre = h1 @ self.W2 + self.b2
                h2 = self._relu(h2_pre) + h1
                pred = h2 @ self.W3 + self.b3
                err = pred - yi
                total += float(np.mean(err**2))
                # backprop
                d3 = (2.0 / GROUP_DIM) * err
                gW3 = np.outer(h2, d3)
                gb3 = d3
                dh2 = self.W3 @ d3
                # residual through relu on h2_pre and skip h1
                d_h2_pre = dh2 * (h2_pre > 0)
                gW2 = np.outer(h1, d_h2_pre)
                gb2 = d_h2_pre
                dh1 = self.W2 @ d_h2_pre + dh2  # skip
                d_h1_pre = dh1 * (h1_pre > 0)
                gW1 = np.outer(xi, d_h1_pre)
                gb1 = d_h1_pre
                self.W3 -= lr * gW3
                self.b3 -= lr * gb3
                self.W2 -= lr * gW2
                self.b2 -= lr * gb2
                self.W1 -= lr * gW1
                self.b1 -= lr * gb1
            history_loss = total / n
        self.trained = True
        return {"mse": history_loss}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            trained=np.array([1 if self.trained else 0]),
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelA:
        data = np.load(path)
        m = cls()
        m.W1 = data["W1"]
        m.b1 = data["b1"]
        m.W2 = data["W2"]
        m.b2 = data["b2"]
        m.W3 = data["W3"]
        m.b3 = data["b3"]
        m.trained = bool(int(data["trained"][0]))
        return m

    def try_export_onnx(self, path: str | Path) -> bool:
        """Optional ONNX export if ``onnx`` is installed."""
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except ImportError:
            return False
        # Export as single Gemm chain approximated via MatMul+Add+Relu
        # Keep simple: store as custom npz sibling; mark onnx path for future.
        # Full ONNX graph for residual MLP is verbose — write weights node only.
        path = Path(path)
        tensors = [
            numpy_helper.from_array(self.W1.astype(np.float32), "W1"),
            numpy_helper.from_array(self.b1.astype(np.float32), "b1"),
            numpy_helper.from_array(self.W2.astype(np.float32), "W2"),
            numpy_helper.from_array(self.b2.astype(np.float32), "b2"),
            numpy_helper.from_array(self.W3.astype(np.float32), "W3"),
            numpy_helper.from_array(self.b3.astype(np.float32), "b3"),
        ]
        nodes = [
            helper.make_node("MatMul", ["x", "W1"], ["h1p"]),
            helper.make_node("Add", ["h1p", "b1"], ["h1b"]),
            helper.make_node("Relu", ["h1b"], ["h1"]),
            helper.make_node("MatMul", ["h1", "W2"], ["h2p"]),
            helper.make_node("Add", ["h2p", "b2"], ["h2b"]),
            helper.make_node("Relu", ["h2b"], ["h2r"]),
            helper.make_node("Add", ["h2r", "h1"], ["h2"]),
            helper.make_node("MatMul", ["h2", "W3"], ["yp"]),
            helper.make_node("Add", ["yp", "b3"], ["y"]),
        ]
        graph = helper.make_graph(
            nodes,
            "ModelA",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 22])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, GROUP_DIM])],
            initializer=tensors,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, str(path))
        return True
