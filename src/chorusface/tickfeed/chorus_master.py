"""Lab CHORUS master consumer — Target Pod that dumps received vectors to spool.

chorus_fabric's stock TargetPod decrypts and ACKs but does not expose vectors
back to ChorusFace. This servicer writes each verified direct-mode vector into
``<spool>/recv/vec_XXXXXXXX.f32`` so ``TickFeedTransport.pull_recv_vector``
can feed the Side A ring (lab multi-host consume path).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch

from chorus_fabric import crypto_engine as ce
from chorus_fabric import fabric_pb2, fabric_pb2_grpc

logger = logging.getLogger("chorusface.chorus_master")

TARGET_PORT = int(os.getenv("TARGET_PORT", "50053"))
CONTROL_PLANE_HOST = os.getenv("CONTROL_PLANE_HOST", "localhost")
CONTROL_PLANE_PORT = int(os.getenv("CONTROL_PLANE_PORT", "50051"))
DIM = int(os.getenv("CHORUS_DIM", str(ce.DEFAULT_DIM)))
RECV_SPOOL = Path(
    os.getenv(
        "CHORUSFACE_CHORUS_RECV_SPOOL",
        "output/worlds/tickfeed/tickfeed_chorus_spool/recv",
    )
)
RECV_KEEP = int(os.getenv("CHORUSFACE_CHORUS_RECV_KEEP", "240"))


class _SessionCache:
    def __init__(self) -> None:
        self._c: dict = {}

    def get(self, sid: str):
        e = self._c.get(sid)
        return e if (e and time.time() < e["expires_at"]) else None

    def ensure(self, sid: str, bundle: fabric_pb2.SessionKeyBundle) -> dict:
        entry = self._c.get(sid)
        if entry is None:
            entry = {
                "keys": {},
                "watermark_seed": bundle.watermark_seed,
                "expires_at": bundle.expires_at,
                "dim": bundle.dim,
            }
            self._c[sid] = entry
        entry["keys"][bundle.key_epoch] = ce.bytes_to_matrix(
            bundle.key_matrix_K_inv, bundle.dim
        )
        entry["expires_at"] = bundle.expires_at
        return entry


class MasterTargetServicer(fabric_pb2_grpc.TargetPodServicer):
    """Target Pod that persists decrypted vectors for ChorusFace Side A."""

    def __init__(self, spool: Path | None = None) -> None:
        self._cache = _SessionCache()
        self.spool = Path(spool) if spool is not None else RECV_SPOOL
        self.spool.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        cp_addr = f"{CONTROL_PLANE_HOST}:{CONTROL_PLANE_PORT}"
        ch = grpc.insecure_channel(cp_addr)
        self._cp = fabric_pb2_grpc.ControlPlaneStub(ch)
        logger.info("ChorusFace master target -> CP %s spool=%s", cp_addr, self.spool)

    def StreamSignal(self, request_iterator, context):
        for payload in request_iterator:
            yield self._process(payload)

    def _process(self, p: fabric_pb2.TensorPayload) -> fabric_pb2.SignalAck:
        sid = p.session_id
        seq = p.seq_num
        dim = p.dim or DIM
        epoch = p.key_epoch
        session = self._cache.get(sid)
        if session is None or epoch not in session["keys"]:
            session = self._fetch_session(sid, p.pod_id, dim, epoch)
        if session is None or epoch not in session["keys"]:
            return fabric_pb2.SignalAck(
                session_id=sid,
                seq_num=seq,
                verified=False,
                key_epoch=epoch,
                status="key_expired",
                message=f"No key for epoch {epoch}",
            )
        try:
            v_enc = ce.bytes_to_tensor(p.data, dim)
            v_raw = ce.decrypt(v_enc, session["keys"][epoch])
        except Exception as exc:  # noqa: BLE001
            return fabric_pb2.SignalAck(
                session_id=sid,
                seq_num=seq,
                verified=False,
                key_epoch=epoch,
                status="error",
                message=str(exc),
            )
        if not ce.verify_watermark(v_raw, session["watermark_seed"], seq):
            return fabric_pb2.SignalAck(
                session_id=sid,
                seq_num=seq,
                verified=False,
                key_epoch=epoch,
                status="tampered",
                message="Watermark verification failed",
            )
        self._dump_vector(v_raw, seq=seq)
        norm = float(v_raw.norm().item())
        return fabric_pb2.SignalAck(
            session_id=sid,
            seq_num=seq,
            verified=True,
            key_epoch=epoch,
            signal_norm=norm,
            status="ok",
            message="chorusface-master-consume",
        )

    def _dump_vector(self, v_raw: torch.Tensor, *, seq: int) -> None:
        vec = v_raw.detach().float().cpu().numpy().astype("<f4").reshape(-1)
        self._seq += 1
        path = self.spool / f"vec_{self._seq:08d}_s{int(seq):08d}.f32"
        path.write_bytes(np.ascontiguousarray(vec).tobytes())
        files = sorted(self.spool.glob("vec_*.f32"))
        overflow = len(files) - RECV_KEEP
        for old in files[: max(0, overflow)]:
            try:
                old.unlink()
            except OSError:
                pass

    def _fetch_session(self, sid, pod_id, dim, epoch):
        try:
            bundle = self._cp.GetSessionKey(
                fabric_pb2.KeyRequest(
                    pod_id=f"chorusface-master-{pod_id}",
                    session_id=sid,
                    key_epoch=epoch,
                )
            )
            if not bundle.session_id:
                return None
            return self._cache.ensure(sid, bundle)
        except grpc.RpcError as exc:
            logger.error("Control Plane unreachable: %s", exc)
            return None


def serve(spool: Path | str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fabric_pb2_grpc.add_TargetPodServicer_to_server(
        MasterTargetServicer(Path(spool) if spool else None), server
    )
    addr = f"[::]:{TARGET_PORT}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("ChorusFace CHORUS master target listening on %s", addr)
    server.wait_for_termination()


def main() -> int:
    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
