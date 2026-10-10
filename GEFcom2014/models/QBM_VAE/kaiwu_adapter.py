# -*- coding: utf-8 -*-

"""Pure-NumPy Kaiwu/SPQC bridge for post-training FA-BM-VAE sampling.

The repository keeps the logical convention
E(s) = -h.T @ s - 0.5 * s.T @ J @ s.
Kaiwu receives an integer 49-by-49 matrix with one auxiliary spin fixed by
the gauge relation u = [s, +1].  No training, precision reduction, clipping,
variable splitting, or platform submission happens in the NumPy functions.
"""

import hashlib
import importlib
import importlib.metadata
import inspect
import os

import numpy as np


LOGICAL_N_BITS = 48
AUXILIARY_SPIN_INDEX = LOGICAL_N_BITS
HARDWARE_N_BITS = LOGICAL_N_BITS + 1
INT8_MIN = -128
INT8_MAX = 127
MIN_SAMPLING_READS = 10
MAX_SAMPLING_READS = 2000


def kaiwu_sdk_versions(kaiwu_module=None):
    """Return both distribution and imported-module Kaiwu versions."""
    try:
        distribution = importlib.metadata.version("kaiwu")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    if kaiwu_module is None:
        try:
            kaiwu_module = importlib.import_module("kaiwu")
        except ImportError:
            kaiwu_module = None
    module = getattr(kaiwu_module, "__version__", None)
    return {
        "importlib_metadata_version": distribution,
        "kaiwu_module_version": module,
        "version_mismatch": bool(distribution and module and distribution != module),
    }


def _validate_logical_ising(h, j, logical_n_bits=LOGICAL_N_BITS):
    h = np.asarray(h, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    if h.shape != (logical_n_bits,):
        raise ValueError(
            "h must have shape (%d,), got %s" % (logical_n_bits, h.shape)
        )
    if j.shape != (logical_n_bits, logical_n_bits):
        raise ValueError(
            "J must have shape (%d, %d), got %s"
            % (logical_n_bits, logical_n_bits, j.shape)
        )
    if not np.isfinite(h).all() or not np.isfinite(j).all():
        raise ValueError("h and J must be finite")
    if not np.allclose(j, j.T, atol=1e-12, rtol=0.0):
        raise ValueError("J must be symmetric")
    if not np.allclose(np.diag(j), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("J must have a zero diagonal")
    return h, j


def hardware_matrix_from_ising(h, j, logical_n_bits=LOGICAL_N_BITS):
    """Return the exact floating-point auxiliary-spin matrix M."""
    h, j = _validate_logical_ising(h, j, logical_n_bits)
    matrix = np.zeros((logical_n_bits + 1, logical_n_bits + 1), dtype=np.float64)
    matrix[:logical_n_bits, :logical_n_bits] = -0.5 * j
    matrix[:logical_n_bits, logical_n_bits] = -0.5 * h
    matrix[logical_n_bits, :logical_n_bits] = -0.5 * h
    return matrix


def validate_hardware_matrix(matrix, n_bits=None):
    matrix = np.asarray(matrix)
    if n_bits is None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Hardware matrix must be square")
        n_bits = matrix.shape[0]
    if matrix.shape != (n_bits, n_bits):
        raise ValueError("Hardware matrix must have shape (%d, %d)" % (n_bits, n_bits))
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("Hardware matrix must have an integer dtype")
    if not np.allclose(matrix, matrix.T, atol=0, rtol=0):
        raise ValueError("Hardware matrix must be exactly symmetric")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("Hardware matrix diagonal must be exactly zero")
    if np.any(matrix < INT8_MIN) or np.any(matrix > INT8_MAX):
        raise ValueError("Hardware matrix entries must lie in [-128, 127]")
    return matrix.astype(np.int8, copy=False)


def matrix_sha256(matrix):
    """Hash a canonical little-endian int16 row-major matrix representation."""
    matrix = validate_hardware_matrix(matrix)
    return hashlib.sha256(
        np.asarray(matrix, dtype="<i2", order="C").tobytes(order="C")
    ).hexdigest()


def quantize_hardware_matrix(h, j, hardware_gain, logical_n_bits=LOGICAL_N_BITS):
    """Build and audit one globally-gained, non-clipped int8-compatible matrix."""
    if not np.isfinite(hardware_gain) or hardware_gain <= 0:
        raise ValueError("hardware_gain must be finite and positive")
    matrix = hardware_matrix_from_ising(h, j, logical_n_bits)
    scaled = float(hardware_gain) * matrix
    rounded = np.rint(scaled)
    if np.any(rounded < INT8_MIN) or np.any(rounded > INT8_MAX):
        raise OverflowError(
            "hardware_gain produces coefficients outside [-128, 127]; "
            "reduce the global gain and rebuild from the original h,J"
        )
    quantized = validate_hardware_matrix(rounded.astype(np.int16))
    upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    source_nonzero = upper & (np.abs(matrix) > 0.0)
    quantized_nonzero = upper & (np.abs(quantized) > 0)
    source_values = matrix[source_nonzero]
    quantized_values = quantized[source_nonzero].astype(np.float64)
    relative_error = np.linalg.norm(
        quantized_values / float(hardware_gain) - source_values
    ) / max(np.linalg.norm(source_values), 1e-15)
    audit = {
        "logical_n_bits": int(logical_n_bits),
        "hardware_n_bits": int(logical_n_bits + 1),
        "auxiliary_spin_index": int(logical_n_bits),
        "hardware_gain": float(hardware_gain),
        "rounding_rule": "numpy.rint, round-half-to-even",
        "clipping": False,
        "precision_reducer": False,
        "variable_splitting": False,
        "max_abs_integer": int(np.abs(quantized).max()),
        "nonzero_edges": int(np.count_nonzero(quantized_nonzero)),
        "source_nonzero_coefficients": int(source_values.size),
        "nonzero_quantized_to_zero_fraction": float(
            np.count_nonzero(source_nonzero & (quantized == 0))
            / max(1, source_values.size)
        ),
        "relative_quantization_error": float(relative_error),
        "matrix_sha256": matrix_sha256(quantized),
    }
    return {
        "matrix": quantized,
        "source_matrix": matrix,
        "hardware_gain": float(hardware_gain),
        "audit": audit,
    }


def normalize_hardware_spins(
    raw_spins, logical_n_bits=LOGICAL_N_BITS, auxiliary_spin_index=None
):
    """Fix the auxiliary-spin gauge and return canonical logical spins."""
    raw_spins = np.asarray(raw_spins)
    hardware_n_bits = logical_n_bits + 1
    if raw_spins.ndim != 2 or raw_spins.shape[1] != hardware_n_bits:
        raise ValueError(
            "Raw hardware spins must have shape (n_reads, %d), got %s"
            % (hardware_n_bits, raw_spins.shape)
        )
    if not np.isin(raw_spins, (-1, 1)).all():
        raise ValueError("Raw hardware spins must contain only -1 and +1")
    if auxiliary_spin_index is None:
        auxiliary_spin_index = logical_n_bits
    if auxiliary_spin_index != logical_n_bits:
        raise ValueError("The auxiliary spin must be the final hardware column")
    logical = raw_spins[:, :logical_n_bits] * raw_spins[:, auxiliary_spin_index, None]
    return logical.astype(np.int8, copy=False)


def binary_to_spin(bits):
    bits = np.asarray(bits)
    if not np.isin(bits, (0, 1)).all():
        raise ValueError("Binary values must be 0 or 1")
    return (2 * bits - 1).astype(np.int8)


def spin_to_binary(spins):
    spins = np.asarray(spins)
    if not np.isin(spins, (-1, 1)).all():
        raise ValueError("Spin values must be -1 or +1")
    return ((spins + 1) // 2).astype(np.int8)


def validate_sampling_reads(num_reads):
    num_reads = int(num_reads)
    if not MIN_SAMPLING_READS <= num_reads <= MAX_SAMPLING_READS:
        raise ValueError(
            "Kaiwu SAMPLING reads must be between %d and %d"
            % (MIN_SAMPLING_READS, MAX_SAMPLING_READS)
        )
    return num_reads


def build_task_name(instance_id, matrix_hash, run_id=None, protocol_id="fa-bm-vae-kaiwu-13k-v1",
                    stage="vs-calibration", gain=None):
    """Derive a stable task name from immutable protocol inputs.

    ``run_id`` is retained for source compatibility but is ignored: a retry must
    never manufacture a billable second task for the same matrix.
    """
    safe_instance = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(instance_id))
    safe_protocol = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(protocol_id))
    safe_stage = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(stage))
    gain_part = "na" if gain is None else str(gain).replace(".", "_")
    return "fa_bm_vae_%s_%s_%s_g%s_%s" % (safe_protocol, safe_stage, safe_instance, gain_part, str(matrix_hash)[:16])


def load_kaiwu_sdk():
    try:
        kaiwu = importlib.import_module("kaiwu")
        cim = importlib.import_module("kaiwu.cim")
    except ImportError as exc:
        raise RuntimeError(
            "Kaiwu SDK is not installed. Install it in the separate hardware "
            "environment and configure its license outside this repository."
        ) from exc
    optimizer_cls = getattr(cim, "CIMOptimizer", None)
    task_mode = getattr(cim, "TaskMode", None)
    if optimizer_cls is None or task_mode is None:
        raise RuntimeError("Installed Kaiwu SDK does not expose kaiwu.cim.CIMOptimizer/TaskMode")
    return kaiwu, optimizer_cls, task_mode


def official_task_query_capability(optimizer_cls=None):
    """Return a public task-name/task-id query method, if the SDK exposes one.

    Private SDK helpers and matrix-only ``get_task_result(ising_matrix)`` are
    deliberately excluded: using either to guess a submitted task could create
    a second billable task.
    """
    if optimizer_cls is None:
        try:
            _, optimizer_cls, _ = load_kaiwu_sdk()
        except RuntimeError:
            return None
    for name in ("query_task", "query_task_result", "get_task_by_name",
                 "get_task_by_id", "get_task_result_by_name", "get_task_result_by_id"):
        method = getattr(optimizer_cls, name, None)
        if not callable(method):
            continue
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            continue
        if "task_name" in parameters or "task_id" in parameters:
            return name
    return None


class KaiwuClient:
    """Optional real-platform client; construction performs no submission."""

    def __init__(self, project_no=None, wait=True, interval=1):
        self.project_no = project_no or os.environ.get("KAIWU_PROJECT_NO")
        self.wait = bool(wait)
        self.interval = int(interval)
        self._kaiwu, self._optimizer_cls, self._task_mode = load_kaiwu_sdk()

    def sample_hardware_matrix(self, matrix, num_reads, task_name=None):
        num_reads = validate_sampling_reads(num_reads)
        matrix = validate_hardware_matrix(matrix)
        task_name = task_name or build_task_name("matrix", matrix_sha256(matrix))
        optimizer = self._optimizer_cls(
            task_name=task_name,
            wait=self.wait,
            interval=self.interval,
            project_no=self.project_no,
            task_mode=self._task_mode.SAMPLING,
            sample_number=num_reads,
        )
        samples = np.asarray(
            optimizer.solve(
                matrix,
                negtail_flip=False,
                sort_solutions=False,
            ),
            dtype=np.int8,
        )
        if samples.ndim != 2 or samples.shape != (num_reads, matrix.shape[0]):
            raise RuntimeError(
                "Kaiwu returned samples with shape %s; expected (%d, %d)"
                % (samples.shape, num_reads, matrix.shape[0])
            )
        if not np.isin(samples, (-1, 1)).all():
            raise RuntimeError("Kaiwu returned values outside {-1,+1}")
        return {
            "samples": samples,
            "metadata": {
                "backend": "kaiwu_cim_sampling",
                "task_name": task_name,
                "task_id": str(
                    getattr(optimizer, "task_id", getattr(optimizer, "id", ""))
                ),
                "hardware_n_bits": int(matrix.shape[0]),
                "sample_number": int(num_reads),
            },
        }

    def sample_ising(self, h=None, j=None, num_reads=100, beta=1.0,
                     matrix=None, task_name=None, **kwargs):
        """Compatibility method for the existing BosonicPlatformSampler.

        The only accepted h,J form is a 49-spin sampler-form encoding of Q:
        h=0 and j=-2Q. Logical 48-spin h,J is rejected because it has not
        passed the explicit global-gain quantization step.
        """
        if matrix is None:
            if h is None or j is None:
                raise ValueError("Pass matrix=Q or the 49-spin h=0, j=-2Q form")
            h = np.asarray(h, dtype=np.float64)
            j = np.asarray(j, dtype=np.float64)
            if h.shape != (HARDWARE_N_BITS,) or j.shape != (
                HARDWARE_N_BITS,
                HARDWARE_N_BITS,
            ) or not np.allclose(h, 0.0):
                raise ValueError(
                    "KaiwuClient refuses logical 48-spin h,J; provide the "
                    "quantized 49x49 hardware matrix"
                )
            matrix = -0.5 * j
        result = self.sample_hardware_matrix(
            matrix, num_reads=num_reads, task_name=task_name
        )
        samples = result["samples"]
        return {
            "samples": samples,
            "metadata": result["metadata"],
        }


BosonicClient = KaiwuClient
