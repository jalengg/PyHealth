"""Tests for HALO model inheriting BaseModel.

TDD tests written before the implementation. All tests use a mock SampleDataset
built with NestedSequenceProcessor — no MIMIC-III data required.

Uses importlib to load pyhealth.models.base_model and pyhealth.models.generators.halo
directly without triggering pyhealth.models.__init__.py (which requires optional
heavy dependencies like litdata, pyarrow, einops not present in the test venv).
"""

import importlib.util
import sys
import types
import unittest
from unittest.mock import MagicMock


def _bootstrap_imports():
    """Set up sys.modules so BaseModel and HALO can be imported cleanly.

    The test venv only has torch and polars installed. Heavy deps declared in
    pyproject.toml (litdata, pyarrow, einops, ...) are not present. This
    function stubs out exactly the modules needed to load base_model.py and
    halo.py without hitting missing-import errors from unrelated models.
    """
    # Import top-level pyhealth package (safe -- __init__.py has no heavy deps)
    import pyhealth  # noqa: F401

    # Stub pyhealth.datasets so that base_model.py's
    # "from ..datasets import SampleDataset" resolves cleanly.
    if "pyhealth.datasets" not in sys.modules:
        ds_stub = MagicMock()
        class _FakeSampleDataset:  # noqa: N801
            pass
        ds_stub.SampleDataset = _FakeSampleDataset
        sys.modules["pyhealth.datasets"] = ds_stub

    # Stub pyhealth.models package so we can control what gets loaded
    # without running the real __init__.py (which requires einops, litdata, etc.)
    if "pyhealth.models" not in sys.modules or isinstance(
        sys.modules["pyhealth.models"], MagicMock
    ):
        models_stub = MagicMock()
        sys.modules["pyhealth.models"] = models_stub
    else:
        models_stub = sys.modules["pyhealth.models"]

    # Load pyhealth.processors (clean — no heavy deps)
    from pyhealth.processors import PROCESSOR_REGISTRY  # noqa: F401

    # Load base_model.py directly via importlib
    def _load_file(mod_name, filepath):
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    base = "/u/jalenj4/PyHealth/pyhealth/models"
    res = "/u/jalenj4/PyHealth/pyhealth/models/generators/halo_resources"

    # Load base_model
    bm_mod = _load_file("pyhealth.models.base_model", f"{base}/base_model.py")
    BaseModel = bm_mod.BaseModel

    # Make BaseModel accessible via the stub so halo.py can import it
    models_stub.BaseModel = BaseModel

    # Load halo_resources sub-modules
    _load_file(
        "pyhealth.models.generators.halo_resources.halo_config",
        f"{res}/halo_config.py",
    )
    _load_file(
        "pyhealth.models.generators.halo_resources.halo_model",
        f"{res}/halo_model.py",
    )

    # Stub the generators package so relative imports in halo.py work
    gen_stub = MagicMock()
    sys.modules.setdefault("pyhealth.models.generators", gen_stub)

    # Load halo.py directly
    halo_mod = _load_file(
        "pyhealth.models.generators.halo",
        f"{base}/generators/halo.py",
    )

    return BaseModel, halo_mod.HALO


BaseModel, HALO = _bootstrap_imports()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from pyhealth.processors.nested_sequence_processor import NestedSequenceProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_mock_dataset():
    """Build a minimal mock SampleDataset with a fitted NestedSequenceProcessor."""
    samples = [
        {"patient_id": "p1", "visits": [["A", "B"], ["C", "D"], ["E"]]},
        {"patient_id": "p2", "visits": [["A", "C"], ["B", "E"], ["D", "F"]]},
        {"patient_id": "p3", "visits": [["G", "H"], ["I", "J"], ["A", "B"]]},
    ]
    processor = NestedSequenceProcessor()
    processor.fit(samples, "visits")

    dataset = MagicMock()
    dataset.input_schema = {"visits": "nested_sequence"}
    dataset.output_schema = {}
    dataset.input_processors = {"visits": processor}
    dataset.output_processors = {}
    return dataset, processor


def _make_visits_tensor(processor, batch_size=2, n_visits=3):
    """Build a padded visits tensor of shape (batch, n_visits, max_inner_len).

    Only writes to positions that exist given the provided n_visits and batch_size.
    """
    max_inner = processor._max_inner_len
    visits = torch.zeros(batch_size, n_visits, max_inner, dtype=torch.long)

    # Batch item 0 — fill up to min(3, n_visits) visits
    if n_visits > 0:
        visits[0, 0, 0] = processor.code_vocab.get("A", 0)
        if max_inner > 1:
            visits[0, 0, 1] = processor.code_vocab.get("B", 0)
    if n_visits > 1:
        visits[0, 1, 0] = processor.code_vocab.get("C", 0)
    if n_visits > 2:
        visits[0, 2, 0] = processor.code_vocab.get("E", 0)

    # Batch item 1 (only if batch_size > 1)
    if batch_size > 1:
        if n_visits > 0:
            visits[1, 0, 0] = processor.code_vocab.get("A", 0)
            if max_inner > 1:
                visits[1, 0, 1] = processor.code_vocab.get("C", 0)
        if n_visits > 1:
            visits[1, 1, 0] = processor.code_vocab.get("B", 0)

    return visits


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestHALOInheritsBaseModel(unittest.TestCase):
    """Structural and inheritance tests for the HALO class."""

    def setUp(self):
        self.dataset, self.processor = _make_mock_dataset()
        # Use tiny dimensions to keep tests fast
        self.model = HALO(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            n_ctx=12,
        )

    def test_halo_inherits_basemodel(self):
        """HALO must be a subclass of BaseModel."""
        self.assertIsInstance(self.model, BaseModel)

    def test_halo_is_nn_module(self):
        """HALO must be a subclass of nn.Module."""
        self.assertIsInstance(self.model, nn.Module)

    def test_halo_feature_keys(self):
        """feature_keys must be ['visits'] from input_schema."""
        self.assertEqual(self.model.feature_keys, ["visits"])

    def test_halo_label_keys_empty(self):
        """label_keys must be [] because output_schema is empty."""
        self.assertEqual(self.model.label_keys, [])

    def test_halo_has_device_property(self):
        """model.device must return a torch.device instance."""
        self.assertIsInstance(self.model.device, torch.device)

    def test_halo_has_halo_model_submodule(self):
        """self.halo_model must exist as a registered nn.Module submodule."""
        self.assertTrue(hasattr(self.model, "halo_model"))
        self.assertIsInstance(self.model.halo_model, nn.Module)

    def test_halo_config_vocab_sizes(self):
        """config.code_vocab_size must equal processor.vocab_size()."""
        self.assertEqual(
            self.model.config.code_vocab_size,
            self.processor.vocab_size(),
        )

    def test_halo_config_label_vocab_size_zero(self):
        """config.label_vocab_size must be 0 (no labels)."""
        self.assertEqual(self.model.config.label_vocab_size, 0)

    def test_halo_config_total_vocab_size(self):
        """total_vocab_size == code_vocab_size + label_vocab_size + 3."""
        expected = self.processor.vocab_size() + 0 + 3
        self.assertEqual(self.model.config.total_vocab_size, expected)

    def test_halo_halo_model_is_registered_submodule(self):
        """halo_model must appear in named_modules() so it participates in state_dict."""
        module_names = [name for name, _ in self.model.named_modules()]
        self.assertIn("halo_model", module_names)

    def test_halo_parameters_non_empty(self):
        """Model must have trainable parameters (from halo_model)."""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)


class TestHALOForward(unittest.TestCase):
    """Tests for HALO.forward() method."""

    def setUp(self):
        self.dataset, self.processor = _make_mock_dataset()
        self.model = HALO(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            n_ctx=12,
        )
        self.visits_tensor = _make_visits_tensor(self.processor, batch_size=2, n_visits=3)

    def test_halo_forward_returns_dict(self):
        """forward() must return a dict."""
        output = self.model(visits=self.visits_tensor)
        self.assertIsInstance(output, dict)

    def test_halo_forward_returns_loss(self):
        """forward() must have 'loss' key in output."""
        output = self.model(visits=self.visits_tensor)
        self.assertIn("loss", output)

    def test_halo_forward_loss_is_tensor(self):
        """loss must be a torch.Tensor."""
        output = self.model(visits=self.visits_tensor)
        self.assertIsInstance(output["loss"], torch.Tensor)

    def test_halo_forward_loss_is_scalar(self):
        """loss must be a scalar (0-dimensional tensor)."""
        output = self.model(visits=self.visits_tensor)
        self.assertEqual(output["loss"].dim(), 0)

    def test_halo_forward_returns_predictions(self):
        """forward() must also return 'predictions' key."""
        output = self.model(visits=self.visits_tensor)
        self.assertIn("predictions", output)

    def test_halo_forward_shape_batch2_visits3(self):
        """forward() with (2, 3, max_codes) visits tensor -> scalar loss."""
        output = self.model(visits=self.visits_tensor)
        self.assertEqual(output["loss"].shape, torch.Size([]))

    def test_halo_train_mode_forward(self):
        """In training mode, forward() must return loss."""
        self.model.train()
        output = self.model(visits=self.visits_tensor)
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)

    def test_halo_eval_mode_forward(self):
        """In eval mode with no_grad, forward() must still return loss."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(visits=self.visits_tensor)
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)

    def test_halo_forward_loss_is_finite(self):
        """loss must be a finite number (not NaN or Inf)."""
        output = self.model(visits=self.visits_tensor)
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_halo_forward_batch_size_1(self):
        """forward() must work with batch_size=1."""
        visits_1 = _make_visits_tensor(self.processor, batch_size=1, n_visits=2)
        output = self.model(visits=visits_1)
        self.assertIn("loss", output)
        self.assertEqual(output["loss"].dim(), 0)

    def test_halo_forward_accepts_kwargs(self):
        """forward() must accept extra kwargs without crashing (e.g., patient_id)."""
        output = self.model(visits=self.visits_tensor, patient_id=["p1", "p2"])
        self.assertIn("loss", output)


class TestHALOInit(unittest.TestCase):
    """Tests for HALO.__init__ parameter handling."""

    def setUp(self):
        self.dataset, self.processor = _make_mock_dataset()

    def test_save_dir_stored(self):
        """save_dir parameter must be stored."""
        model = HALO(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            n_ctx=12,
            save_dir="/tmp/test_halo/",
        )
        self.assertEqual(model.save_dir, "/tmp/test_halo/")

    def test_visits_processor_stored(self):
        """visits_processor must be stored as a reference."""
        model = HALO(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            n_ctx=12,
        )
        self.assertTrue(hasattr(model, "visits_processor"))

    def test_config_n_ctx(self):
        """config.n_ctx must reflect the n_ctx parameter."""
        model = HALO(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            n_ctx=10,
        )
        self.assertEqual(model.config.n_ctx, 10)

    def test_config_embed_dim(self):
        """config.n_embd must reflect the embed_dim parameter."""
        model = HALO(
            dataset=self.dataset,
            embed_dim=32,
            n_heads=2,
            n_layers=2,
            n_ctx=12,
        )
        self.assertEqual(model.config.n_embd, 32)

    def test_config_epochs(self):
        """config.epoch must reflect the epochs parameter."""
        model = HALO(
            dataset=self.dataset,
            embed_dim=16,
            n_heads=2,
            n_layers=2,
            n_ctx=12,
            epochs=10,
        )
        self.assertEqual(model.config.epoch, 10)


if __name__ == "__main__":
    unittest.main()
