import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from build_bank import merge
from guarded_executor import guarded_action
from two_seed_disagreement import sample_two_chunks


class MergeTest(unittest.TestCase):
    def test_four_expert_shards_produce_matched_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite_index, suite in enumerate(("goal", "spatial", "object", "10")):
                source = root / f"libero_{suite}"
                source.mkdir()
                np.savez_compressed(source / "bank_shard.npz",
                                    features=np.full((10, 2), suite_index, dtype=np.float32),
                                    states=np.full((10, 7), suite_index, dtype=np.float32))
                (source / "bank_shard_manifest.json").write_text(json.dumps({
                    "source_name": source.name, "source_kind": "lerobot",
                    "checkpoint_sha256": "same-checkpoint", "entries": 10,
                }))
                (source / "bank_shard.jsonl").write_text("".join(
                    json.dumps({"row": index, "canonical_task": f"task-{suite_index}-{index}",
                                "source_name": source.name, "episode_index": index}) + "\n"
                    for index in range(10)
                ))

            merge(SimpleNamespace(output_dir=root, checkpoint=None))
            with np.load(root / "success_id_bank.npz") as bank:
                self.assertEqual(bank["features"].shape, (40, 2))
                self.assertEqual(bank["states"].shape, (40, 7))
                self.assertEqual(bank["features"][:, 0].tolist(), [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
            rows = [json.loads(line) for line in (root / "success_id_bank.jsonl").read_text().splitlines()]
            self.assertEqual([row["row"] for row in rows], list(range(40)))
            summary = json.loads((root / "success_id_bank_summary.json").read_text())
            self.assertEqual(summary["entries"], 40)
            self.assertEqual({item["checkpoint_sha256"] for item in summary["source_shards"]}, {"same-checkpoint"})

            manifest = root / "libero_10/bank_shard_manifest.json"
            data = json.loads(manifest.read_text())
            data["checkpoint_sha256"] = "different-checkpoint"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                merge(SimpleNamespace(output_dir=root, checkpoint=None))


class TwoSeedTest(unittest.TestCase):
    def test_k1_sampling_replays_seeds_and_shares_prior(self):
        response = {
            "state_servo_prior": [0.1] * 7,
            "jepa_residual_action_chunk": [[0.2] * 7],
            "action_log_std_chunk": [[-2.0] * 7],
        }
        chunks, metrics = sample_two_chunks(response, (7, 11))
        repeated, _ = sample_two_chunks(response, (7, 11))
        self.assertEqual([chunk.shape for chunk in chunks], [(1, 7), (1, 7)])
        for first, second in zip(chunks, repeated):
            np.testing.assert_array_equal(first, second)
        self.assertGreater(metrics["xyz_l2_m"], 0.0)
        with self.assertRaisesRegex(ValueError, "distinct"):
            sample_two_chunks(response, (7, 7))


class GuardedExecutorTest(unittest.TestCase):
    def test_accept_and_fallback_preserve_servo_gripper(self):
        calibration = {
            "native_steps": 5, "meters_per_native_command": [0.01] * 3,
            "rotation_gain_matrix": np.eye(3).tolist(), "max_native_command": 1.0,
            "max_translation_delta": 0.03, "max_rotation_delta": 0.05,
            "min_translation_cosine": 0.5, "min_rotation_cosine": 0.0,
            "max_correction_ratio": 0.5,
        }
        health = {"guarded_native_config": calibration, "bank_checkpoint_sha256": "same",
                  "intent_state_servo_prior": "state_goal_delta_guarded_single_tick_v7"}
        recovery = {"status": "ok", "residual_representation": "executed_correction",
                    "residual_gain_applied": False, "residual_execution_mode": "guarded_single_tick",
                    "bank_checkpoint_sha256": "same", "state_servo_delta": [0.02, 0, 0, 0, 0, 0, 0.4],
                    "jepa_residual_action": [0.001, 0, 0, 0, 0, 0, -1.0]}
        state = [0, 0, 0, 0, 0, 0, 0.2]
        action, diagnostics = guarded_action(recovery, state, health)
        self.assertTrue(diagnostics["accepted"])
        np.testing.assert_allclose(action[[0, 6]], [0.5, 1.0])

        recovery["jepa_residual_action"][0] = -0.001
        action, diagnostics = guarded_action(recovery, state, health)
        self.assertEqual(diagnostics["translation"]["reason"], "direction_mismatch")
        np.testing.assert_allclose(action[[0, 6]], [0.4, 1.0])

        recovery["jepa_residual_action"][0] = 0.003
        action, diagnostics = guarded_action(recovery, state, health)
        self.assertEqual(diagnostics["translation"]["reason"], "correction_too_large")
        np.testing.assert_allclose(action[[0, 6]], [0.4, 1.0])

        health["bank_checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint/bank"):
            guarded_action(recovery, state, health)


class ContractTest(unittest.TestCase):
    def test_training_and_server_guarded_prior_match(self):
        import torch

        from app.vjepa_intent_action.train import _state_servo_prior
        from vjepa_recovery_service import RecoveryEngine

        current = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        goal = np.array([0.3, -0.2, 0.4, -0.1, 0.15, 0.5, 0.9], dtype=np.float32)
        expected = _state_servo_prior(
            torch.as_tensor(current).unsqueeze(0), torch.as_tensor(goal).unsqueeze(0),
            1, 5, "state_goal_delta_fractional_so3_by_condition_horizon_v4",
        )[0, 0].numpy()
        engine = RecoveryEngine.__new__(RecoveryEngine)
        engine.intent_state_servo_prior = "state_goal_delta_guarded_single_tick_v7"
        engine.intent_guarded_native_steps = 5
        actual = engine._state_servo_prior(current, goal)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()