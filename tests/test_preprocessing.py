import unittest

import torch
import torchio as tio

from dataset.preprocessing import (
    normalize_target_shape,
    normalize_volume,
    preprocess_volume,
    resize_volume,
)


class PreprocessingTest(unittest.TestCase):
    def test_resize_volume_changes_only_spatial_shape(self):
        image = tio.ScalarImage(tensor=torch.zeros(1, 8, 10, 6))

        resized = resize_volume(image, (12, 14, 4))

        self.assertEqual(resized.shape, (1, 12, 14, 4))

    def test_resize_volume_is_noop_for_matching_shape(self):
        image = tio.ScalarImage(tensor=torch.zeros(1, 8, 10, 6))

        self.assertIs(resize_volume(image, (8, 10, 6)), image)

    def test_none_disables_resize(self):
        image = tio.ScalarImage(tensor=torch.zeros(1, 8, 10, 6))

        self.assertIs(resize_volume(image, None), image)

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_target_shape((512, 512))

    def test_minmax_normalizes_to_unit_range(self):
        values = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) - 4
        image = tio.ScalarImage(tensor=values)

        normalized = normalize_volume(image, "minmax")

        self.assertEqual(normalized.data.amin().item(), 0.0)
        self.assertEqual(normalized.data.amax().item(), 1.0)

    def test_minmax_handles_constant_volume(self):
        image = tio.ScalarImage(tensor=torch.full((1, 2, 2, 2), 42.0))

        normalized = normalize_volume(image, "minmax")

        self.assertTrue(torch.equal(normalized.data, torch.zeros_like(normalized.data)))

    def test_none_accepts_pre_normalized_data(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 2, 2))

        normalized = normalize_volume(image, "none")

        self.assertTrue(torch.allclose(normalized.data, image.data))

    def test_none_rejects_raw_intensities(self):
        image = tio.ScalarImage(tensor=torch.tensor([[[[-1000.0, 1000.0]]]]))

        with self.assertRaisesRegex(ValueError, "requires input data in"):
            normalize_volume(image, "none")

    def test_preprocess_resizes_then_normalizes(self):
        values = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        image = tio.ScalarImage(tensor=values)

        processed = preprocess_volume(image, (4, 4, 2), "minmax")

        self.assertEqual(processed.shape, (1, 4, 4, 2))
        self.assertGreaterEqual(processed.data.amin().item(), 0.0)
        self.assertLessEqual(processed.data.amax().item(), 1.0)


if __name__ == '__main__':
    unittest.main()
