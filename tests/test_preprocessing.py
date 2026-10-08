import unittest

import torch
import torchio as tio

from dataset.preprocessing import normalize_target_shape, resize_volume


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


if __name__ == '__main__':
    unittest.main()
