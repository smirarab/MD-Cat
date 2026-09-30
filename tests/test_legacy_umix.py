"""The legacy driver must fail clearly before touching the input tree."""
import unittest
from unittest.mock import Mock

from emd.emd_umix_lib import EM_date_umix


class LegacyUmixTest(unittest.TestCase):
    def test_unsupported_driver_directs_users_to_supported_api(self):
        tree = Mock()
        with self.assertRaisesRegex(NotImplementedError, 'unsupported.*MDCat'):
            EM_date_umix(tree, {})
        self.assertEqual(tree.mock_calls, [])
