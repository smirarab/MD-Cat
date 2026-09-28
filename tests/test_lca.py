"""Calibration lookup must not silently change the requested MRCA."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from treeswift import Node, Tree, read_tree_newick
from emd.lca_lib import find_LCAs


ROOT = Path(__file__).resolve().parents[1]


class CalibrationLookupTest(unittest.TestCase):
    def setUp(self):
        self.tree = read_tree_newick('((A:1,B:1)X:1,C:1)R;')

    def test_valid_queries_include_internal_node_names(self):
        nodes = find_LCAs(self.tree, [['A', 'B'], ['A', 'C'], ['A'], ['X']])
        self.assertEqual([node.label for node in nodes], ['X', 'R', 'A', 'X'])

    def test_deep_caterpillar_tree(self):
        tree = Tree()
        tree.root.label = 'N0'
        node = tree.root
        depth = max(3000, sys.getrecursionlimit() + 100)
        for i in range(depth):
            child = Node(label=f'N{i + 1}')
            node.add_child(child)
            node.add_child(Node(label=f'T{i}'))
            node = child

        queries = [
            [f'N{depth}', f'T{depth - 1}'],
            [f'N{depth}', 'T0'],
            [f'T{depth - 1}', 'T1'],
            [f'N{depth}'],
            ['N100', f'N{depth}', 'T100'],
        ]
        nodes = find_LCAs(tree, queries)
        self.assertEqual([node.label for node in nodes],
                         [f'N{depth - 1}', 'N0', 'N1', f'N{depth}', 'N100'])

    def test_missing_names_reject_entire_query(self):
        for query in (['A', 'TYPO'], ['TYPO', 'A'], ['TYPO'],
                      ['A', 'B', 'TYPO'], ['TYPO', 'OTHER']):
            with self.subTest(query=query):
                with self.assertRaises(ValueError) as error:
                    find_LCAs(self.tree, [query])
                self.assertIn('+'.join(query), str(error.exception))
                self.assertIn('not found in the input tree', str(error.exception))
                for name in query:
                    if name in ('TYPO', 'OTHER'):
                        self.assertIn(name, str(error.exception))

    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one node name'):
            find_LCAs(self.tree, [[]])

    def test_cli_fails_before_dating_or_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tree, calibration, output = [folder / name for name in
                                         ('input.tre', 'times.txt', 'output.tre')]
            tree.write_text(self.tree.newick() + '\n')
            for query in ('A+TYPO', 'FOSSIL=A+B+TYPO'):
                with self.subTest(query=query):
                    calibration.write_text(query + ' 2\n')
                    result = subprocess.run(
                        [sys.executable, str(ROOT / 'md_cat.py'), '-i', str(tree),
                         '-t', str(calibration), '-o', str(output), '-b',
                         '-k', '2', '-p', '1', '--maxIter', '1', '--threads', '1'],
                        cwd=ROOT, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn('TYPO', result.stderr)
                    self.assertIn('not found in the input tree', result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertNotIn('Solving EM', result.stdout)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
