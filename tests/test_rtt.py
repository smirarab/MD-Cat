"""RTT compatibility, numerical stability, and traversal complexity."""
import unittest
from unittest.mock import patch

from treeswift import Node, Tree, read_tree_newick

from emd.rtt_lib import optimize_rtt, rtt_mu


def legacy_fit(root, times, pseudo=0):
    distances = {root: 0.}
    pairs = []
    for node in root.traverse_preorder():
        if node is not root:
            distances[node] = distances[node.parent] + node.edge_length + pseudo
        if node.label in times:
            pairs.append((distances[node], times[node.label]))
    n = len(pairs)
    st = sum(t for b, t in pairs)
    stt = sum(t*t for b, t in pairs)
    if st*st == n*stt:
        return None, None, None
    sb = sum(b for b, t in pairs)
    sbt = sum(b*t for b, t in pairs)
    mu = max(.001, (sb*st-n*sbt)/(st*st-n*stt))
    t0 = (st-sb/mu)/n
    return mu, t0, sum((b-mu*(t-t0))**2 for b, t in pairs)


class RTTTest(unittest.TestCase):
    def test_matches_legacy_regressions_and_unweighted_average(self):
        for newick in ('((A:.2,B:.9)X:.4,(C:.5,D:.1)Y:.3,E:.8)R;',
                       '((((A:.2,B:.9)X:.4,C:.5)Y:.3,D:.1)Z:.2,E:.8)R;'):
            tree = read_tree_newick(newick)
            for times in ({'R': 0., 'X': 1., 'A': 2., 'B': 3., 'C': 4., 'D': 2., 'E': 5.},
                          {'A': 5., 'B': 4., 'C': 3., 'D': 2., 'E': 1.},
                          {'R': 0., 'A': 1., 'B': 1., 'C': 1.}):
                slopes = []
                for node in tree.traverse_postorder():
                    expected = legacy_fit(node, times)
                    if expected[0] is not None:
                        slopes.append(expected[0])
                    for pseudo in (0., .02):
                        expected = legacy_fit(node, times, pseudo)
                        actual = optimize_rtt(node, times, pseudo)
                        if expected[0] is None:
                            self.assertEqual(actual, expected)
                        else:
                            for a, b in zip(actual, expected):
                                self.assertAlmostEqual(a, b, places=11)
                self.assertAlmostEqual(rtt_mu(tree, times), sum(slopes)/len(slopes), places=12)

    def test_large_time_origin(self):
        tree = read_tree_newick('((A:2,B:4)X:1,C:8)R;')
        times = {'R': 0., 'X': 1., 'A': 2., 'B': 3., 'C': 5.}
        shifted = {label: t+1e12 for label, t in times.items()}
        self.assertAlmostEqual(rtt_mu(tree, times), rtt_mu(tree, shifted), places=12)
        mu, t0, score = optimize_rtt(tree.root, times)
        shifted_mu, shifted_t0, shifted_score = optimize_rtt(tree.root, shifted)
        self.assertAlmostEqual(mu, shifted_mu, places=12)
        self.assertAlmostEqual(score, shifted_score, places=12)
        self.assertAlmostEqual(t0, shifted_t0-1e12, delta=.0001)

    def test_degenerate_times_and_rate_floor(self):
        tree = read_tree_newick('(A:1,B:2)R;')
        for times in ({}, {'A': 1.}, {'A': 1., 'B': 1.}):
            self.assertEqual(optimize_rtt(tree.root, times), (None, None, None))
            with self.assertRaises(ZeroDivisionError):
                rtt_mu(tree, times)
        times = {'A': 2., 'B': 1.}
        self.assertEqual(optimize_rtt(tree.root, times)[0], .001)
        self.assertEqual(rtt_mu(tree, times), .001)

    def test_deep_tree_uses_one_postorder_and_no_subtree_regressions(self):
        tree = Tree()
        tree.root.label = 'R'
        node = tree.root
        times = {'R': 0.}
        depth = 4000
        for i in range(depth):
            child = Node(label=f'N{i}', edge_length=1.)
            tip = Node(label=f'T{i}', edge_length=1.)
            node.add_child(child)
            node.add_child(tip)
            times[tip.label] = float(i+1)
            node = child
        times[node.label] = float(depth)
        visited = []
        original = tree.traverse_postorder

        def counted():
            for node in original():
                visited.append(node)
                yield node

        with patch.object(tree, 'traverse_postorder', side_effect=counted) as traversal, \
             patch('emd.rtt_lib.optimize_rtt', side_effect=AssertionError('subtree regression')), \
             patch.object(Node, 'traverse_preorder', side_effect=AssertionError('subtree walk')):
            self.assertAlmostEqual(rtt_mu(tree, times), 1., places=12)
        traversal.assert_called_once_with()
        self.assertEqual(len(visited), 2*depth+1)


if __name__ == '__main__':
    unittest.main()
