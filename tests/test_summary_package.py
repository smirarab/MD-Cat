"""Run with: python -m unittest discover -s tests -p test_summarize_mdcat.py"""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import treeswift
from emd.summary import summarize


class TestPooledCI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = [self.write('a.tre', '((a:2,b:2):3,c:5);'),
                     self.write('b.tre', '(c:9,(b:4,a:4):5);')]
        self.cis = [self.write('a.CIreplicates', '((a:1,b:1):3,c:4);\n((a:3,b:3):3,c:6);'),
                    self.write('b.CIreplicates', '(c:8,(b:5,a:5):3);\n(c:10,(b:7,a:7):3);')]

    def write(self, name, text):
        path = self.root/name
        path.write_text(text)
        return path

    def test_pooling_and_central_estimates(self):
        for q, expected in [(None, 7), (.5, 7), (.25, 6)]:
            output = self.root/f'{q}.tre'
            summarize(self.runs, output, quantile=q, ci_files=self.cis, ci=(.1,.9))
            with Path(str(output)+'.tsv').open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            root = next(r for r in rows if r['n_tips']=='3')
            self.assertEqual(float(root['age']), expected)
            np.testing.assert_allclose([float(root['t_lower']),float(root['t_upper'])],
                                       np.quantile([4,6,8,10],[.1,.9]))
            self.assertEqual(root['n_ci'], '4')
            tree = treeswift.read_tree_nexus(str(output))['summary']
            self.assertIn('t_ci={', tree.root.node_params)
            self.assertIn('height_CI={', tree.root.node_params)
            self.assertIn('CI_method=pooled_percentile', tree.root.node_params)
            self.assertNotIn('HPD', output.read_text())
            self.assertIsNone(tree.root.label)
            data=json.loads(Path(str(output)+'.json').read_text())
            self.assertEqual(data['ci_replicates_per_file'],[2,2])

    def test_clean_retains_ci_table(self):
        out=self.root/'clean.tre'
        summarize(self.runs,out,clean=True,ci_files=self.cis,ci=(0,1))
        tree = treeswift.read_tree_nexus(str(out))['summary']
        self.assertTrue(all(not hasattr(n,'node_params') for n in tree.traverse_postorder()))
        self.assertIn('t_lower',Path(str(out)+'.tsv').read_text())

    def test_invalid_inputs(self):
        for ci, files in [((.9,.1),self.cis), ((0,1),self.cis[:1]),
                          ((0,1),[self.cis[0]]*2), (None,self.cis)]:
            with self.assertRaises(ValueError):
                summarize(self.runs,self.root/'bad.tre',ci_files=files,ci=ci)
        bad=self.write('bad.CIreplicates','((a:2,c:2):3,b:5);')
        with self.assertRaisesRegex(ValueError,'topology'):
            summarize(self.runs,self.root/'bad.tre',ci_files=[self.cis[0],bad],ci=(0,1))
        self.assertFalse((self.root/'bad.tre').exists())

    def test_quoted_mdcat_annotations(self):
        quoted=self.write('quoted.tre', "(('a[t=0,mu=0.1]':2,'b[t=0,mu=0.2]':2)'X[t=2,mu=1]':3,'c[t=0,mu=3]':5)'R[t=5]';")
        quoted_ci=self.write('quoted.CIreplicates', "(('a[t=0,mu=7]':1,'b[t=0,mu=8]':1)'X[t=1]':3,'c[t=0,mu=9]':4)'R[t=4]';")
        out=self.root/'quoted.nex'
        summarize([quoted,self.runs[1]],out,ci_files=[quoted_ci,self.cis[1]],ci=(0,1))
        tree=treeswift.read_tree_nexus(str(out))['summary']
        self.assertEqual({n.label for n in tree.traverse_leaves()},{'a','b','c'})
        collision=self.write('collision.tre', "('a[t=0,mu=1]':1,'a[t=0,mu=2]':1);")
        with self.assertRaisesRegex(ValueError,'duplicate tip'):
            summarize([collision],self.root/'collision.nex')

    def test_negative_clipping(self):
        bad=self.write('negative.tre','((a:2,b:2):-0.002,c:1.998);')
        with self.assertRaisesRegex(ValueError,'negative'):
            summarize([bad],self.root/'strict.nex',clip_negative=False)
        out=self.root/'clipped.nex'
        summarize([bad],out,clip_negative=True)
        data=json.loads(Path(str(out)+'.json').read_text())
        self.assertEqual(data['clipped_edges'],1)
        self.assertEqual(data['clipping'][0]['minimum_length'],-.002)
        tree=treeswift.read_tree_nexus(str(out))['summary']
        self.assertTrue(all(n.edge_length is None or n.edge_length>=0 for n in tree.traverse_postorder()))


if __name__ == '__main__':
    unittest.main()
