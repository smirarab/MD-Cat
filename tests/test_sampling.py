"""Packaged sampling workflow regression and end-to-end tests."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from treeswift import read_tree_nexus, read_tree_newick
from emd.calibration import bottom_up_draws, convert, sample_to_folder
from emd import emd_normal_lib as engine

ROOT = Path(__file__).resolve().parents[1]


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mdcat sampling ')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.tree = self.path/'input.tre'
        self.tree.write_text('(A:0.1,B:0.2);\n')
        self.config = self.path/'input.config'
        self.config.write_text('numsites = 1234\nmrca = ROOT A B\nmin = ROOT 1\nmax = ROOT 2\n')
        self.work = self.path/'work'
        self.output = self.path/'summary.nex'

    def cli(self, *args, ok=True):
        result = subprocess.run([sys.executable, '-m', 'emd.sample', *map(str,args)],
                                cwd=ROOT, text=True, capture_output=True, timeout=90)
        if ok:
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0)
        return result

    def prepare(self, *extra):
        self.cli('-i',self.tree,'-t',self.config,'-o',self.output,'--workdir',self.work,
                 '-S','2','-p','1','-k','2','--maxIter','2','--randSeed','42',
                 '--min-branch','.02','--dry-run',*extra)
        return json.loads((self.work/'manifest.json').read_text())

    def test_defaults_and_dry_run(self):
        plan=self.prepare()
        self.assertEqual(plan['options']['seqLen'],1234)
        self.assertEqual(plan['options']['strategy'],'bottom-up')
        self.assertEqual(plan['options']['distribution'],'exponential')
        self.assertEqual(len(plan['jobs']),2)
        self.assertFalse(self.output.exists())
        self.assertFalse((Path(plan['jobs'][0]['run'])/'fitted.tre').exists())
        self.assertNotEqual(plan['jobs'][0]['fit_seed'],plan['jobs'][1]['fit_seed'])
        self.assertNotEqual(plan['jobs'][0]['fit_seed'],plan['jobs'][0]['ci_seed'])
        with self.assertRaises(ValueError):
            convert('mrca = R A B\nmin = R -1\nmax = R 2')
        for flag in ('-b','-r','-f','-d','--clean','--clip-negative'):
            self.cli(flag,ok=False)
        changed=self.cli('--workdir',self.work,'--resume','-p','2',ok=False)
        self.assertIn('conflicts',changed.stderr)
        self.assertIn('conflicts',self.cli('--workdir',self.work,'--resume','-S3',ok=False).stderr)

    def test_external_partial_resume_and_ci(self):
        plan=self.prepare('--CI','2 0.025 0.975')
        command=plan['jobs'][0]['command']
        self.assertEqual(command[command.index('--CI')+1],'2 0 1')
        # Execute an actual generated job script with whitespace in its path.
        result=subprocess.run(['sh',str(self.work/'jobs/sample_001.sh')],
                              cwd=ROOT,text=True,capture_output=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.cli('--workdir',self.work,'--summarize-only')
        partial=json.loads(Path(str(self.output)+'.json').read_text())
        self.assertEqual(partial['included_jobs'],[1])
        self.assertEqual(partial['n_ci'],2)
        self.assertEqual(len(partial['excluded_jobs']),1)
        first=Path(plan['jobs'][0]['run'])/'fitted.tre'
        before=first.stat().st_mtime_ns
        self.cli('--workdir',self.work,'--resume','--jobs','2')
        self.assertEqual(first.stat().st_mtime_ns,before)
        meta=json.loads(Path(str(self.output)+'.json').read_text())
        self.assertEqual(meta['n_ci'],4)
        self.assertEqual(meta['included_jobs'],[1,2])
        tree=read_tree_nexus(str(self.output))['summary']
        self.assertIn('height_CI={',tree.root.node_params)
        self.assertEqual(len((self.work/'fitted.trees').read_text().splitlines()),2)
        # Truncation excludes the whole run, including its central estimate.
        ci=Path(plan['jobs'][1]['run'])/'ci-replicates.tre'
        ci.write_text(ci.read_text().splitlines()[0]+'\n')
        self.cli('--workdir',self.work,'--summarize-only')
        meta=json.loads(Path(str(self.output)+'.json').read_text())
        self.assertEqual(meta['included_jobs'],[1])
        self.assertEqual(meta['n_ci'],2)
        # Resume repairs the incomplete job, while preserving the first job.
        self.cli('--workdir',self.work,'--resume')
        self.assertEqual(first.stat().st_mtime_ns,before)
        self.assertEqual(len(ci.read_text().splitlines()),2)

    def test_parallel_no_ci_and_override(self):
        self.cli('-i',self.tree,'-t',self.config,'-o',self.output,'--workdir',self.work,
                 '-S','2','-p','1','-k','2','--maxIter','2','--randSeed','123',
                 '--strategy','bottom-up','--distribution','uniform','--jobs','2',
                 '-l','321','--format','clean')
        plan=json.loads((self.work/'manifest.json').read_text())
        self.assertEqual(plan['options']['seqLen'],321)
        self.assertNotIn('[',self.output.read_text())
        meta=json.loads(Path(str(self.output)+'.json').read_text())
        self.assertEqual(meta['n_ci'],0)
        self.assertEqual(meta['n_runs'],2)
        with (Path(plan['jobs'][0]['calibration'])).open('a') as f:
            f.write('\n')
        self.assertIn('calibration input changed',self.cli('--workdir',self.work,'--resume',ok=False).stderr)

    def test_bottom_up_distribution_and_reproducibility(self):
        size=25; lower=np.array([1.,2.]); upper=np.array([10.,7.])
        draws,valid=bottom_up_draws(np.random.default_rng(17),size,lower,upper,[0],[1],[.003])
        rng=np.random.default_rng(17)
        children=2+rng.exponential(np.full(size,5/np.log(20)))
        offset=np.maximum(1,children+.003)
        parents=offset+rng.exponential(np.maximum(10-offset,0)/np.log(20))
        np.testing.assert_array_equal(draws,np.column_stack((parents,children)))
        np.testing.assert_array_equal(valid,(children<=7)&(offset<=10)&(parents<=10))
        _,metadata=convert(self.config.read_text())
        for mode in ('independent','bottom-up'):
            for dist in ('uniform','exponential'):
                for repeat in (1,2):
                    with redirect_stdout(io.StringIO()):
                        sample_to_folder(metadata,self.tree,self.path/f'{mode}{dist}{repeat}',3,42,10000,.00001,dist,False,mode)
                for number in range(1,4):
                    name=f'sample_{number:03d}.txt'
                    self.assertEqual((self.path/f'{mode}{dist}1'/name).read_text(),(self.path/f'{mode}{dist}2'/name).read_text())

    def test_exact_calibration_at_minimum_duration(self):
        _,metadata=convert('mrca = ROOT A B\nmin = ROOT 0.001\nmax = ROOT 0.001\n')
        folder=self.path/'exact'
        sample_to_folder(metadata,self.tree,folder,2,42,10,.001)
        for f in folder.glob('sample_*.txt'):
            self.assertEqual(float(f.read_text().split()[1]),.001)
        self.assertEqual(json.loads((folder/'manifest.json').read_text())['sampling_min_branch'],.001)

    def test_min_branch_reaches_initialization_optimizer_and_ci(self):
        calls=[]
        original_em=engine.EM_date
        original_ci=engine.get_confidence_interval
        original_init=engine.init_EM
        def em(*args,**kw):
            calls.append(('em',kw['eps_tau']))
            return original_em(*args,**kw)
        def ci(*args,**kw):
            calls.append(('ci',kw['eps_tau']))
            return original_ci(*args,**kw)
        def init(*args,**kw):
            calls.append(('init',kw['eps_tau']))
            return original_init(*args,**kw)
        with patch.object(engine,'EM_date',em),patch.object(engine,'get_confidence_interval',ci),patch.object(engine,'init_EM',init),redirect_stdout(io.StringIO()):
            result=engine.MDCat(read_tree_newick('(A:.1,B:.2);'),2,nrep=1,maxIter=2,randseed=1,
                                min_branch=.02,CI_options=dict(nboots=2,p_lower=0,p_upper=1),threads=1)
        self.assertTrue(calls)
        self.assertEqual({c[0] for c in calls},{'em','init','ci'})
        self.assertTrue(all(c[1]==.02 for c in calls))
        self.assertIsNotNone(result[0])


if __name__ == '__main__':
    unittest.main()
