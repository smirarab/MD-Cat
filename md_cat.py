#! /usr/bin/env python

import argparse
import sys
import time
import os

def positive_int(value):
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer")
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def ci_options(value):
    tokens = value.split()
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError(
            'expected three space-separated numbers in quotes, e.g. --CI "100 0.025 0.975"')
    try:
        nboots = positive_int(tokens[0])
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError("N must be a positive integer")
    try:
        lower, upper = map(float, tokens[1:])
    except ValueError:
        raise argparse.ArgumentTypeError("LOWER and UPPER must be numeric quantiles")
    if not 0 <= lower < upper <= 1:
        raise argparse.ArgumentTypeError(
            "quantiles must satisfy 0 <= LOWER < UPPER <= 1; "
            'for a 95% interval use --CI "100 0.025 0.975"')
    return {'nboots': nboots, 'p_lower': lower, 'p_upper': upper}


parser = argparse.ArgumentParser()

parser.add_argument("-i","--input",required=True,help="Input trees")
parser.add_argument("-t","--samplingTime",required=False,help="Sampling times / Calibration points. Will override -r and -f whenever conflicts occur. Default: None")
parser.add_argument("-k","--ncat",required=False,help="The number of rate categories. Default: 50")
parser.add_argument("-r","--rootTime",required=False,help="Divergence time at root. Will be overrided by -t if conflict occurs. Default: if -t is used, set root time to None. Otherwise, set root time to 1 if -b is used else 0.")
parser.add_argument("-f","--leafTime",required=False,help="Divergence time at leaves. Will be overrided by -t whenever conflicts occur. To be used with either -r or -b to produce ultrametric tree. Default: if -b is off, set leaf times to None if -t is used else 1. If -b is on, set leaf times to 0 and allow -t to override it.")
parser.add_argument("-b","--backward",action='store_true',help="Use backward time and enforce ultrametricity. This option is useful for fossil calibrations with present-time sequences. Default: NO") 
parser.add_argument("-d","--asDate",action='store_true',help="Read and write divergence times as date format (YYY-mm-dd). If it is used, the output tree has branch lengths in days and divergence times shown as date (YYYY-mm-dd). Note: this option cannot be used with -b and will override -b if both are used. Default: FALSE")
parser.add_argument("-o","--output",required=False,help="The output trees with branch lengths in time unit. Default: [input].emDate")
parser.add_argument("-p","--rep",required=False, help="The number of random replicates for initialization. Default: 100")
parser.add_argument("-l","--seqLen",required=False, help="The length of the sequences. Default: 1000")
parser.add_argument("-v","--verbose",action='store_true',help="Verbose")
parser.add_argument("--maxIter",required=False,help="The maximum number of iterations for EM search. Default: 100")
parser.add_argument("--CI", required=False, type=ci_options, metavar='"N LOWER UPPER"',
                    help='Estimate confidence intervals for branch lengths. Pass three space-separated numbers inside one pair of quotes (no commas): N is a positive integer number of posterior mutation-rate samples; quantiles must satisfy 0 <= LOWER < UPPER <= 1 (not percentages; 0 and 1 select the minimum and maximum). Example: --CI "100 0.025 0.975" uses 100 samples for a 95%% confidence interval. Default: off.')
parser.add_argument("--CI-samples", metavar="FILE",
                    help="Write every CI replicate to FILE, one Newick tree per line, with sampled time branch lengths and node time/rate annotations. Requires --CI; N samples produce N trees.")
#parser.add_argument("--nSamplings",required=False,type=int,default=100,help="The number of repeating samplings on mutation rate posteriors to estimate the confidence interval for branch lengths. Default: 100")
parser.add_argument("--randSeed",required=False,help="Random seed; either a number or a list of p numbers where p is the number of replicates specified by -p. Default: auto-select")
parser.add_argument("--annotate",required=False,help="Annotation option. Select one of these options: 1: Annotate divergent times; 2: Annotate divergent times and expected mutation rates; 3: Annotate divergent times, expected mutation rates, and the full posterior distribution of the mutation rate. Default: 2")

parser.add_argument("--threads", "--cores", type=positive_int, metavar="N",
                    help="Maximum threads for MOSEK and numerical libraries. Must be positive. Default: library defaults. Replicates remain sequential.")

args = vars(parser.parse_args())
if args["CI_samples"] is not None:
    if args["CI"] is None:
        parser.error("--CI-samples requires --CI")
    args["CI"]["samples_file"] = args["CI_samples"]

# Set limits before importing NumPy/SciPy or any solver libraries.
if args["threads"] is not None:
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(args["threads"])

from emd.emd_normal_lib import *
from treeswift import *
from simulator.multinomial import *
from emd.util import date_to_years

start = time.time()
print("EMDate was called as follow: " + " ".join(sys.argv))

k = int(args["ncat"]) if args["ncat"] else 50
intreeFile = args["input"]
outtreeFile = args["output"] if args["output"] else (intreeFile + ".mdcatTree")
#infoFile = args["clockout"] if args["clockout"] else (intreeFile + ".mdcatClock")

nreps = int(args["rep"]) if args["rep"] else 100
seqLen = int(args["seqLen"]) if args["seqLen"] else 1000
maxIter = int(args["maxIter"]) if args["maxIter"] else 100

timeFile = args["samplingTime"]
bw_time = args["backward"]
as_date = args["asDate"]
#leaf_time = 0 if bw_time else None

if args["rootTime"] is None:
    tR = None if timeFile else ( 1 if bw_time else 0 )
else:
    tR = float(args["rootTime"]) if not as_date else date_to_years(args["rootTime"])
    
if args["leafTime"] is None:
    if timeFile is not None:
        if bw_time:
            tL = 0
        else:
            tL = None
    else:
        tL = 0 if bw_time else 1            
else:
    tL = float(args["leafTime"]) if not as_date else date_to_years(args["leafTime"])            

try:
    opt = int(args["annotate"])
except:
    opt = 2
place_mu = (opt >= 2)
place_q = (opt >= 3)

try:
    randseed = [int(x) for x in args["randSeed"].strip().split()]
    if len(randseed) == 1 and nreps != 1:
        randseed = randseed[0]
except:
    randseed = None

tree = read_tree_newick(intreeFile)
CI_options = args["CI"]

best_tree,best_llh,best_phi,best_omega = MDCat(tree,k,sampling_time=timeFile,s=seqLen,nrep=nreps,maxIter=maxIter,refTree=None,fixed_tau=False,fixed_omega=False,verbose=args["verbose"],pseudo=1,randseed=randseed,place_mu=place_mu,place_q=place_q,init_Q=None,root_time=tR,leaf_time=tL,bw_time=bw_time,as_date=as_date,CI_options=CI_options,threads=args["threads"])
best_tree.write_tree_newick(outtreeFile)
print("Best log-likelihood: " + str(best_llh))       
end = time.time()
print("Runtime: ", end - start)
