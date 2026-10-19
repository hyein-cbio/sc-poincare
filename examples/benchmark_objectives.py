"""Compare fixed-budget objectives on a user-provided CSV, without label fitting.

Optional legacy-floor reference reproduces this package's former numerical
policy, not an external implementation. No download, reference fit, display
forces, dataset-specific recipe or runtime monkeypatch is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from sc_poincare._disk import pairwise_disk_length
from sc_poincare._fitting import (
    _seed_from_connectivities, _torch_disk_length,
    _riemannian_retract, fit_disk_embedding,
)
from sc_poincare._objectives import DEFAULT_OBJECTIVE, CONDITIONAL_OBJECTIVE
from sc_poincare._proximity import estimate_proximity, local_affinity
from sc_poincare._resources import DEFAULT_MAX_DENSE_BYTES
from sc_poincare._research_policy import ensure_active_paths


def _package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _run_provenance():
    """Import path, revision, and versions immediately before fitting."""
    import sc_poincare
    revision = dirty = diff_sha = None
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        porcelain = subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL)
        diff = subprocess.check_output(['git', 'diff', 'HEAD'], stderr=subprocess.DEVNULL)
        dirty = bool(porcelain.strip())
        diff_sha = hashlib.sha256(diff).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return {
        'import_file': sc_poincare.__file__,
        'git_revision': revision,
        'git_dirty': dirty,
        'git_diff_sha256': diff_sha,
        'versions': {name: _package_version(name) for name in
                     ('numpy', 'scipy', 'scikit-learn', 'pandas', 'torch', 'sc-poincare')},
    }


def _neighbor_purity(coordinates, labels, k):
    """Post-fit chart neighbor agreement. Labels do not enter the fit."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    indices = nn.kneighbors(coordinates, return_distance=False)[:, 1:]
    purity = np.array([np.mean(labels[row] == labels[i]) for i, row in enumerate(indices)])
    scores = {str(label): float(purity[labels == label].mean())
              for label in sorted(set(labels), key=str)}
    scores['overall'] = float(purity.mean())
    return scores


# BEGIN LEGACY FLOOR CONTROL — historical ablation only, never a default
# The explicit --include-legacy-floor flag is the sole CLI entry to this block.
def _legacy_reference(target, initial, args):
    """Isolated historical 1e-12 policy; all columns/global gradients retained."""
    import torch

    x=torch.tensor(initial,dtype=torch.float64,requires_grad=True)
    p=torch.as_tensor(target,dtype=torch.float64)
    best=np.inf;best_xy=initial.copy();trace=[]
    for epoch in range(args.epochs+1):
        x.grad=None;total=0.
        with torch.set_grad_enabled(epoch<args.epochs):
            for start in range(0,len(x),args.batch_size):
                stop=min(start+args.batch_size,len(x))
                d=_torch_disk_length(x[start:stop],x,self_columns=torch.arange(start,stop))
                pp=p[start:stop].clamp_min(1e-12);pp=pp/pp.sum(1,keepdim=True)
                q=torch.softmax(-d/args.gamma,dim=1).clamp_min(1e-12);q=q/q.sum(1,keepdim=True)
                loss=(pp*(pp.log()-q.log())+q*(q.log()-pp.log())).sum()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite legacy reference loss')
                total+=float(loss.detach())
                if epoch<args.epochs:loss.backward()
        if epoch:trace.append(total)
        if total<best:best=total;best_xy=x.detach().numpy().copy()
        if epoch==args.epochs:break
        with torch.no_grad():
            step=args.learning_rate/10 if epoch<args.burn_in else args.learning_rate
            x.copy_(_riemannian_retract(x,x.grad,step,2.5))
    return best_xy,trace,best,x.detach().numpy().copy()
# END LEGACY FLOOR CONTROL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--label-column',help='Default: last CSV column; removed before fitting')
    parser.add_argument('--pca',type=int,help='Optional PCA of numeric features, no labels')
    parser.add_argument('--n-neighbors',type=int,default=15)
    parser.add_argument('--sigma',type=lambda s:'median' if s=='median' else float(s),default='median')
    parser.add_argument('--gamma',type=float,default=.5)
    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--learning-rate',type=float,default=.01)
    parser.add_argument('--burn-in',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--evaluation-k',type=int,nargs='+',default=[15,20])
    parser.add_argument('--anchor-seed',type=int,default=714)
    parser.add_argument('--anchors',type=int,default=128)
    # LEGACY ONLY: explicit historical control, not a model recommendation.
    parser.add_argument('--include-legacy-floor',action='store_true',
                        help='LEGACY historical probability-floor ablation only; never changes the corrected default')
    parser.add_argument('--max-dense-mib',type=float,default=DEFAULT_MAX_DENSE_BYTES/1024**2)
    parser.add_argument('--output',type=Path,required=True,help='JSON plus NPZ outputs; preferably under ignored data/')
    args=parser.parse_args()
    modes=[DEFAULT_OBJECTIVE,CONDITIONAL_OBJECTIVE]
    if args.include_legacy_floor:modes.insert(0,'legacy_floor_1e-12')
    outputs=[args.output]+[args.output.with_name(f'{args.output.stem}_{mode}_seed{seed}.npz')
                           for mode in modes for seed in args.seeds]
    ensure_active_paths(args.csv, *outputs)
    for name in ['threads','epochs','batch_size','anchors']:
        if getattr(args,name)<1:parser.error(f'--{name.replace("_","-")} must be positive')
    if args.burn_in<0 or not np.isfinite(args.max_dense_mib) or args.max_dense_mib<=0:
        parser.error('burn-in must be nonnegative and dense-array budget finite/positive')
    if len(set(args.seeds))!=len(args.seeds):parser.error('Seed values must be distinct')
    try:
        import torch
    except ImportError:
        torch=None
    if args.include_legacy_floor and torch is None:parser.error('Legacy reference requires the torch extra')
    if torch is not None:torch.set_num_threads(args.threads)
    budget=int(args.max_dense_mib*1024**2)
    input_bytes=args.csv.read_bytes()
    input_hash=hashlib.sha256(input_bytes).hexdigest()
    frame=pd.read_csv(BytesIO(input_bytes));label_column=args.label_column or frame.columns[-1]
    labels=frame[label_column].to_numpy(dtype=str)
    label_counts={str(label):int(count) for label,count in zip(*np.unique(labels,return_counts=True))}
    features=frame.drop(columns=label_column).to_numpy(dtype=np.float64)
    if len(features)<3:parser.error('At least three cells are required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if any(path.exists() for path in outputs):
        parser.error('Output evidence already exists; use a fresh --output prefix')
    records=[]
    with threadpool_limits(limits=args.threads):
        if args.pca is not None:
            if not 1<=args.pca<=min(features.shape):parser.error('--pca exceeds available dimensions')
            features=PCA(args.pca,svd_solver='full').fit_transform(features)
        started=time.perf_counter();proximity=estimate_proximity(features,args.n_neighbors,args.sigma,max_dense_bytes=budget)
        preparation_seconds=time.perf_counter()-started
        anchors=np.sort(np.random.default_rng(args.anchor_seed).choice(len(features),min(args.anchors,len(features)),replace=False))
        mask=np.ones((len(anchors),len(features)),bool);mask[np.arange(len(anchors)),anchors]=False
        evaluations={}
        for k in args.evaluation_k:
            _,lengths=local_affinity(features,k,2*np.linalg.norm(features,axis=1).max()+1)
            distances=dijkstra(lengths,directed=False,indices=anchors)
            ranked=distances.copy();ranked[~mask]=np.inf
            nn=np.argsort(ranked,axis=1)[:,:min(15,len(features)-1)]
            evaluations[k]=(distances,nn)
        # Same anchors as the graph scores. This is not a graph geodesic.
        feature_d=np.linalg.norm(features[anchors,None,:]-features[None,:,:],axis=-1)
        provenance=_run_provenance()
        for seed in args.seeds:
            init_status={};initial=_seed_from_connectivities(proximity.weights,2,np.random.default_rng(seed),status=init_status)
            for mode in modes:
                started=time.perf_counter()
                if mode=='legacy_floor_1e-12':
                    xy,trace,loss,current=_legacy_reference(proximity.accessibility,initial,args)
                    stop_reason='epoch_limit'
                    grad_trace=None
                else:
                    report=fit_disk_embedding(proximity.accessibility,initial_coordinates=initial,
                        temperature=args.gamma,n_epochs=args.epochs,patience=args.epochs,
                        min_delta=0.,learning_rate=args.learning_rate,burn_in_epochs=args.burn_in,
                        batch_size=args.batch_size,random_state=seed,max_dense_bytes=budget,objective=mode)
                    xy,trace,loss=report.coordinates,report.loss_trace,report.best_loss
                    current=report.current_coordinates
                    stop_reason=report.stop_reason
                    grad_trace=[float(v) for v in report.gradient_norm_trace]
                seconds=time.perf_counter()-started
                h=pairwise_disk_length(xy[anchors],xy);ranked=h.copy();ranked[~mask]=np.inf
                hn=np.argsort(ranked,axis=1)[:,:min(15,len(features)-1)]
                scores={}
                for k,(g,gn) in evaluations.items():
                    scores[str(k)]={'global_distance_rank':float(spearmanr(g[mask],h[mask]).statistic),
                        'local_overlap_at15':float(np.mean([len(set(a)&set(b))/len(a) for a,b in zip(gn,hn)]))}
                scores['feature']={'global_distance_rank':float(spearmanr(feature_d[mask],h[mask]).statistic)}
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform!='darwin':rss*=1024
                purity_k=min(args.evaluation_k[0],len(features)-1)
                row={'seed':seed,'objective':mode,'epochs_run':len(trace),'stop_reason':stop_reason,
                    'fitting_seconds':seconds,'best_loss_of_own_objective':float(loss),
                    'gradient_norm_trace':grad_trace,'evaluation':scores,
                    'initialization_source':init_status,'radius_quantiles':np.quantile(np.linalg.norm(xy,axis=1),[0,.1,.5,.9,1]).tolist(),
                    'label_neighbor_purity_k':purity_k,
                    'label_neighbor_purity':_neighbor_purity(xy,labels,purity_k),
                    'process_high_water_rss_bytes':rss}
                records.append(row)
                np.savez(args.output.with_name(f'{args.output.stem}_{mode}_seed{seed}.npz'),
                         coordinates=xy,current_coordinates=current,initial_coordinates=initial,
                         labels=labels,loss_trace=np.asarray(trace,dtype=float))
                print(json.dumps(row),flush=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    result={'input_sha256':input_hash,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'n_cells':len(features),'n_features':features.shape[1],'configuration':config,
        'sigma_effective':proximity.kernel_width,'proximity_diagnostics':proximity.diagnostics,
        'label_counts':label_counts,'provenance':provenance,
        'preparation_seconds':preparation_seconds,'records':records,
        'notes':['All objectives share the same initial coordinates per seed.',
                 'Epoch limits do not imply convergence. Losses differ across objectives.',
                 'coordinates is the lowest training loss. current_coordinates is the final evaluated state.',
                 'Evaluation graph/anchors are diagnostic, not blinded biological holdouts.',
                 'feature.global_distance_rank compares embedding distances with direct feature distances on the same anchors.',
                 'Label counts and neighbor purity are recorded after fitting and do not select the checkpoint.',
                 'No labels or reference coordinates enter fitting.',
                 'provenance is the import path, Git revision, diff hash, and versions before the first fit.',
                 'Dense-array budget is not an RSS cap. High-water RSS includes earlier modes/imports/evaluation.']}
    args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
