"""Summarize finished/budget-ended MU stages; never refit or select conditions.

A study root contains small/ and full/ supervisor outputs, inputs/{stage}/*.npz
and study_input_manifest.json. Root-centered views are secondary and use only the
predeclared input root. Raw disk views and native Euclidean axes remain separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sc_poincare._disk import pairwise_disk_length, place_point_at_origin
from sc_poincare.evaluation import NeighborhoodEvaluation, _array_hash, _canonical_labels
from sc_poincare.visibility import evaluate_visibility

P_METHODS=('rfa','mu_fixed','mu_local')
E_METHODS=('euclidean_fixed','euclidean_local')


def _safe(value):
    return isinstance(value,str) and value not in ('','.','..') and '/' not in value and '\\' not in value


def _snapshot(metadata,points,labels,ids):
    labels=_canonical_labels(labels,len(points))
    summary=deepcopy(metadata['evaluation'])
    for name,value in [('coordinates',points),('labels',labels),('cell_ids',ids)]:
        if _array_hash(value)!=summary['provenance'][f'{name}_sha256']:
            raise ValueError(f'Saved {name} hash mismatch.')
    return NeighborhoodEvaluation(points.copy(),labels,summary,{}, {},ids.copy())


def _root_view(snapshot,index):
    shifted=place_point_at_origin(snapshot.coordinates,index)
    anchors=np.sort(np.random.default_rng(714).choice(len(shifted),min(16,len(shifted)),replace=False))
    before=pairwise_disk_length(snapshot.coordinates[anchors],snapshot.coordinates)
    after=pairwise_disk_length(shifted[anchors],shifted)
    if not np.allclose(before,after,rtol=1e-5,atol=1e-6):
        raise FloatingPointError('Root isometry lost distance precision; raw view retained, no clipping.')
    summary=deepcopy(snapshot.summary);summary['provenance']['coordinates_sha256']=_array_hash(shifted)
    result=NeighborhoodEvaluation(shifted,snapshot.labels.copy(),summary,{}, {},snapshot.cell_ids.copy())
    return result,dict(max_absolute_distance_error=float(np.max(np.abs(before-after))),
                       fixed_validation_anchors=anchors.tolist())


def _plot(name,stage,methods,records,input_path,root_index=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with np.load(input_path,allow_pickle=False) as data:
        features=data['features'];labels=_canonical_labels(data['labels'],len(features))
        ids=np.asarray(data['cell_ids'],dtype=object).astype(str)
    snapshots={};details={}
    for method in methods:
        entry=records.get((stage,name,method,0))
        if entry is None or entry['status']!='completed':
            details[method]=dict(status='unavailable',reason=entry.get('stop_reason','failed') if entry else 'not_planned')
            continue
        metadata=entry['result']
        with np.load(entry['result_path'],allow_pickle=False) as data:points=data['coordinates']
        if _array_hash(np.asarray(features,dtype=np.float64))!=metadata['evaluation']['provenance']['features_sha256']:
            raise ValueError('Plot input feature hash mismatch.')
        snapshot=_snapshot(metadata,points,labels,ids)
        details[method]=dict(status='raw')
        if root_index is not None:
            try:
                snapshot,validation=_root_view(snapshot,root_index)
                details[method]=dict(status='root_centered',validation=validation)
            except (ValueError,FloatingPointError) as exc:
                details[method]=dict(status='root_view_unavailable',reason=str(exc))
                continue
        snapshots[method]=snapshot
        if metadata['geometry']=='poincare':details[method]['visibility']=evaluate_visibility(snapshot)
    dpi=100;diameter=600;width=len(methods)*750;height=850
    fig=plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi)
    classes=np.unique(labels);palette=plt.get_cmap('tab20')
    colors={label:palette(i%20) for i,label in enumerate(classes)}
    native=methods[0].startswith('euclidean')
    limit=max([float(np.abs(s.coordinates).max()) for s in snapshots.values()]+[1e-6])*1.05 if native else 1.
    view='native Euclidean axes' if native else ('fixed input-root isometry' if root_index is not None else 'raw disk')
    fig.suptitle(f'{stage} {name} | n={len(labels)}, seed 0 | {view}\n'
                 'Shared inputs; bundled objective/optimizer comparison, not biological validation',fontsize=13,y=.98)
    for i,method in enumerate(methods):
        ax=fig.add_axes([(50+i*750)/width,140/height,diameter/width,diameter/height])
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal')
        if not native:ax.add_patch(plt.Circle((0,0),1,fill=False,color='.6',linewidth=.7))
        if method not in snapshots:
            ax.set_axis_off()
            ax.text(.05,.5,f'{method}\n{details[method]["status"]}\n{details[method].get("reason","")}',
                    transform=ax.transAxes,fontsize=9,wrap=True)
            continue
        snapshot=snapshots[method];xy=snapshot.coordinates
        ax.scatter(xy[:,0],xy[:,1],c=[colors[x] for x in labels],s=(4*72/dpi)**2,linewidths=0,alpha=.8)
        scores=snapshot.summary['metrics']['15'];entry=records[(stage,name,method,0)]
        mse=entry['result']['expression']['normalized_mse']['15'] if entry['result'].get('expression') else None
        display_names={'rfa':'RFA (shared-init)','mu_fixed':'MU fixed','mu_local':'MU local-radius',
                       'euclidean_fixed':'Euclidean MU fixed','euclidean_local':'Euclidean MU local-radius'}
        caption=f"{display_names[method]}\nraw K15 purity={scores['overall_purity']:.3f}, macro={scores['macro_purity']:.3f}"
        if mse is not None:caption+=f', MSE={mse:.3f}'
        ax.set_title(caption,fontsize=9)
        handles=[Line2D([],[],marker='o',linestyle='',color=colors[x],label=x,markersize=3) for x in classes]
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,-.12),ncol=4,fontsize=7,frameon=False)
    return fig,details


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('Output directory must not exist.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    input_manifest=json.loads((args.study/'study_input_manifest.json').read_text())
    stages={};records={};rows=[]
    for stage in ('small','full'):
        path=args.study/stage/'summary.json'
        if not path.exists():continue
        summary=json.loads(path.read_text())
        if summary['status']=='running':raise ValueError('Wait for the supervisor to finish or stop it explicitly.')
        stages[stage]=dict(status=summary['status'],elapsed_seconds=summary['elapsed_seconds'],
            observed_peak_aggregate_rss_bytes=summary['observed_peak_aggregate_rss_bytes'],
            supervisor_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for entry in summary['jobs']:
            spec=entry['spec'];name=spec['dataset'];method=spec['method'];seed=spec['seed']
            if not _safe(name) or not _safe(spec['id']) or method not in P_METHODS+E_METHODS:
                raise ValueError('Unsafe dataset/job ID or unknown method.')
            if type(seed) is not int or seed<0:raise ValueError('Invalid seed.')
            key=(stage,name,method,seed)
            if key in records:raise ValueError('Duplicate study job.')
            record=deepcopy(entry)
            record['result_path']=str(args.study/stage/spec['id']/'result.npz')
            records[key]=record
            row=dict(stage=stage,dataset=name,method=method,seed=seed,status=entry['status'])
            if entry['status']=='completed':
                r=entry['result'];m=r['evaluation']['metrics'];g=r['evaluation']['global_feature_rank']
                row.update(n_cells=r['n_cells'],epochs=r['n_epochs'],best_epoch=r['best_epoch'],
                    fitting_seconds=r['fitting_seconds'],preparation_seconds=r['preparation_seconds'],
                    process_peak_rss_bytes=r['process_peak_rss_bytes'],
                    k15_purity=m['15']['overall_purity'],k15_macro=m['15']['macro_purity'],
                    k15_recall=m['15']['input_neighbor_recall'],k30_purity=m['30']['overall_purity'],
                    k30_macro=m['30']['macro_purity'],k30_recall=m['30']['input_neighbor_recall'],
                    global_feature_rank=g['mean_spearman'],undefined_global_anchors=g['n_undefined'],
                    initial_disk_sha256=r['initial_disk_sha256'],initial_tangent_sha256=r['initial_tangent_sha256'])
                if r.get('resume_from_epoch') is not None:
                    row['resume_from_epoch']=r['resume_from_epoch']
                    row['fitting_seconds_semantics']='resumed segment only'
                    prior=entry.get('prior_attempt',{}).get('result') or {}
                    row['prior_attempt_process_wall_seconds']=prior.get('elapsed_seconds')
                if r.get('expression'):row.update(k15_expression_mse=r['expression']['normalized_mse']['15'],
                                                  k30_expression_mse=r['expression']['normalized_mse']['30'])
                if r.get('visibility'):row['raw_cross_label_overlap']=r['visibility']['cross_label_overlap_fraction']
            else:row['error']=(entry.get('result') or {}).get('error',entry.get('stop_reason','not completed'))
            rows.append(row)
    # All same-stage/dataset/seed methods must share the exact disk/tangent initializer.
    for stage,name,_,seed in records:
        matched=[r for r in rows if r['stage']==stage and r['dataset']==name and r['seed']==seed and r['status']=='completed']
        for key in ('initial_disk_sha256','initial_tangent_sha256'):
            if len({r[key] for r in matched})>1:raise ValueError(f'Shared initialization mismatch: {stage}/{name}/{seed}.')
    args.output.mkdir(parents=True,exist_ok=False)
    fields=sorted({k for row in rows for k in row})
    with (args.output/'scores.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result=dict(stages=stages,scores=rows,views={},
                analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                plot_policy=dict(disk_diameter_px=600,marker_diameter_px=4,dpi=100,
                    representative_seed=0,native_euclidean_axes='shared native limits, no disk projection',
                    root_isometry='predeclared input root; 16 fixed validation rows; no clipping'))
    for stage in stages:
        names=sorted({r['dataset'] for r in rows if r['stage']==stage})
        for name in names:
            input_path=args.study/f'inputs/{stage}/{name}.npz'
            for methods,suffix in [(P_METHODS,'poincare_raw'),(E_METHODS,'euclidean_native')]:
                figure,details=_plot(name,stage,methods,records,input_path)
                try:figure.savefig(args.output/f'{stage}_{name}_{suffix}.png',dpi=figure.dpi)
                finally:plt.close(figure)
                result['views'][f'{stage}_{name}_{suffix}']=details
            if stage=='full' and name in input_manifest:
                index=input_manifest[name]['root_index']
                figure,details=_plot(name,stage,P_METHODS,records,input_path,root_index=index)
                try:figure.savefig(args.output/f'{stage}_{name}_poincare_root.png',dpi=figure.dpi)
                finally:plt.close(figure)
                result['views'][f'{stage}_{name}_poincare_root']=details
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':main()
