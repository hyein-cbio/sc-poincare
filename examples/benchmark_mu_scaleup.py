"""Supervise declared MU comparison jobs with a wall/RSS budget; no tuning.

Jobs JSON is a list of {id,input,method,epochs,seed,threads} dictionaries.
Optional dataset defaults to the input filename stem and is stored explicitly
for summaries; supplied dataset names must be safe path components. Inputs
are aligned NPZ artifacts. Outputs are new per-job directories; failed, killed
and unstarted jobs remain explicit. Children are isolated Python processes.
No parameter, epoch count or seed is chosen from evaluation quality.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def atomic_json(path, value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def _stop(child):
    if child.poll() is None:
        child.terminate()
        try: child.wait(timeout=15)
        except subprocess.TimeoutExpired: child.kill();child.wait()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--wall-seconds',type=float,default=2700)
    parser.add_argument('--rss-limit-gib',type=float,default=6)
    parser.add_argument('--min-available-gib',type=float,default=2)
    parser.add_argument('--keep-checkpoints',action='store_true')
    args=parser.parse_args(argv)
    if args.output.exists(): parser.error('Output directory must not exist.')
    if args.workers<1 or any(not (0 < value < float('inf')) for value in
        (args.wall_seconds,args.rss_limit_gib,args.min_available_gib)):
        parser.error('Limits must be finite and positive.')
    try:
        jobs=json.loads(args.jobs.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f'Cannot read jobs JSON: {exc}')
    if not isinstance(jobs,list):
        parser.error('Jobs JSON must be a list of job dictionaries.')
    required={'id','input','method','epochs','threads','seed'}
    seen=set()
    for job in jobs:
        if not isinstance(job,dict) or not required.issubset(job):
            parser.error('Each job must be a dictionary with id, input, method, epochs, threads and seed.')
        name=job['id']
        if not isinstance(name,str) or not name or name in ('.','..') or '/' in name or '\\' in name or name in seen:
            parser.error('Job IDs must be distinct safe path components.')
        if not isinstance(job.get('input'),str) or not job['input']:
            parser.error('Each job requires a nonempty input path string.')
        dataset=job.get('dataset',Path(job['input']).stem)
        if not isinstance(dataset,str) or dataset in ('','.','..') or '/' in dataset or '\\' in dataset:
            parser.error('Each job requires a safe dataset name; omit it to use the input filename stem.')
        job['dataset']=dataset
        if job['method'] not in ('rfa','mu_fixed','mu_local','euclidean_fixed','euclidean_local'):
            parser.error('Unsupported method.')
        for field in ('epochs','threads'):
            if type(job[field]) is not int or job[field]<1:parser.error('Epoch/thread counts must be positive integers.')
        if type(job['seed']) is not int or job['seed']<0:parser.error('Seeds must be nonnegative integers.')
        if job.get('resume') is not None and (job['method']=='rfa' or not isinstance(job['resume'],str) or not job['resume']):
            parser.error('resume requires a trusted MU checkpoint path, not an RFA warm start.')
        seen.add(name)
    if not jobs:parser.error('Jobs must not be empty.')
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    summary=dict(status='running',supervisor_pid=os.getpid(),wall_seconds=args.wall_seconds,
        aggregate_rss_limit_bytes=int(args.rss_limit_gib*1024**3),observed_peak_aggregate_rss_bytes=0,
        jobs=[dict(spec=job,status='pending') for job in jobs])
    path=args.output/'summary.json';atomic_json(path,summary)
    active={}; next_job=0; reason=None
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):
        nonlocal reason
        reason='operator_interrupt'
    for s in old_handlers:signal.signal(s,interrupted)
    try:
        while next_job<len(jobs) or active:
            if reason is None and time.monotonic()-started>args.wall_seconds:reason='stage_time_limit'
            if reason is None and (args.output/'STOP').exists():reason='operator_stop_file'
            if reason is None and psutil.virtual_memory().available<args.min_available_gib*1024**3:
                reason='system_memory_pressure'
            total_rss=0
            for index,(child,log) in list(active.items()):
                entry=summary['jobs'][index]
                try:rss=psutil.Process(child.pid).memory_info().rss
                except psutil.NoSuchProcess:rss=0
                total_rss+=rss
                entry['observed_peak_rss_bytes']=max(entry.get('observed_peak_rss_bytes',0),rss)
                if child.poll() is not None:
                    log.close()
                    entry.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-entry['started_monotonic'])
                    metadata=args.output/jobs[index]['id']/'run.json'
                    if metadata.exists():
                        try:entry['result']=json.loads(metadata.read_text())
                        except json.JSONDecodeError:entry['metadata_error']='invalid_json'
                    complete=child.returncode==0 and entry.get('result',{}).get('status')=='completed'
                    entry['status']='completed' if complete else 'failed'
                    if complete:
                        (args.output/f"{jobs[index]['id']}.log").unlink(missing_ok=True)
                        if not args.keep_checkpoints:
                            for filename in ('checkpoint.pt','checkpoint.npz'):
                                (args.output/jobs[index]['id']/filename).unlink(missing_ok=True)
                    del active[index]
            summary['observed_peak_aggregate_rss_bytes']=max(summary['observed_peak_aggregate_rss_bytes'],total_rss)
            if reason is None and total_rss>summary['aggregate_rss_limit_bytes']:reason='aggregate_rss_limit'
            if reason:
                for index,(child,log) in list(active.items()):
                    _stop(child);log.close()
                    summary['jobs'][index].update(status='interrupted',stop_reason=reason,returncode=child.returncode)
                    metadata=args.output/jobs[index]['id']/'run.json'
                    if metadata.exists():
                        try:summary['jobs'][index]['result']=json.loads(metadata.read_text())
                        except json.JSONDecodeError:pass
                    del active[index]
                for entry in summary['jobs'][next_job:]:entry.update(status='not_started',stop_reason=reason)
                break
            while next_job<len(jobs) and len(active)<args.workers:
                job=jobs[next_job];entry=summary['jobs'][next_job]
                command=[sys.executable,str(root/'examples/run_mu_study.py'),
                         '--input',str(Path(job['input']).resolve()),'--output',str((args.output/job['id']).resolve()),
                         '--method',job['method'],'--epochs',str(job['epochs']),'--seed',str(job['seed']),
                         '--threads',str(job['threads'])]
                if job.get('resume') is not None:
                    command.extend(['--resume',str(Path(job['resume']).resolve())])
                env=os.environ.copy();env['PYTHONPATH']=str(root/'src')
                for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
                    env[key]=str(job['threads'])
                log=(args.output/f"{job['id']}.log").open('w')
                try:
                    child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    entry['status']='failed_to_launch'
                    raise
                entry.update(status='running',pid=child.pid,started_monotonic=time.monotonic())
                active[next_job]=(child,log);next_job+=1
            summary['elapsed_seconds']=time.monotonic()-started
            atomic_json(path,summary)
            if active:time.sleep(.5)
        summary['status']='completed' if all(e['status']=='completed' for e in summary['jobs']) else 'partial'
        summary['stop_reason']=reason
    except BaseException as exc:
        summary.update(status='failed', error=dict(type=type(exc).__name__,message=str(exc)))
        for index in active:
            summary['jobs'][index].update(status='interrupted',stop_reason='supervisor_exception')
        for entry in summary['jobs']:
            if entry['status']=='pending':entry.update(status='not_started',stop_reason='supervisor_exception')
        raise
    finally:
        for child,log in active.values():_stop(child);log.close()
        for s,handler in old_handlers.items():signal.signal(s,handler)
        summary['elapsed_seconds']=time.monotonic()-started
        atomic_json(path,summary)
    return summary


if __name__=='__main__':main()
