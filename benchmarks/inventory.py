"""Freeze archive compatibility and complete input fingerprints before timing."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path

from .common import _dataset_sources,load_benchmark_dataset


def inspect_archive(data,datasets=None):
    args=argparse.Namespace(data=str(data),datasets=datasets)
    sources=_dataset_sources(args);skipped=[];compatible=[]
    for name,train,test in sources:
        d=load_benchmark_dataset(args,name,train,test,skipped)
        if d is None:continue
        if d.X_train.shape[1]!=1:
            skipped.append(dict(name=name,sources=d.sources,reason='Requires univariate data'))
            continue
        compatible.append(dict(name=name,fingerprint=d.fingerprint,sources=d.sources,
                               n_train=len(d.X_train),n_test=len(d.X_test),length=d.X_train.shape[-1]))
        del d
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),data=str(Path(data).resolve()),
                source_inventory=[(n,str(a),str(b) if b else None) for n,a,b in sources],
                discovered_datasets=len(sources),datasets=compatible,skipped_datasets=skipped)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=inspect_archive(args.data)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(discovered=result['discovered_datasets'],compatible=len(result['datasets']),
                         skipped=len(result['skipped_datasets']))))


if __name__=='__main__':main()
