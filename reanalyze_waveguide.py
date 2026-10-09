#!/usr/bin/env python3
"""Recompute physical metrics from saved port traces without rerunning the solver."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_waveguide import metrics
from fiber_pulse import slab_mode
from mesh_waveguide import MeshConfig
from warp_waveguide import RunConfig,mode_data


def reanalyze(root):
    root=Path(root)
    digest=hashlib.sha256(Path('analyze_waveguide.py').read_bytes()).hexdigest()
    by_name={}
    for path in sorted(root.glob('*/metrics.json')):
        result=json.loads(path.read_text())
        r=RunConfig(**result['run'])
        c=MeshConfig(**result['mesh'])
        data=np.load(path.parent/'traces.npz')
        if 'higher_modes' not in data:
            continue
        md,ng=mode_data(c,r)
        modes=[slab_mode(2*np.pi/r.wavelength_um,c.width,2.04,1.46,i)
               for i in range(int(np.ceil(2*md['V']/np.pi)))]
        full=np.concatenate((data['trace'],data['higher_modes']),axis=1)
        with np.load(path.parent/'mesh.npz') as mesh:
            center=mesh['center']
        result.update(metrics(data['time_fs'],full,data['ports'],md,ng,center,r,modes,data['mode_norms']))
        result['analysis_sha256']=digest
        path.write_text(json.dumps(result,indent=2)+'\n')
        by_name[path.parent.name]=result
    for study in root.glob('*.json'):
        data=json.loads(study.read_text())
        if isinstance(data,list) and all('name' in r for r in data):
            updated=[dict(name=r['name'],**by_name.get(r['name'],r)) for r in data]
            study.write_text(json.dumps(updated,indent=2)+'\n')
    print('Reanalyzed',len(by_name),'runs')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',default='results/study')
    a=ap.parse_args();reanalyze(a.study)
