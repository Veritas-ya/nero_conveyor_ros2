"""Compare old and migrated deprojection on one captured frame; offline only."""
import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--old-core',required=True); p.add_argument('--frame',required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; frame=Path(a.frame)
code='''import json, numpy as np, sys\nfrom mujoco_conveyor_sorting.perception import deproject_depth_to_world\np=sys.argv[1]; m=json.load(open(p+".json")); i=m["depth_intrinsics"]; K=np.array([[i["fx"],0,i["cx"]],[0,i["fy"],i["cy"]],[0,0,1.]])\nd=np.load(p+"_depth_m.npy"); x=deproject_depth_to_world(d,K,np.array(m["depth_to_base_cv"]),camera_convention="opencv"); np.save(sys.argv[2],x)'''
with tempfile.TemporaryDirectory() as td:
    outputs=[]
    for core in (Path(a.old_core),root/'src/nero_core'):
        dest=Path(td)/f'{len(outputs)}.npy'; env=dict(os.environ, PYTHONPATH=str(core),PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,'-c',code,str(frame),str(dest)],env=env,cwd=td,check=True)
        outputs.append(dest)
    import numpy as np
    x,y=(np.load(f) for f in outputs)
    result={'frame_metadata_sha256':hashlib.sha256(Path(str(frame)+'.json').read_bytes()).hexdigest(),'depth_sha256':hashlib.sha256(Path(str(frame)+'_depth_m.npy').read_bytes()).hexdigest(),'shape':list(x.shape),'array_equal_including_nan':bool(np.array_equal(x,y,equal_nan=True)),'finite_points':int(np.isfinite(x).all(axis=-1).sum())}
    print(json.dumps(result,ensure_ascii=False,indent=2)); sys.exit(0 if result['array_equal_including_nan'] else 1)
