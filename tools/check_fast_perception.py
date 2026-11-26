#!/usr/bin/env python3
"""Offline GPU acceptance test; no camera, CAN, robot or configuration writes.

Use the official demo only as a runtime smoke test, never as site accuracy
evidence. Real saved target crops can be replayed with --input *.input.npz.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/nero_core'))
from mujoco_conveyor_sorting.runtime.fast_stereo import CACHE,FastStereoProcessBackend


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo',action='store_true',help='官方示例，非现场准确率验证')
    source.add_argument('--input',type=Path,help='历史Fast保存的*.input.npz裁框')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=3)
    args=parser.parse_args(argv)
    if not 1<=args.repeat<=30:parser.error('--repeat须在1到30之间')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(camera_connections=0,can_connections=0,robot_commands_sent=0,
        source='official_demo_not_site_measurement' if args.demo else str(args.input),
        gpu_inference_verified=False,physical_success=None,calls=[])
    backend=None
    try:
        if args.demo:
            left=cv2.imread(str(CACHE/'repo/demo_data/left.png'),cv2.IMREAD_GRAYSCALE)
            right=cv2.imread(str(CACHE/'repo/demo_data/right.png'),cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:raise FileNotFoundError('缺少官方demo图像')
            # Runtime timing only. A bounded 640x384 crop matches deployment;
            # arbitrary metric K/baseline are forbidden for a site comparison.
            lines=(CACHE/'repo/demo_data/K.txt').read_text().strip().splitlines()
            K=np.asarray([float(v) for v in lines[0].split()]).reshape(3,3)
            baseline=float(lines[1])
            h,w=left.shape
            x0,y0=max(0,(w-640)//2),max(0,(h-384)//2)
            crop=(x0,y0,min(w,x0+640),min(h,y0+384))
        else:
            with np.load(args.input,allow_pickle=False) as data:
                left,right=data['left_ir'].copy(),data['right_ir'].copy()
                K=data['K_crop'].copy();baseline=float(data['baseline_m'])
            h,w=left.shape;crop=(0,0,w,h)
        backend=FastStereoProcessBackend(args.output/'backend',request_timeout_s=120.)
        started=time.monotonic();backend.start()
        backend.warmup((crop[3]-crop[1],crop[2]-crop[0]))
        report.update(startup_and_warmup_s=time.monotonic()-started,provenance=backend.provenance)
        backend.request_timeout_s=2.
        for index in range(args.repeat):
            result=backend.refine(left,right,K,baseline,crop,
                capture_metadata={'offline':True,'input_source':report['source'],'repeat':index+1})
            valid=np.isfinite(result.depth_crop_m)&(result.depth_crop_m>0)
            if not valid.any():raise RuntimeError('GPU推理没有返回有效深度')
            report['calls'].append(dict(timings_s=result.timings_s,
                valid_depth_pixels=int(valid.sum()),valid_fraction=float(valid.mean()),
                artifact_prefix=result.artifact_prefix))
        report['gpu_inference_verified']=True
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
    finally:
        if backend is not None:backend.close()
        path=args.output/'report.json'
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(report_file=str(path),gpu_inference_verified=report['gpu_inference_verified'],
            error=report.get('error')),ensure_ascii=False,indent=2))
    return 0 if report['gpu_inference_verified'] else 1


if __name__=='__main__':raise SystemExit(main())
