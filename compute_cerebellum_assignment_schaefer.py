#!/usr/bin/env python3
"""compute_cerebellum_assignment_schaefer.py — test 版小脑 crossFC WTA (Schaefer1000 作为皮层 parcel)

不动 compute_cerebellum_assignment_atlas.py (主盘 K=1000 版)
输入: cortex_schaefer1000.npy (20484 顶点 Schaefer1000 标签) + cortex_cereb.npy
输出: cereb_assignment_schaefer.npy
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

ATLAS = '/data/usersdir/yiluo/atlas'
FC = os.path.join(ATLAS, 'group_fc')

# Schaefer1000 顶点标签 = lh(1-500) + rh(1-500 偏移+500=501-1000) 拼接, 共 1000 parcel
cx_lh = np.load(os.path.join(ATLAS, 'lh_schaefer1000.npy')).astype(np.int32)
cx_rh = np.load(os.path.join(ATLAS, 'rh_schaefer1000.npy')).astype(np.int32)
cx_rh[cx_rh > 0] += 500
cx_lbl = np.concatenate([cx_lh, cx_rh])
cortex_cereb = np.load(os.path.join(FC, 'cortex_cereb.npy')).astype(np.float64)
n_cx, n_cb = cortex_cereb.shape

parcels = np.unique(cx_lbl)
parcels = parcels[parcels > 0]
mean_fc = np.zeros((len(parcels), n_cb), dtype=np.float64)
for i, pl in enumerate(parcels):
    m = cx_lbl == pl
    mean_fc[i] = cortex_cereb[m].mean(axis=0) if m.sum() else 0.0
best = mean_fc.argmax(axis=0)
final = parcels[best].astype(np.int32)

np.save(os.path.join(ATLAS, 'cereb_assignment_schaefer.npy'), final)
print(f'小脑 crossFC WTA (Schaefer1000) 完成: {n_cb} 顶点, {len(parcels)} parcel, 归属 parcel 数 {len(np.unique(final))}')
