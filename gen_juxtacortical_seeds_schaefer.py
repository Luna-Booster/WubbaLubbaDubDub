#!/usr/bin/env python3
"""gen_juxtacortical_seeds_oishi.py — Oishi 2008 方法生成大脑 SWM seed (LPBA40 gyrus 最大似然归属)

替代 gen_juxtacortical_seeds.py 的"表面栅格化 cx_vol + FA>=0.15 + 膨胀2次"做法
核心: 浅表白质体素按 LPBA40 gyrus 最大似然归属, 再映射到 918 皮层 parcel (最近 parcel 灰质顶点细分)

输入:
  LPBA40 gm.label (gyrus 21-122)  / group_aseg (aseg_MNI)  / group_parcel_cc/cortex_labels.npy (918)
  fsaverage5 白质表面 (parcel 顶点坐标)
输出:
  all_seeds_label_oishi.nii.gz (大脑 1000-1999, parcel 归属; 小脑/其他继承旧版 all_seeds_label.nii.gz)
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import nibabel as nb
from scipy.ndimage import zoom, distance_transform_edt
from nilearn.image import resample_to_img

LPBA40_GM = '/data/usersdir/yiluo/atlas_new/LPBA40/lpba40_dseg.nii.gz'  # CAT12 版, 带正确 sform (1.5mm MNI)
GROUP_ASEG = '/data/shares/GPUStore2/usersdir/yiluo/CS-DSI_processed/derivatives/group_aseg/aseg_MNI.nii.gz'
GROUP_PARCEL = '/data/usersdir/yiluo/atlas'  # 主盘 (cortex_labels.npy 在根目录)
CBP = '/data/usersdir/yiluo/atlas/cbp_parcel_mni.nii.gz'  # 参考网格 193x229x193 @1mm
FSAVG_L = '/data/usersdir/yiluo/atlas/fsaverage5/hemi-L_white.surf.gii'
FSAVG_R = '/data/usersdir/yiluo/atlas/fsaverage5/hemi-R_white.surf.gii'
OLD_SEED = '/data/usersdir/yiluo/atlas/juxtacortical_seeds/all_seeds_label.nii.gz'
OUT = '/data/usersdir/yiluo/atlas/juxtacortical_seeds/all_seeds_label_schaefer.nii.gz'

# 0. CBP 参考网格 (193x229x193 @1mm)
ref_img = nb.load(CBP)
ref_aff = ref_img.affine
shape = ref_img.shape
inv_aff = np.linalg.inv(ref_aff)

# 1. aseg 重采样到 CBP 网格
aseg = resample_to_img(nb.load(GROUP_ASEG), ref_img, interpolation='nearest', force_resample=True).get_fdata().astype(np.int32)
wm = np.isin(aseg, [2, 41])  # 大脑白质 (解剖边界, 不用 FA 阈值)

# 2. LPBA40 重采样到 CBP 网格
lpba = resample_to_img(nb.load(LPBA40_GM), ref_img, interpolation='nearest', force_resample=True).get_fdata().astype(np.int32).squeeze()
gyrus_mask = (lpba >= 21) & (lpba <= 122)  # 24 gyrus 左右 (排除深部核团 161-166)

# 3. 浅表白质 = WM 中离 gyrus 灰质 <= 1mm 的体素
dist_to_gyrus = distance_transform_edt(~gyrus_mask)
juxta = wm & (dist_to_gyrus <= 1.0)

# 4. Schaefer1000 parcel 灰质顶点 (fsaverage5) -> 体积坐标 (lh 1-500 + rh 偏移 501-1000)
cx_lh = np.load(os.path.join(GROUP_PARCEL, 'lh_schaefer1000.npy')).astype(np.int32)
cx_rh = np.load(os.path.join(GROUP_PARCEL, 'rh_schaefer1000.npy')).astype(np.int32)
cx_rh[cx_rh > 0] += 500
cortex_labels = np.concatenate([cx_lh, cx_rh])
verts = np.vstack([nb.load(FSAVG_L).darrays[0].data, nb.load(FSAVG_R).darrays[0].data])
voxx = np.round(nb.affines.apply_affine(inv_aff, verts)).astype(np.int64)
valid = ((voxx[:, 0] >= 0) & (voxx[:, 0] < shape[0])
         & (voxx[:, 1] >= 0) & (voxx[:, 1] < shape[1])
         & (voxx[:, 2] >= 0) & (voxx[:, 2] < shape[2]))

# 5. 浅表白质体素 -> 最近 parcel 灰质顶点 (1 次距离变换, 非逐 parcel)
parcel_vol = np.zeros(shape, dtype=np.int32)
ok = valid & (cortex_labels > 0)  # 排除内侧壁 0
parcel_vol[voxx[ok][:, 0], voxx[ok][:, 1], voxx[ok][:, 2]] = cortex_labels[ok] + 1000
_, idx = distance_transform_edt(parcel_vol == 0, return_indices=True)
nearest = parcel_vol[idx[0], idx[1], idx[2]]
all_label = np.where(juxta, nearest, 0).astype(np.int32)

# 6. 小脑/其他部分继承旧版 all_seeds_label (只替换大脑 1000-1999)
if os.path.exists(OLD_SEED):
    old = nb.load(OLD_SEED).get_fdata().astype(np.int32)
    if old.shape != shape:
        old = zoom(old, [shape[i] / old.shape[i] for i in range(3)], order=0).astype(np.int32)
    old[(old >= 1000) & (old < 2000)] = 0
    all_label = np.where(all_label > 0, all_label, old)

nb.Nifti1Image(all_label.astype(np.int32), ref_aff).to_filename(OUT)
n_swm = int(((all_label >= 1000) & (all_label < 2000)).sum())
print(f'Oishi SWM seed 完成 -> {OUT}, 大脑 seed {n_swm} 体素, {len(np.unique(all_label[(all_label >= 1000) & (all_label < 2000)]))} parcel')

# 7. per-subject ACPC seed (warp 组级 oishi seed 到各 sub 的 ACPC, 追踪 10_sc_cdomain_dsi.py ① 用)
import subprocess
SIF_FMR = '/home/yiluo/fmri/soft_img/fmriprep.sif'
QSIP = '/data/shares/GPUStore2/usersdir/yiluo/CS-DSI_processed/derivatives/qsiprep'


def warp_image(src, ref, out, xfm):
    subprocess.run(f"singularity exec -B /data:/data {SIF_FMR} antsApplyTransforms -d 3 "
                   f"-i {src} -r {ref} -o {out} -n NearestNeighbor -t {xfm}", shell=True)


subs = sorted(d for d in os.listdir(QSIP) if d.startswith('sub-') and os.path.isdir(os.path.join(QSIP, d)))
for sid in subs:
    mni_to_acpc = f'{QSIP}/{sid}/anat/{sid}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5'
    acpc_ref = f'{QSIP}/{sid}/ses-1/dwi/{sid}_ses-1_space-ACPC_desc-preproc_dwi.nii.gz'
    if not (os.path.exists(mni_to_acpc) and os.path.exists(acpc_ref)):
        print(f'{sid}: 缺变换/ref, 跳过')
        continue
    out_dir = f'/data/usersdir/yiluo/atlas/juxtacortical_seeds_schaefer/{sid}'
    os.makedirs(out_dir, exist_ok=True)
    acpc_seed = f'{out_dir}/all_seeds_label.nii.gz'
    warp_image(OUT, acpc_ref, acpc_seed, mni_to_acpc)
    print(f'{sid}: ACPC seed -> {acpc_seed}')
print('=== per-subject ACPC seed 完成 ===')
