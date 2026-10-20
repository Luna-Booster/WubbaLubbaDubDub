#!/usr/bin/env python3
"""gen_cbp_parcel_volume_schaefer.py — test 版 CBP 分区全脑 parcel 体积图 (Schaefer1000 作为皮层)

不动 gen_cbp_parcel_volume.py (K=1000 版)
皮层 = Schaefer1000 (lh+rh 偏移拼接) + 小脑 = cereb_assignment_schaefer + 核团 + 皮下 (同原版)
输出: cbp_parcel_mni_schaefer.nii.gz + cbp_parcel_network_schaefer.npy
"""
import os
import numpy as np
import nibabel as nb
from scipy.ndimage import zoom, map_coordinates
from scipy.spatial import cKDTree

os.environ['OMP_NUM_THREADS'] = '1'

ATLAS = '/data/usersdir/yiluo/atlas'
GROUP_ASEG = '/data/shares/GPUStore2/usersdir/yiluo/CS-DSI_processed/derivatives/group_aseg/aseg_MNI.nii.gz'
SUIT_DIR = '/data/usersdir/yiluo/atlas_new/SUIT'
DEF_SUIT2MNI = os.path.join(SUIT_DIR, 'templates', 'def_suit2mni.nii')
WHITE_SUIT = os.path.join(SUIT_DIR, 'flatmap', 'PIAL_SUIT.surf.gii')
DS_DIR = '/data/usersdir/yiluo/scripts/cerebellum_downsample'
V2C = os.path.join(DS_DIR, 'vertex_to_cluster.npy')
FSAVG_L = os.path.join(ATLAS, 'fsaverage5', 'lh.pial')
FSAVG_R = os.path.join(ATLAS, 'fsaverage5', 'rh.pial')
OUT = os.path.join(ATLAS, 'cbp_parcel_mni_schaefer.nii.gz')
NET_OUT = os.path.join(ATLAS, 'cbp_parcel_network_schaefer.npy')

LUT_PATH = os.path.join(ATLAS, 'schaefer1000_net_parcel.npy')  # parcel 1-1000 -> Yeo17 网络

# Schaefer 网络名 -> Yeo17 编号 (空间重叠确认 100%, LimbicA=TempPole→10, LimbicB=OFC→9)
NET2YEO = {'VisCent': 1, 'VisPeri': 2, 'SomMotA': 3, 'SomMotB': 4,
           'DorsAttnA': 5, 'DorsAttnB': 6, 'SalVentAttnA': 7, 'SalVentAttnB': 8,
           'LimbicB': 9, 'LimbicA': 10, 'ContA': 11, 'ContB': 12, 'ContC': 13,
           'DefaultA': 14, 'DefaultB': 15, 'DefaultC': 16, 'TempPar': 17}


def surface_to_volume(labels, verts, inv_aff, shape):
    vox = np.round(nb.affines.apply_affine(inv_aff, verts)).astype(int)
    valid = ((vox[:, 0] >= 0) & (vox[:, 0] < shape[0]) &
             (vox[:, 1] >= 0) & (vox[:, 1] < shape[1]) &
             (vox[:, 2] >= 0) & (vox[:, 2] < shape[2]))
    vol = np.zeros(shape, dtype=np.int32)
    for p in np.unique(labels):
        if p <= 0:
            continue
        m = labels == p
        if valid[m].any():
            v = vox[m & valid]
            vol[v[:, 0], v[:, 1], v[:, 2]] = p
    return vol


def cereb_surface_to_mni(centers, def_suit2mni_path):
    warp_img = nb.load(def_suit2mni_path)
    warp = warp_img.get_fdata()
    wv = nb.affines.apply_affine(np.linalg.inv(warp_img.affine), centers)
    disp = np.zeros((len(centers), 3), dtype=np.float64)
    for k in range(3):
        disp[:, k] = map_coordinates(warp[..., k], wv.T, order=1, mode='nearest')
    return disp


def main():
    aseg_img = nb.load(GROUP_ASEG)
    aseg = aseg_img.get_fdata().astype(np.int32)
    ref_img = nb.load(os.path.join(ATLAS, '05_cerebellum.nii.gz'))
    aff = ref_img.affine.copy()
    shape = ref_img.shape
    grid = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'), axis=-1)
    src_mm = nb.affines.apply_affine(aff, grid.reshape(-1, 3).astype(np.float64))
    src_idx = nb.affines.apply_affine(np.linalg.inv(aseg_img.affine), src_mm)
    aseg = map_coordinates(aseg, src_idx.T, order=0, mode='constant', cval=0).reshape(shape).astype(np.int32)
    inv_aff = np.linalg.inv(aff)
    all_label = np.zeros(shape, dtype=np.int32)

    # 皮层 1000+ (Schaefer1000: lh 1-500 + rh 1-500 偏移+500=501-1000)
    from brainspace.mesh.mesh_io import read_surface
    cx_lh = np.load(os.path.join(ATLAS, 'lh_schaefer1000.npy')).astype(np.int32)
    cx_rh = np.load(os.path.join(ATLAS, 'rh_schaefer1000.npy')).astype(np.int32)
    cx_rh[cx_rh > 0] += 500
    cortex_labels = np.concatenate([cx_lh, cx_rh])
    verts = np.vstack([read_surface(FSAVG_L).Points, read_surface(FSAVG_R).Points])
    tree = cKDTree(verts)
    cx_vol = np.zeros(shape, dtype=np.int32)
    sv = np.round(nb.affines.apply_affine(inv_aff, verts)).astype(np.int32)
    sv_safe = np.clip(sv, 0, np.array(shape) - 1)
    sok = ((sv[:, 0] >= 0) & (sv[:, 0] < shape[0]) &
           (sv[:, 1] >= 0) & (sv[:, 1] < shape[1]) &
           (sv[:, 2] >= 0) & (sv[:, 2] < shape[2]))
    good = sok & (cortex_labels > 0)
    gv = np.argwhere(good)[:, 0]
    clg = cortex_labels[good]
    counts = np.bincount(clg)
    order = np.argsort([counts[l] for l in clg])[::-1]
    cx_vol[sv[gv[order], 0], sv[gv[order], 1], sv[gv[order], 2]] = 1000 + clg[order]
    cx_mask = (aseg == 3) | (aseg == 42)
    cx_vox = np.argwhere(cx_mask & (cx_vol == 0))
    cx_mm = nb.affines.apply_affine(aff, cx_vox)
    _, idx = tree.query(cx_mm, k=1)
    good2 = cortex_labels[idx] > 0
    cx_vol[cx_vox[good2, 0], cx_vox[good2, 1], cx_vox[good2, 2]] = 1000 + cortex_labels[idx[good2]]
    all_label[cx_vol > 0] = cx_vol[cx_vol > 0]
    print(f'大脑皮层(Schaefer1000): {len(np.unique(cortex_labels[cortex_labels > 0]))} parcel, 体素 {int((cx_vol > 0).sum())}', flush=True)

    # 小脑皮层 2000+ (Schaefer1000 版 crossFC WTA)
    cereb_assignment = np.load(os.path.join(ATLAS, 'cereb_assignment_schaefer.npy')).astype(np.int32)
    white = nb.load(WHITE_SUIT).darrays[0].data
    v2c = np.load(V2C)
    n_cluster = int(v2c.max()) + 1
    white_centers = np.zeros((n_cluster, 3), dtype=np.float64)
    for c in range(n_cluster):
        white_centers[c] = white[v2c == c].mean(axis=0)
    mni_mm = cereb_surface_to_mni(white_centers, DEF_SUIT2MNI)
    bad = ~np.isfinite(mni_mm).all(axis=1)
    if bad.any():
        mni_mm[bad] = np.inf
        print(f'小脑空簇(NaN中心) {int(bad.sum())} 个, 已排除', flush=True)
    cb_img = nb.load(os.path.join(ATLAS, '05_cerebellum.nii.gz'))
    cb_data = cb_img.get_fdata().astype(np.int32)
    cb_full = ((cb_data >= 1) & (cb_data <= 28)).astype(np.float32)
    cb_mm_grid = nb.affines.apply_affine(aff, grid.reshape(-1, 3).astype(np.float64))
    cb_src = nb.affines.apply_affine(np.linalg.inv(cb_img.affine), cb_mm_grid)
    cb_res = map_coordinates(cb_full, cb_src.T, order=0, mode='constant', cval=0).reshape(shape)
    cb_mask = cb_res > 0.5
    cb_vox = np.argwhere(cb_mask)
    cb_mm = nb.affines.apply_affine(aff, cb_vox)
    tree_cb = cKDTree(mni_mm)
    _, cidx = tree_cb.query(cb_mm, k=1)
    all_label[cb_vox[:, 0], cb_vox[:, 1], cb_vox[:, 2]] = 2000 + cereb_assignment[cidx]
    print(f'小脑皮层: mask 体素 {int(cb_mask.sum())}', flush=True)

    cb_edge = ((aseg == 8) | (aseg == 47)) & (all_label == 0)
    cb_edge_vox = np.argwhere(cb_edge)
    if len(cb_edge_vox):
        cb_edge_mm = nb.affines.apply_affine(aff, cb_edge_vox)
        _, eidx = tree_cb.query(cb_edge_mm, k=1)
        all_label[cb_edge_vox[:, 0], cb_edge_vox[:, 1], cb_edge_vox[:, 2]] = 2000 + cereb_assignment[eidx]
    print(f'小脑皮层边缘补: {len(cb_edge_vox)} 体素', flush=True)

    # 小脑核团 4000+ (Diedrichsen, 同原版)
    nuclei_path = os.path.join(ATLAS, '05_cerebellum_nuclei.nii.gz')
    if os.path.exists(nuclei_path):
        nuc = nb.load(nuclei_path).get_fdata().astype(np.int32)
        if nuc.shape != shape:
            nuc = zoom(nuc, [a / b for a, b in zip(shape, nuc.shape)], order=0).astype(np.int32)
        for i, lab in enumerate(sorted([l for l in np.unique(nuc) if l > 0])):
            all_label[nuc == lab] = 4000 + i + 1

    # 皮下 5000+ (TianS4 + aseg + HCPex, 同原版)
    tian_path = os.path.join(ATLAS, '03_subcortex.nii.gz')
    if os.path.exists(tian_path):
        tian_img = nb.load(tian_path)
        tian_1mm = tian_img.get_fdata().astype(np.int32)
        aseg_2mm = nb.load(GROUP_ASEG).get_fdata().astype(np.int32)
        aseg_1mm = zoom(aseg_2mm, [a / b for a, b in zip(shape, aseg_2mm.shape)], order=0).astype(np.int32)
        wm = np.isin(aseg_1mm, [2, 41])
        csf = np.isin(aseg_1mm, [4, 5, 14, 15, 24, 43, 44])
        sub_vol = np.where((tian_1mm > 0) & ~(wm | csf), 5000 + tian_1mm, 0).astype(np.int32)
        max_tian = int(tian_1mm.max())
        extra = [(17, '海马L'), (53, '海马R'), (18, '杏仁核L'), (54, '杏仁核R'),
                 (26, '伏隔核L'), (58, '伏隔核R'), (10, '丘脑L'), (49, '丘脑R'),
                 (11, '尾状核L'), (50, '尾状核R'), (12, '壳核L'), (51, '壳核R'),
                 (13, '苍白球L'), (52, '苍白球R')]
        for i, (lab, _name) in enumerate(extra):
            sub_vol[(aseg_1mm == lab) & (sub_vol == 0)] = 5000 + max_tian + 1 + i
        hcpex_path = os.path.join(ATLAS, '12_hcpex.nii.gz')
        if os.path.exists(hcpex_path):
            hc = nb.load(hcpex_path).get_fdata().astype(np.int32)
            if hc.shape != shape:
                hc = zoom(hc, [a / b for a, b in zip(shape, hc.shape)], order=0).astype(np.int32)
            hcpex_extra = [(388, 'SNcL'), (421, 'SNcR'), (389, 'SNrL'), (422, 'SNrR'),
                           (390, 'VTAL'), (423, 'VTAR'), (391, 'MamL'), (424, 'MamR'),
                           (392, 'SeptL'), (425, 'SeptR'), (393, 'NBL'), (426, 'NBR')]
            for j, (lab, _name) in enumerate(hcpex_extra):
                sub_vol[(hc == lab) & (sub_vol == 0)] = 5000 + max_tian + len(extra) + 1 + j
        all_label[sub_vol > 0] = sub_vol[sub_vol > 0]

    nb.Nifti1Image(all_label.astype(np.int32), aff).to_filename(OUT)
    n_cx = len([l for l in np.unique(all_label) if 1000 <= l < 2000])
    n_cb = len([l for l in np.unique(all_label) if 2000 <= l < 4000])
    n_nuc = len([l for l in np.unique(all_label) if 4000 <= l < 5000])
    n_sub = len([l for l in np.unique(all_label) if 5000 <= l < 6000])
    print(f'cbp_parcel_mni_schaefer: 皮层 {n_cx}, 小脑皮层 {n_cb}, 小脑核团 {n_nuc}, 皮下 {n_sub} -> {OUT}')

    # 网络标注: schaefer1000_net_parcel.npy (parcel 1-1000 -> Yeo17 网络)
    schaefer_net = np.load(LUT_PATH).astype(np.int32) if os.path.exists(LUT_PATH) else np.zeros(1001, dtype=np.int32)
    net_of_parcel = {}
    for lab in np.unique(all_label):
        if lab <= 0:
            continue
        if 1000 <= lab < 2000:
            p = lab - 1000
            net_of_parcel[int(lab)] = int(schaefer_net[p]) if p < len(schaefer_net) else 0
        elif 2000 <= lab < 4000:
            p = lab - 2000
            net_of_parcel[int(lab)] = net_of_parcel.get(1000 + p, 0)
        elif 4000 <= lab < 5000:
            net_of_parcel[int(lab)] = -1
        elif lab >= 5000:
            net_of_parcel[int(lab)] = -1
    np.save(NET_OUT, net_of_parcel)
    print(f'cbp_parcel_network_schaefer: {len(net_of_parcel)} parcels -> 网络标注')


if __name__ == '__main__':
    main()
