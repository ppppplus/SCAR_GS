#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import numpy as np


def find_pairs(root: Path):
    """
    在 root 下找到成对的 *_f.npy 和 *_s.npy，按共同前缀分组并排序。
    返回 [(stem, f_path, s_path), ...]（按 stem 排序）
    """
    f_files = {re.sub(r'_f\.npy$', '', p.name): p for p in root.glob('*_f.npy')}
    s_files = {re.sub(r'_s\.npy$', '', p.name): p for p in root.glob('*_s.npy')}
    # print(f_files)
    stems = sorted(set(f_files.keys()) & set(s_files.keys()))
    pairs = [(stem, f_files[stem], s_files[stem]) for stem in stems]
    missing_f = sorted(set(s_files.keys()) - set(f_files.keys()))
    missing_s = sorted(set(f_files.keys()) - set(s_files.keys()))
    return pairs, missing_f, missing_s


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir or args.input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, missing_f, missing_s = find_pairs(in_dir)
    if not pairs:
        raise RuntimeError("未找到任何成对的 *_f.npy 与 *_s.npy。")

    if missing_f:
        print(f"[WARN] 有 {len(missing_f)} 个前缀缺少 *_f.npy：{missing_f[:5]}{' ...' if len(missing_f)>5 else ''}")
    if missing_s:
        print(f"[WARN] 有 {len(missing_s)} 个前缀缺少 *_s.npy：{missing_s[:5]}{' ...' if len(missing_s)>5 else ''}")

    # 第一遍：统计总 N、检测 C，以及收集 segmap 形状
    total_N = 0
    C_ref = None
    seg_shapes = []
    K_list = []   # 记录每张图的实例数 K_i（= f.shape[0]）
    for stem, f_path, s_path in pairs:
        feats = np.load(f_path)       # [K_i, C]
        seg   = np.load(s_path)       # [H, W]
        if feats.ndim != 2:
            raise ValueError(f"{f_path} 不是二维数组 [K_i, C]。")
        if seg.ndim != 2:
            raise ValueError(f"{s_path} 不是二维数组 [H, W]。")

        K, C = feats.shape
        if C_ref is None:
            C_ref = C
        elif C_ref != C:
            raise ValueError(f"特征维度不一致：{f_path} 的 C={C}，之前的是 C={C_ref}。")

        total_N += K
        K_list.append(K)
        seg_shapes.append(seg.shape)

    print(f"[INFO] 将合并 {len(pairs)} 张图；总特征数 N={total_N}；特征维度 C={C_ref}。")

    # 第二遍：预分配 cache，并顺序回填
    cache_path   = out_dir / "cache.npy"
    offsets_path = out_dir / "offsets.npy"
    segmap_path  = out_dir / "segmaps.npy"        # 固定尺寸堆叠
    # seglist_path = out_dir / "segmap_list.npy"   # 尺寸不一致时的替代

    # cache = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=(total_N, C_ref))
    flist = []
    offsets = np.zeros((len(pairs), 2), dtype=np.int64)

    # 判断 segmap 是否能直接 stack（是否同形状）
    uniform_seg = all(sh == seg_shapes[0] for sh in seg_shapes)
    if uniform_seg:
        H, W = seg_shapes[0]
        # seg_stack = np.memmap(segmap_path, dtype=np.int32, mode='w+', shape=(len(pairs), H, W))
        seg_list = []
        print(f"[INFO] segmap 尺寸统一：[{H}, {W}]，将保存为 {segmap_path.name}")
    else:
        exit(f"[ERROR] segmap 尺寸不统一：{seg_shapes}")

    cursor = 0
    for i, (stem, f_path, s_path) in enumerate(pairs):
        feats = np.load(f_path).astype(np.float32)  # [K_i, C]
        seg   = np.load(s_path)                     # [H, W]，int

        K = feats.shape[0]
        # cache[cursor:cursor+K] = feats
        flist.append(feats)
        offsets[i, 0] = cursor
        offsets[i, 1] = cursor + K
        cursor += K

        # if seg_stack is not None:
        #     seg_stack[i] = seg.astype(np.int32)
        # else:
        seg_list.append(seg.astype(np.int32))

    # flush 到磁盘
    # del cache
    cache = np.concatenate(flist)
    print(cache.shape)
    np.save(cache_path, cache)
    a = np.load(cache_path)
    print(a.shape)
    np.save(offsets_path, offsets)

    segmaps = np.array(seg_list)
    print(segmaps.shape)
    np.save(segmap_path, segmaps, allow_pickle=True)
    b = np.load(segmap_path)
    print(b.shape)
    print(f"[OK] 已生成：\n  - {cache_path}\n  - {offsets_path}\n  - {segmap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合成 cache.npy 与 segmap.npy（或 segmap_list.npy）")
    parser.add_argument("--input_dir", type=str, required=True, help="包含 *_f.npy 与 *_s.npy 的目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认与输入相同）")
    args = parser.parse_args()
    main(args)
