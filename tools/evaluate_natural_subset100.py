"""Frozen, prediction-independent DSEC natural-subset checkpoint comparison."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from src.config.load import load_config
from src.datasets.build import build_datasets
from src.datasets.transforms import NormalizedMapDataset
from src.models.build import build_model
from tools.complexity import _disable_pretrained
from tools.dsec_natural_degradation import (
    load_quality_rows, matched_controls, metrics_from_confusion,
    bootstrap_interval, _block_bootstrap_mean, sequence_frame_sample, write_rows,
)

AUDIT = ROOT / 'outputs/natural_degradation/dsec'
OUT = AUDIT / 'subset100_20260910'
RUNS = {
    'image_only': ROOT / 'outputs/logs/26-09-09--15-43-58_redesign_dsec_r1_seed1_dsec_rcafnet',
    'rcafnet': ROOT / 'outputs/logs/26-08-23--21-44-00_ce_lovasz_dsec_dsec_rcafnet',
}
NAMES = {'underexposure': '相对低照度', 'overexposure': '局部高光饱和',
         'motion_blur': '低清晰度/模糊候选'}


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('model', 'state_dict', 'model_state_dict', 'state_dict_ema'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze(rows):
    source = json.loads((AUDIT / 'frozen_subsets.json').read_text(encoding='utf-8'))
    hist = np.load(AUDIT / 'label_histograms.npy')
    manifest = {'count_per_condition': 100, 'control_count_per_condition': 100,
                'seed': 1, 'selection_uses_predictions': False,
                'source_sha256': sha(AUDIT / 'frozen_subsets.json'),
                'selection': 'Proportional sequence-stratified random subsample of original degraded pools; '
                             'largest-remainder allocation, NumPy default_rng(1). '
                             'Match 100 controls without replacement from each original 281-control pool '
                             'using the original semantic/event/sequence matching function.',
                'conditions': {}}
    rng = np.random.default_rng(1)
    for condition, spec in source['conditions'].items():
        pools = {seq: [i for i in spec['degraded_indices'] if rows[i]['sequence'] == seq]
                 for seq in sorted({rows[i]['sequence'] for i in spec['degraded_indices']})}
        counts = np.array([len(pool) for pool in pools.values()])
        quota = counts / counts.sum() * 100
        allocation = np.floor(quota).astype(int)
        for j in np.argsort(-(quota-allocation), kind='stable')[:100-allocation.sum()]:
            allocation[j] += 1
        chosen = sorted(int(i) for pool, count in zip(pools.values(), allocation)
                        for i in rng.choice(pool, int(count), replace=False))
        controls = sorted(matched_controls(rows, hist, chosen, spec['control_indices'], condition))
        assert len(set(chosen)) == len(set(controls)) == 100
        assert not set(chosen) & set(controls)
        manifest['conditions'][condition] = {
            'degraded_indices': chosen, 'control_indices': controls,
            'degraded_samples': [rows[i]['sample_id'] for i in chosen],
            'control_samples': [rows[i]['sample_id'] for i in controls],
            'sequence_counts': dict(zip(pools, map(int, allocation))),
        }
    manifest['unique_indices'] = sorted({i for spec in manifest['conditions'].values()
                                        for key in ('degraded_indices', 'control_indices') for i in spec[key]})
    path = OUT / 'frozen_subsets_100.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest, 'Existing manifest differs; do not overwrite.'
    else:
        save(path, manifest)
    return manifest


def infer(role, run, indices, rows):
    cfg = load_config(run / 'config.yaml')
    _disable_pretrained(cfg)
    checkpoint_path = run / 'best.pth'
    checksum = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = build_model('rcafnet', cfg, device='cuda:0').eval()
    model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    epoch = int(checkpoint['epoch'])
    del checkpoint
    _, raw = build_datasets('dsec', cfg)
    assert len(raw) == len(rows)
    assert [sequence_frame_sample(item)[2] for item in raw.data_name] == [r['sample_id'] for r in rows]
    loader = DataLoader(Subset(NormalizedMapDataset(raw), indices), batch_size=4,
                        shuffle=False, num_workers=4, pin_memory=True)
    matrices = []
    digest = hashlib.sha256()
    start = time.perf_counter()
    with torch.inference_mode():
        for step, (image, event, target) in enumerate(loader):
            # Verify exact RGB/GT equivalence despite different event configurations.
            for rgb, gt in zip(image, target):
                digest.update(rgb.numpy().tobytes())
                digest.update(gt.numpy().tobytes())
            image, event, target = [v.to('cuda:0', non_blocking=True) for v in (image, event, target)]
            pred = model(image, event).argmax(1)
            batch = target.shape[0]
            valid = (target >= 0) & (target < 11) & (target != cfg.ignore_index)
            offset = torch.arange(batch, device=target.device)[:, None, None] * 121
            encoded = offset + target.long()*11 + pred
            cm = torch.bincount(encoded[valid], minlength=batch*121).reshape(batch, 11, 11)
            matrices.extend(cm.cpu().numpy())
            if step % 25 == 0 or (step+1)*4 >= len(indices):
                print(f'{role}: {min((step+1)*4, len(indices))}/{len(indices)}', flush=True)
    torch.cuda.synchronize()
    assert sha(checkpoint_path) == checksum, 'Checkpoint changed during inference.'
    matrices = np.stack(matrices)
    assert len(matrices) == len(indices)
    np.savez_compressed(OUT / f'{role}_frame_confusions.npz', indices=indices,
                        confusion_matrices=matrices)
    metadata = {'run_dir': str(run), 'checkpoint': str(checkpoint_path), 'sha256': checksum,
                'epoch': epoch, 'modality': cfg.modality, 'encoder_type': cfg.encoder.get('type', 'rcafnet'),
                'config_sha256': sha(run / 'config.yaml'), 'precision': 'FP32, no autocast, no TF32',
                'gpu': torch.cuda.get_device_name(), 'elapsed_seconds': time.perf_counter()-start,
                'rgb_gt_sha256': digest.hexdigest(), 'unique_frames': len(indices)}
    save(OUT / f'{role}_inference.json', metadata)
    del model
    torch.cuda.empty_cache()
    return matrices, metadata


def summarize(manifest, rows, confusions, metadata):
    indices = manifest['unique_indices']
    block_length = int(manifest.get('bootstrap_block_length', 20))
    lookup = {idx: j for j, idx in enumerate(indices)}
    scores = {role: np.full(len(rows), np.nan) for role in RUNS}
    score_rows = []
    for idx in indices:
        entry = {'index': idx, 'sample_id': rows[idx]['sample_id']}
        for role in RUNS:
            score = metrics_from_confusion(confusions[role][lookup[idx]])['miou']
            scores[role][idx] = score
            entry[f'{role}_miou'] = score
        score_rows.append(entry)
    write_rows(OUT / 'per_frame_scores.csv', score_rows)
    gap = scores['rcafnet'] - scores['image_only']
    results = {'metric_protocol': 'mean of per-frame mIoU; aggregate confusion mIoU also saved separately',
               'selection_uses_predictions': manifest.get('selection_uses_predictions', False),
               'selection_description': manifest.get('selection', ''),
               'models': metadata, 'unique_frames': len(indices),
               'bootstrap': {'repetitions': 5000, 'block_length': block_length,
                             'type': 'sequence-stratified circular moving-block, selected-sample order'},
               'conditions': {}}
    table = []
    for k, (condition, spec) in enumerate(manifest['conditions'].items()):
        result = {}
        for name in ('degraded', 'control'):
            subset = spec[f'{name}_indices']
            positions = [lookup[i] for i in subset]
            result[name] = {
                'count': len(subset),
                'image_only_miou': float(scores['image_only'][subset].mean()),
                'rcafnet_miou': float(scores['rcafnet'][subset].mean()),
                'model_gap': float(gap[subset].mean()),
                'model_gap_95ci': bootstrap_interval(gap, subset, rows, 5000, block_length, 1+100*k+(name=='control')),
                'aggregate_confusion_metrics': {role: metrics_from_confusion(confusions[role][positions].sum(0))
                                               for role in RUNS},
            }
        extra = result['degraded']['model_gap'] - result['control']['model_gap']
        rng = np.random.default_rng(1001+k)
        draws = [_block_bootstrap_mean(gap, spec['degraded_indices'], rows, rng, block_length) -
                 _block_bootstrap_mean(gap, spec['control_indices'], rows, rng, block_length) for _ in range(5000)]
        ci = np.quantile(draws, [.025, .975]).tolist()
        result.update(complementarity_gain=extra, complementarity_gain_95ci=ci)
        results['conditions'][condition] = result
        table.append({'condition': condition, 'count': len(spec['degraded_indices']),
                      'image_only_miou_pct': result['degraded']['image_only_miou']*100,
                      'rcafnet_miou_pct': result['degraded']['rcafnet_miou']*100,
                      'degraded_gap_pp': result['degraded']['model_gap']*100,
                      'control_image_only_miou_pct': result['control']['image_only_miou']*100,
                      'control_rcafnet_miou_pct': result['control']['rcafnet_miou']*100,
                      'control_gap_pp': result['control']['model_gap']*100,
                      'extra_gain_pp': extra*100, 'extra_ci_low_pp': ci[0]*100, 'extra_ci_high_pp': ci[1]*100})
    save(OUT / 'results.json', results)
    write_rows(OUT / 'summary.csv', table)
    text = ['# DSEC自然退化子集：每类100帧重新评估', '',
            '## 评估口径', '',
            '- 两个指定实验的 best.pth，严格加载全部权重，GPU FP32重新推理；无增强、无训练。',
            '- redesign_r1 实际为 Image-only（CE，MiT-B2）；ce_lovasz 为双模态 RCAFNet。',
            '- 每种退化100帧、匹配对照100帧，而非整个实验总共100帧。',
            f'- 六组存在重叠，实际每个模型推理 {len(indices)} 张不同图像。',
            '- 从原冻结退化池按序列比例、固定种子1抽样，三个序列分配为14/42/44帧。',
            '- 从原冻结对照池重新匹配100帧，未按预测指标选择样本，也未重复抽样挑选最佳结果。',
            '- 主表沿用旧表的逐帧mIoU算术平均；标准聚合混淆矩阵mIoU另存results.json，不能混用。',
            '- 95%区间采用原来的按序列分层循环移动块Bootstrap，5000次，块长20个已选样本，非20个连续原始帧；100帧下区间仅作探索性参考。',
            '- 两个模型的RGB和真值逐像素校验一致，RCAFNet事件预处理保持其训练配置。', '',
            '## 逐帧平均mIoU统计表', '',
            '| 场景 | n | Image-only (%) | RCAFNet (%) | 退化组优势 (pp) | 对照组优势 (pp) | 额外增益 (pp) | 额外增益95% CI (pp) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        text.append(f'| {NAMES[r["condition"]]} | {r["count"]} | {r["image_only_miou_pct"]:.2f} | {r["rcafnet_miou_pct"]:.2f} | '
                    f'{r["degraded_gap_pp"]:+.2f} | {r["control_gap_pp"]:+.2f} | {r["extra_gain_pp"]:+.2f} | '
                    f'[{r["extra_ci_low_pp"]:.2f}, {r["extra_ci_high_pp"]:.2f}] |')
    text += ['', '优势=RCAFNet−Image-only；额外增益=退化组优势−匹配对照组优势。', '',
             '局部高光饱和不等同于全局过曝；低清晰度候选不等同于已确认的方向性运动模糊。',
             '该比较更换了Image-only检查点并缩小子集，不能将与旧表的变化全部归因于抽样数量。',
             '小规模观察性子集不能单独证明事件模态的因果性补偿效果。', '',
             '## 检查点记录', '']
    for role, meta in metadata.items():
        text.append(f'- {role}: `{meta["run_dir"]}`, checkpoint epoch字段={meta["epoch"]}, SHA256={meta["sha256"]}')
    if 'report_replacements' in manifest:
        text = [manifest['report_replacements'].get(line, line) for line in text]
    text.extend(manifest.get('report_extra_lines', []))
    (OUT / 'REPORT_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps(table, ensure_ascii=False, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_quality_rows(AUDIT / 'quality_scores.csv')
    manifest = freeze(rows)
    print(f'Frozen {len(manifest["unique_indices"])} unique frames, 100 per group', flush=True)
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback.'
    torch.set_num_threads(4)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    confusions, metadata = {}, {}
    for role, run in RUNS.items():
        confusions[role], metadata[role] = infer(role, run, manifest['unique_indices'], rows)
    assert metadata['image_only']['rgb_gt_sha256'] == metadata['rcafnet']['rgb_gt_sha256']
    summarize(manifest, rows, confusions, metadata)


if __name__ == '__main__':
    main()
