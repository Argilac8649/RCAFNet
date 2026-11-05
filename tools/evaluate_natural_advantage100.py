"""Explicitly performance-selected advantage cases, not unbiased robustness evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from tools import evaluate_natural_subset100 as base
from tools.dsec_natural_degradation import load_quality_rows, metrics_from_confusion, write_rows, matched_controls

OUT = base.AUDIT / 'advantage100_20260910'
PREVIOUS = base.AUDIT / 'subset100_20260910'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    rows = load_quality_rows(base.AUDIT / 'quality_scores.csv')
    source = json.loads((base.AUDIT / 'frozen_subsets.json').read_text(encoding='utf-8'))
    # Infer the full pre-existing candidate/control pools before ranking any frame.
    candidates = sorted({i for s in source['conditions'].values()
                         for key in ('degraded_indices', 'control_indices') for i in s[key]})
    previous = json.loads((PREVIOUS / 'frozen_subsets_100.json').read_text(encoding='utf-8'))['unique_indices']
    remaining = sorted(set(candidates) - set(previous))
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    matrices, metadata = {}, {}
    for role, run in base.RUNS.items():
        old_meta = json.loads((PREVIOUS / f'{role}_inference.json').read_text(encoding='utf-8'))
        assert base.sha(run / 'best.pth') == old_meta['sha256'], 'Checkpoint changed; cached scores invalid.'
        assert base.sha(run / 'config.yaml') == old_meta['config_sha256']
        old = np.load(PREVIOUS / f'{role}_frame_confusions.npz')
        old_map = dict(zip(old['indices'].tolist(), old['confusion_matrices']))
        assert set(old_map) == set(previous)
        new_matrices, meta = base.infer(role, run, remaining, rows)
        assert meta['sha256'] == old_meta['sha256']
        old_map.update(zip(remaining, new_matrices))
        matrices[role] = old_map
        metadata[role] = {**meta, 'reused_cache': str(PREVIOUS / f'{role}_frame_confusions.npz'),
                          'reused_frames': len(previous), 'total_candidate_control_frames': len(candidates)}
        np.savez_compressed(OUT / f'{role}_all_pool_confusions.npz', indices=candidates,
                            confusion_matrices=np.stack([old_map[i] for i in candidates]))
    assert metadata['image_only']['rgb_gt_sha256'] == metadata['rcafnet']['rgb_gt_sha256']
    scores = {role: {i: metrics_from_confusion(matrices[role][i])['miou'] for i in candidates}
              for role in base.RUNS}
    gap = {i: scores['rcafnet'][i]-scores['image_only'][i] for i in candidates}
    manifest = {'selection_uses_predictions': True,
                'purpose': 'Post-hoc RCAFNet advantage-case analysis; not representative or unbiased robustness evaluation',
                'ranking': 'Descending per-frame mIoU(RCAFNet)-mIoU(Image-only), index as tie breaker; top100 per original281 degraded pool',
                'source_manifest_sha256': base.sha(base.AUDIT / 'frozen_subsets.json'), 'conditions': {}}
    hist = np.load(base.AUDIT / 'label_histograms.npy')
    result = {'selection_uses_predictions': True, 'metric_protocol': 'mean per-frame mIoU',
              'models': metadata, 'conditions': {},
              'inference_warning': 'No selection-unadjusted confidence intervals or significance claims; performance-guided selection makes gains optimistically biased.'}
    table = []
    for condition, spec in source['conditions'].items():
        ranked = sorted(spec['degraded_indices'], key=lambda i: (-gap[i], i))
        selected = ranked[:100]
        control = sorted(matched_controls(rows, hist, sorted(selected), spec['control_indices'], condition))
        def entry(i):
            return {'index': i, 'sample_id': rows[i]['sample_id'], 'sequence': rows[i]['sequence'],
                    'image_only_miou_pct': scores['image_only'][i]*100,
                    'rcafnet_miou_pct': scores['rcafnet'][i]*100, 'gap_pp': gap[i]*100,
                    'quality_score': rows[i][spec['score']]}
        write_rows(OUT / f'{condition}_top100.csv', [dict(rank=r+1, **entry(i)) for r, i in enumerate(selected)])
        write_rows(OUT / f'{condition}_all281_ranked.csv', [dict(rank=r+1, **entry(i)) for r, i in enumerate(ranked)])
        manifest['conditions'][condition] = {'selected_indices_ranked': selected, 'control_indices': control,
                                            'samples': [rows[i]['sample_id'] for i in selected]}
        res = {'candidate_count': len(ranked), 'selected_count': len(selected),
               'positive_gap_count_in_candidates': sum(gap[i] > 0 for i in ranked),
               'positive_gap_count_selected': sum(gap[i] > 0 for i in selected),
               'minimum_selected_gap_pp': min(gap[i] for i in selected)*100,
               'maximum_selected_gap_pp': max(gap[i] for i in selected)*100,
               'sequence_counts': {seq: sum(rows[i]['sequence'] == seq for i in selected)
                                   for seq in sorted({rows[i]['sequence'] for i in selected})}}
        for name, indices in [('selected', selected), ('control', control), ('all_candidates', ranked)]:
            res[name] = {f'{role}_miou_pct': float(np.mean([scores[role][i] for i in indices])*100)
                         for role in base.RUNS}
            res[name]['gap_pp'] = float(np.mean([gap[i] for i in indices])*100)
        res['descriptive_difference_of_gaps_pp'] = res['selected']['gap_pp']-res['control']['gap_pp']
        result['conditions'][condition] = res
        table.append({'condition': condition, **res['selected'], 'control_gap_pp': res['control']['gap_pp'],
                      'descriptive_difference_of_gaps_pp': res['descriptive_difference_of_gaps_pp'],
                      'positive_gap_count_in_candidates': res['positive_gap_count_in_candidates'],
                      'minimum_selected_gap_pp': res['minimum_selected_gap_pp']})
    base.save(OUT / 'selected_advantage100.json', manifest)
    base.save(OUT / 'results.json', result)
    write_rows(OUT / 'summary.csv', table)
    text = ['# DSEC自然退化候选中的RCAFNet优势场景（事后筛选）', '',
            '**这是按模型表现选择的优势案例分析，不是代表性自然退化子集评估。**', '',
            '## 方法', '',
            '- 使用用户指定的两个best.pth。redesign_r1是Image-only，ce_lovasz是双模态RCAFNet。',
            '- 每类保持原先281帧自然退化候选池不变，按RCAFNet减Image-only的逐帧mIoU差值降序选前100帧。',
            '- 无序列配额，无随机种子重复筛选；同分按原索引排序。三个场景可有重叠。',
            '- 复用本次相同检查点、相同FP32设置的GPU预测，缺失候选和对照在GPU补齐；严格校验检查点SHA256及RGB/真值一致性。',
            '- 分数为逐帧mIoU的算术平均，不是聚合混淆矩阵mIoU。',
            '- 匹配对照由原冻结对照池按语义分布、事件活动度及序列匹配，低清晰度额外匹配亮度。',
            '- 由于依据预测筛选，优势和差分都存在选择偏差；不提供未经选择校正的置信区间或显著性判断。', '',
            '## 选中100帧的描述性结果', '',
            '| 场景 | Image-only (%) | RCAFNet (%) | 所选场景优势 (pp) | 对照优势 (pp) | 描述性差分 (pp) |',
            '|---|---:|---:|---:|---:|---:|']
    for t in table:
        text.append(f'| {base.NAMES[t["condition"]]} | {t["image_only_miou_pct"]:.2f} | {t["rcafnet_miou_pct"]:.2f} | '
                    f'{t["gap_pp"]:+.2f} | {t["control_gap_pp"]:+.2f} | {t["descriptive_difference_of_gaps_pp"]:+.2f} |')
    text += ['', '## 原候选池与选择覆盖', '',
             '| 场景 | 原候选池平均优势 (pp) | RCAFNet占优帧数/281 | 所选100帧最小优势 (pp) |',
             '|---|---:|---:|---:|']
    for c, r in result['conditions'].items():
        text.append(f'| {base.NAMES[c]} | {r["all_candidates"]["gap_pp"]:+.2f} | '
                    f'{r["positive_gap_count_in_candidates"]}/281 | {r["minimum_selected_gap_pp"]:+.2f} |')
    text += ['', '## 论文使用边界', '',
             '可表述为：为分析模型的优势场景，在各自然退化候选池中按逐帧mIoU差值选取100帧进行事后案例分析。',
             '不应表述为：随机或代表性选择100帧证明RCAFNet在自然退化下稳定提升这些百分点。',
             '局部高光饱和、相对低照度和低清晰度是输入质量筛选标签，不能单凭该分析推出事件对过曝、欠曝或运动模糊的因果性补偿。',
             '完整排序表同时保留全部281帧，便于检查未选中的样本与模型失败案例。']
    (OUT / 'REPORT_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps(result['conditions'], indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
