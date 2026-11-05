"""Top100 by existing input-quality severity, with no sequence allocation."""
import json
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from tools import evaluate_natural_subset100 as base
from tools.dsec_natural_degradation import load_quality_rows, matched_controls, write_rows


def main(count=100):
    output = base.AUDIT / f'severe{count}_20260910'
    output.mkdir(parents=True, exist_ok=True)
    source_path = base.AUDIT / 'frozen_subsets.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    quality = load_quality_rows(base.AUDIT / 'quality_scores.csv')
    histograms = np.load(base.AUDIT / 'label_histograms.npy')
    manifest = {
        'selection_uses_predictions': False,
        'selection': f'Top{count} descending existing input-degradation score within each original281 pool; '
                     'no sequence quotas; index tie-break; controls matched from original control pool.',
        'source_manifest_sha256': base.sha(source_path),
        'quality_scores_sha256': base.sha(base.AUDIT / 'quality_scores.csv'),
        'conditions': {},
        'report_replacements': {
            '# DSEC自然退化子集：每类100帧重新评估': f'# DSEC自然退化子集：每类退化评分最高的{count}帧',
            '- 两个指定实验的 best.pth，严格加载全部权重，GPU FP32重新推理；无增强、无训练。':
                '- 复用两个指定best.pth已完成的GPU FP32预测缓存；本次仅CPU重新选择与统计，不重新推理或训练。',
            '- 从原冻结退化池按序列比例、固定种子1抽样，三个序列分配为14/42/44帧。':
                f'- 每类在原281帧候选池中，按既有退化评分降序选{count}帧，不设置序列配额；同分按原索引排序。',
            '- 每种退化100帧、匹配对照100帧，而非整个实验总共100帧。':
                f'- 每种退化{count}帧、各自匹配对照{count}帧。',
            '- 从原冻结对照池重新匹配100帧，未按预测指标选择样本，也未重复抽样挑选最佳结果。':
                f'- 从原冻结对照池重新匹配{count}帧，不依据模型表现选样。',
        },
        'report_extra_lines': ['', '## 严重程度定义与边界', '',
            '低照度按既有underexposure_score排序，高光饱和按overexposure_score排序，低清晰度按motion_blur_score排序。',
            '这些既有综合分数含序列内部的百分位排名，因此这里的“最严重”严格指原候选池中现有评分最高，'
            f'并非按跨序列统一物理阈值确定的整个DSEC验证集绝对最严重{count}帧。此次不重新定义退化评分。',
            '不设置序列配额不等于取消对照匹配：对照仍同序列匹配，以减少场景内容差异。',
            '选中样本的序列构成与各帧质量指标保存在清单中，不根据模型表现调整名单。'],
    }
    # Freeze selection BEFORE opening any prediction cache.
    for condition, spec in source['conditions'].items():
        score = spec['score']
        ranked = sorted(spec['degraded_indices'], key=lambda i: (-quality[i][score], i))
        chosen = ranked[:count]
        controls = sorted(matched_controls(quality, histograms, sorted(chosen), spec['control_indices'], condition))
        assert len(set(chosen)) == len(set(controls)) == count
        assert not set(chosen) & set(controls)
        manifest['conditions'][condition] = {
            'score': score, 'degraded_indices': sorted(chosen), 'control_indices': controls,
            'severity_ranked_indices': chosen,
            'sequence_counts': {s: sum(quality[i]['sequence'] == s for i in chosen)
                                for s in sorted({quality[i]['sequence'] for i in chosen})},
            'score_min': float(min(quality[i][score] for i in chosen)),
            'score_max': float(max(quality[i][score] for i in chosen)),
            'degraded_samples': [quality[i]['sample_id'] for i in sorted(chosen)],
            'control_samples': [quality[i]['sample_id'] for i in controls],
        }
        write_rows(output / f'{condition}_top{count}_quality.csv',
                   [dict(severity_rank=j+1, **quality[i]) for j, i in enumerate(chosen)])
        print(condition, manifest['conditions'][condition]['sequence_counts'], flush=True)
    degraded_union = sorted({i for s in manifest['conditions'].values() for i in s['degraded_indices']})
    manifest['unique_degraded_count'] = len(degraded_union)
    manifest['report_extra_lines'] += ['', f'三类合计{3*count}条样本记录，对应{len(degraded_union)}张不同图像。']
    if count == 30:
        assert len(degraded_union) == 90, 'Overlapping severity groups need an explicit allocation rule.'
        manifest['bootstrap_block_length'] = 5
        manifest['report_replacements']['- 95%区间采用原来的按序列分层循环移动块Bootstrap，5000次，块长20个已选样本，非20个连续原始帧；100帧下区间仅作探索性参考。'] = (
            '- 95%区间采用按序列分层循环移动块Bootstrap，5000次，块长改为5个已选样本；'
            '每组仅30帧，原20样本块可能覆盖整个序列层而导致区间退化。仍仅作探索性参考，不作强显著性论断。')
    indices = sorted({i for s in manifest['conditions'].values()
                      for key in ('degraded_indices', 'control_indices') for i in s[key]})
    manifest['unique_indices'] = indices
    base.save(output / f'frozen_severe{count}.json', manifest)
    cache = base.AUDIT / 'advantage100_20260910'
    cached_result = json.loads((cache / 'results.json').read_text(encoding='utf-8'))
    metadata = cached_result['models']
    confusions = {}
    for role, run in base.RUNS.items():
        assert metadata[role]['run_dir'] == str(run)
        assert base.sha(run / 'best.pth') == metadata[role]['sha256'], 'Checkpoint changed from evaluated version.'
        assert base.sha(run / 'config.yaml') == metadata[role]['config_sha256']
        path = cache / f'{role}_all_pool_confusions.npz'
        with np.load(path) as data:
            lookup = {int(i): j for j, i in enumerate(data['indices'])}
            confusions[role] = data['confusion_matrices'][[lookup[i] for i in indices]]
        metadata[role] = {**metadata[role], 'statistics_cache': str(path), 'cache_sha256': base.sha(path)}
    base.OUT = output
    base.summarize(manifest, quality, confusions, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, choices=[30, 100], default=100)
    main(parser.parse_args().count)
