"""检测分数评估指标：AUROC / AUPRC / FPR95 / ECE / 阈值下的 P-R-F1。"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("cumsum was found to be unstable: its last element does not correspond to sum")
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0]) or np.array_equal(classes, [-1]) or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]


def compute_ece(y_true, y_score, n_bins=15):
    y_true = np.asarray(y_true)
    y_score = np.clip(np.asarray(y_score), 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bin_edges[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        bin_mask = bin_ids == bin_id
        bin_count = np.sum(bin_mask)
        if bin_count == 0:
            continue
        ece += (bin_count / len(y_true)) * np.abs(np.mean(y_true[bin_mask]) - np.mean(y_score[bin_mask]))
    return ece


def threshold_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def summarize(true_scores, false_scores, metric="vqa", ece_bins=15, thresholds=None):
    """汇总正/负样本分数并打印指标。metric='caption' 额外输出 FPR95/ECE，'vqa' 额外输出 ACC。"""
    true_scores = np.array(true_scores)
    false_scores = np.array(false_scores)
    y_true = np.concatenate([np.ones_like(true_scores), np.zeros_like(false_scores)])
    y_scores = np.concatenate([true_scores, false_scores])
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    if metric == "caption":
        fpr95, thr = fpr_and_fdr_at_recall(y_true, y_scores, recall_level=0.95)
        ece = compute_ece(y_true, y_scores, n_bins=ece_bins)
        print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, FPR95: {fpr95:.4f}, Threshold: {thr}, ECE: {ece:.4f}")
        if thresholds:
            for t in thresholds:
                r, pcs, f1 = threshold_metrics(y_true, y_scores, t)
                print(f"Threshold {t:.2f} - Recall: {r:.4f}, Precision: {pcs:.4f}, F1: {f1:.4f}")
    else:
        acc = len(true_scores) / (len(true_scores) + len(false_scores)) if (len(true_scores) + len(false_scores)) else 0.0
        print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, ACC: {acc:.4f}")
    return auroc, auprc
