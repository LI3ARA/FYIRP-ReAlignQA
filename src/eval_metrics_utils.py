# vqa_eval_utils.py

import re
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
from bert_score import score as bert_score_fn


# --- Text Normalization ---
def normalize_text(s):
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return ''.join(e.lower() for e in s if e.isalnum() or e.isspace()).strip()

# --- F1 Score ---
def simple_f1(pred, truth):
    pred_tokens = normalize_text(pred).split()
    truth_tokens = normalize_text(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

# --- Extract Reason ---
def extract_reason(response):
    match = re.search(r'Reason for answer:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response.strip()

# --- Evaluate DataFrame ---
def evaluate_vqa_responses(
    df,
    pred_col: str = "cleaned_response",
    truth_col: str = "answer",
    extract_reason_flag: bool = True
):
    """
    Evaluate predictions in a DataFrame against the ground truth using various metrics.
    
    Parameters:
    - df: pandas DataFrame containing prediction and ground truth columns
    - pred_col: name of the column with model responses
    - truth_col: name of the column with ground truth answers
    - extract_reason_flag: whether to extract reason from the predicted text

    Returns:
    - Updated DataFrame with metric columns
    - Dictionary of overall scores
    """
    
    if extract_reason_flag:
        df[pred_col] = df[pred_col].apply(extract_reason)

    # Exact Match
    df['exact_match'] = df.apply(
        lambda row: normalize_text(row[pred_col]) == normalize_text(row[truth_col]),
        axis=1
    )
    # BERTScore
    P, R, F1 = bert_score_fn(row[pred_col], row[truth_col], lang="en", verbose=False)
    bert_f1 = F1[0].item()

    # F1 Score
    df['f1'] = df.apply(
        lambda row: simple_f1(row[pred_col], row[truth_col]),
        axis=1
    )

    # BLEU
    smoothie = SmoothingFunction().method4
    df['bleu'] = df.apply(
        lambda row: sentence_bleu([normalize_text(row[truth_col]).split()],
                                  normalize_text(row[pred_col]).split(),
                                  smoothing_function=smoothie),
        axis=1
    )

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    df['rougeL'] = df.apply(
        lambda row: scorer.score(row[truth_col], row[pred_col])['rougeL'].fmeasure,
        axis=1
    )

    # Semantic Similarity
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    emb_truth = model.encode(df[truth_col].tolist(), convert_to_tensor=True)
    emb_pred = model.encode(df[pred_col].tolist(), convert_to_tensor=True)
    cos_sim = util.cos_sim(emb_truth, emb_pred).diagonal().cpu().numpy()
    df['semantic_sim'] = cos_sim

    summary = {
        "Exact Match": df['exact_match'].mean(),
        "F1 Score": df['f1'].mean(),
        "BLEU": df['bleu'].mean(),
        "ROUGE-L": df['rougeL'].mean(),
        "Semantic Similarity": df['semantic_sim'].mean(),
    }

    return df, summary


# --- Visualization ---
def plot_summary(summary):
    plt.figure(figsize=(10, 6))
    plt.bar(summary.keys(), summary.values(), color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Summary")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_summary(summary):
    print("Evaluation Summary:")
    for metric, score in summary.items():
        print(f"{metric}: {score:.4f}")

#  For evaluating a single instance

def evaluate_single_vqa_instance(
    ground_truth: str,
    predictions: dict,
    reference_text: str = None,
    extract_reason_flag: bool = True,
    plot: bool = False
):
    """
    Evaluate multiple predictions against a single ground truth answer and optionally plot results.

    Parameters:
    - ground_truth: the correct answer string
    - predictions: a dict {pipeline_name: answer_str}
    - reference_text: optional reference for context (e.g., question or caption)
    - extract_reason_flag: whether to extract reason from prediction texts
    - plot: whether to generate a bar chart comparing metrics

    Returns:
    - A dictionary of per-pipeline metric results
    """

    results = {}

    if extract_reason_flag:
        predictions = {k: extract_reason(v) for k, v in predictions.items()}

    norm_truth = normalize_text(ground_truth)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    emb_truth = model.encode(norm_truth, convert_to_tensor=True)

    for name, pred in predictions.items():
        norm_pred = normalize_text(pred)

        exact_match = norm_pred == norm_truth
        f1 = simple_f1(pred, ground_truth)

        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([norm_truth.split()], norm_pred.split(), smoothing_function=smoothie)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = scorer.score(norm_truth, norm_pred)['rougeL'].fmeasure

        emb_pred = model.encode(norm_pred, convert_to_tensor=True)
        semantic_sim = util.cos_sim(emb_truth, emb_pred).item()

        results[name] = {
            "Exact Match": float(exact_match),
            "F1 Score": f1,
            "BLEU": bleu,
            "ROUGE-L": rougeL,
            "Semantic Similarity": semantic_sim
        }

    if plot:
        _plot_single_instance_metrics(results)

    return results

def _plot_single_instance_metrics(results_dict):
    """
    Helper function to plot metric comparisons across pipelines
    """
    metrics = list(next(iter(results_dict.values())).keys())
    pipelines = list(results_dict.keys())

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 2 * len(metrics)), sharex=True)

    for i, metric in enumerate(metrics):
        scores = [results_dict[p][metric] for p in pipelines]
        axs[i].bar(pipelines, scores, color='skyblue')
        axs[i].set_ylim(0, 1)
        axs[i].set_ylabel(metric)
        axs[i].set_title(metric)
        axs[i].grid(True, linestyle="--", alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# result = evaluate_single_vqa_instance(
#     ground_truth="yellow",
#     predictions={
#         "pipeline1": "yellow color",
#         "pipeline2": "red",
#         "pipeline3": "It is yellow."
#     },
#     reference_text="What is the color of the banana?",
#     plot=True
# )



def plot_radar_comparison(results_dict):
    metrics = list(next(iter(results_dict.values())).keys())
    labels = metrics
    pipelines = list(results_dict.keys())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name in pipelines:
        scores = list(results_dict[name].values())
        scores += scores[:1]  # loop back to the start
        ax.plot(angles, scores, label=name, marker='o')
        ax.fill(angles, scores, alpha=0.1)

    ax.set_title("Pipeline Comparison (Radar Chart)")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
