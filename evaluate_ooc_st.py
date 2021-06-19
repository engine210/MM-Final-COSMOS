""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

import cv2
import os
from utils.config import *
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores
from sentence_transformers import SentenceTransformer, util


def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    

    return score_c1, score_c2


def evaluate_context_with_bbox_overlap(v_data):
    """
        Computes predicted out-of-context label for the given data point

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            context_label (int): Returns 0 if its same/similar context and 1 if out-of-context
    """
    bboxes = v_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])

    top_bbox_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)
    if bbox_overlap:
        # Check for captions with same context : Same grounding with high textual overlap (Not out of context)
        if textual_sim >= textual_sim_threshold:
            context = 0
        # Check for captions with different context : Same grounding with low textual overlap (Out of context)
        else:
            context = 1
        return context
    else:
        # Check for captions with same context : Different grounding (Not out of context)
        return 0


if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    test_samples = read_json_data(os.path.join(DATA_DIR, 'mmsys_anns', 'public_test_mmsys_final.json'))
    ours_correct = 0
    lang_correct = 0

    for i, v_data in enumerate(test_samples):
        actual_context = int(v_data['context_label'])
        
        #language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1

        language_model = SentenceTransformer('stsb-mpnet-base-v2')
        cap1_embedding = language_model.encode(v_data['caption1'])
        cap2_embedding = language_model.encode(v_data['caption2'])
        sim = util.pytorch_cos_sim(cap1_embedding, cap2_embedding)
        language_context = 0 if sim >= textual_sim_threshold else 1

        pred_context = evaluate_context_with_bbox_overlap(v_data)

        if pred_context == actual_context:
            ours_correct += 1

        if language_context == actual_context:
            lang_correct += 1

    print("Cosmos Accuracy", ours_correct / len(test_samples))
    print("Language Baseline Accuracy", lang_correct / len(test_samples))
