""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

import os
from io import BytesIO
from PIL import Image
from utils.config import *
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-32')

def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    img = Image.open(img_path)
    img_shape = img.size

    bbox_list = v_data['maskrcnn_bboxes']
    bbox_list.append([0, 0, img_shape[0], img_shape[1]])  # For entire image (global context)

    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']
    # cap1 = v_data['caption1']
    # cap2 = v_data['caption2']

    # print(cap1)
    # print(cap2)

    score_c1 = []
    score_c2 = []

    for i, box in enumerate(bbox_list):
        box = list(map(int, box))
        if box[0] == box[2]: box[2] += 1
        if box[1] == box[3]: box[3] += 1

        crop_img = img.crop(box)
        b = BytesIO()
        crop_img.save(b,format="jpeg")
        crop_img = Image.open(b)
        # crop_img.save('tmp'+str(i)+'.jpg')
        crop_img_emb = model.encode(crop_img)
        text_emb = model.encode([cap1, cap2])

        # #Compute cosine similarities 
        cos_scores = util.cos_sim(crop_img_emb, text_emb)
        # print(i, cos_scores.numpy()[0])
        score_c1.append(cos_scores.numpy()[0][0])
        score_c2.append(cos_scores.numpy()[0][1])

    print(v_data["img_local_path"], score_c1.index(max(score_c1)), score_c2.index(max(score_c2)))
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
    ooc_to_nooc = 0
    nooc_to_ooc = 0

    for i, v_data in enumerate(test_samples):
        actual_context = int(v_data['context_label'])
        
        language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1

        pred_context = evaluate_context_with_bbox_overlap(v_data)

        if pred_context == actual_context:
            ours_correct += 1
        else:
            if actual_context == 0:
                nooc_to_ooc += 1
            else:
                ooc_to_nooc += 1
                

        if language_context == actual_context:
            lang_correct += 1

    print("Cosmos Accuracy", ours_correct / len(test_samples))
    print("Language Baseline Accuracy", lang_correct / len(test_samples))
    print("Total Data Size,", len(test_samples))
    print("OOC to NOOC,", ooc_to_nooc)
    print("NOOC to OOC,", nooc_to_ooc)
