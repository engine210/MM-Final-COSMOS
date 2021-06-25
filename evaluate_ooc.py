""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

import cv2
import os
from utils.config import *
from utils.text_utils import get_text_metadata
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores

import numpy as np

# Word Embeddings
text_field, word_embeddings, vocab_size = get_text_metadata()

# Models (create model according to text embedding)
if embed_type == 'use':
    # For USE (Universal Sentence Embeddings)
    model_name = 'text_aug'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
else:
    # For Glove and Fasttext Embeddings
    model_name = 'img_lstm_glove_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(use=False, hidden_size=300, embedding_length=word_embeddings.shape[1]).to(device)


def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    checkpoint = torch.load(BASE_DIR + 'models/' + model_name + '.pt')
    
    # paper's (val 0.76)
    # checkpoint = torch.load('/mnt/shared/engine210/MMFinal/models/img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner.pt')

    combined_model.load_state_dict(checkpoint)
    combined_model.to(device)
    combined_model.eval()

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])  # For entire image (global context)
    bbox_classes.append(-1)
    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']

    img_tensor = [torch.tensor(img).to(device)]
    bboxes = [torch.tensor(bbox_list).to(device)]
    bbox_classes = [torch.tensor(bbox_classes).to(device)]

    if embed_type != 'use':
        # For Glove, Fasttext embeddings
        cap1_p = text_field.preprocess(cap1)
        cap2_p = text_field.preprocess(cap2)
        embed_c1 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap1_p]).unsqueeze(
            0).to(device)
        embed_c2 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap2_p]).unsqueeze(
            0).to(device)
    else:
        # For USE embeddings
        embed_c1 = torch.tensor(use_embed([cap1]).numpy()).to(device)
        embed_c2 = torch.tensor(use_embed([cap2]).numpy()).to(device)

        # mimi use USE to calculate text similarity
        # embed_c1_norm = embed_c1.div(torch.norm(embed_c1).expand_as(embed_c1))
        # embed_c2_norm = embed_c2.div(torch.norm(embed_c2).expand_as(embed_c2))
        # use_sim = torch.matmul(embed_c1_norm.squeeze(), embed_c2_norm.squeeze())
        

    with torch.no_grad():
        z_img, z_t_c1, z_t_c2 = combined_model(img_tensor, embed_c1, embed_c2, 1, [embed_c1.shape[1]],
                                               [embed_c2.shape[1]], bboxes, bbox_classes)

    z_img = z_img.permute(1, 0, 2)
    z_text_c1 = z_t_c1.unsqueeze(2)
    z_text_c2 = z_t_c2.unsqueeze(2)

    # Compute Scores
    score_c1 = torch.bmm(z_img, z_text_c1).squeeze()
    score_c2 = torch.bmm(z_img, z_text_c2).squeeze()

    return score_c1, score_c2
    # mimi use USE to calculate text similarity
    # return score_c1, score_c2, use_sim


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
    # mimi use USE to calculate text similarity
    # score_c1, score_c2, use_sim = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])

    # top_bbox_c1 = top_bbox_from_scores(bboxes, score_c1)
    # top_bbox_c2 = top_bbox_from_scores(bboxes, score_c2)
    # mimi theshold
    top_bbox_c1, top_score_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2, top_score_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)

    # mimi visualize the max bbox
    # img_path = os.path.join(DATA_DIR, v_data['img_local_path'])
    # img = cv2.imread(img_path)
    # start_point1 = (int(top_bbox_c1[0]), int(top_bbox_c1[1]))
    # end_point1 = (int(top_bbox_c1[2]), int(top_bbox_c1[3]))
    # start_point2 = (int(top_bbox_c2[0]), int(top_bbox_c2[1]))
    # end_point2 = (int(top_bbox_c2[2]), int(top_bbox_c2[3]))
    # img = cv2.rectangle(img, start_point1, end_point1, (0, 0, 255), 3)
    # img = cv2.rectangle(img, start_point2, end_point2, (0, 255, 0), 3)
    # cv2.imwrite(os.path.join('/home/engine211/MMFinal/MM-Final-COSMOS-mimi', v_data['img_local_path']), img)

    # if bbox_overlap:
    #     # Check for captions with same context : Same grounding with high textual overlap (Not out of context)
    #     if textual_sim >= textual_sim_threshold:
    #         context = 0
    #     # Check for captions with different context : Same grounding with low textual overlap (Out of context)
    #     else:
    #         context = 1
    #     return context
    # else:
    #     # Check for captions with same context : Different grounding (Not out of context)
    #     return 0
    
    # mimi's test 
    if textual_sim < textual_sim_threshold:
        if bbox_overlap or (top_score_c1 < 2) or (top_score_c2 < 2):
            context = 1
        else:
            context = 0 ### ????
    else:
        context = 0
    
    # return context
    # mimi for threshold and use_sim
    return context, top_score_c1, top_score_c2
            


if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    test_samples = read_json_data(os.path.join(DATA_DIR, 'annotations', 'test_data.json'))
    ours_correct = 0
    lang_correct = 0
    
    # test
    lang_uncorrect_ooc_to_nooc = 0
    lang_uncorrect_nooc_to_ooc = 0
    ours_uncorrect_ooc_to_nooc = 0
    ours_uncorrect_nooc_to_ooc = 0

    for i, v_data in enumerate(test_samples):
        actual_context = int(v_data['context_label'])
        language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1

        # pred_context = evaluate_context_with_bbox_overlap(v_data)
        # mimi for threshold
        pred_context, score_c1, score_c2 = evaluate_context_with_bbox_overlap(v_data)
        
        # mimi use USE to calculate text similarity
        # language_context = 0 if float(use_sim) >= 0.5 else 1
        # print(i,': ' ,use_sim, actual_context)
        
        # (1=Out-of-Context, 0=Not-Out-of-Context )
        
        # mimi for eval log
        # ooc_file = open("./eval_results/ooc_score.txt","a")
        # ooc_to_nooc_file = open("./eval_results/ooc_to_nooc_score.txt","a")
        # nooc_to_ooc_file = open("./eval_results/nooc_to_ooc_score.txt","a")
        # ooc_to_nooc_textsim_file = open("./eval_results/ooc_to_nooc_textsim.txt","a")

        # ours model
        if pred_context == actual_context:
            ours_correct += 1
            # if pred_context == 1:
            #     log_text = str(i) + ' correct: ' + str(score_c1) + str(score_c2) + '\n'
            #     ooc_file.write(log_text)
        elif pred_context == 0:
            ours_uncorrect_ooc_to_nooc += 1
            if float(v_data['bert_base_score']) < textual_sim_threshold:
                # log_text = str(i) + ' wrong-ooc-to-nooc: ' + str(score_c1) + str(score_c2) + '\n'
                # ooc_to_nooc_file.write(log_text)
                pass
            else:
                # log_text = str(i) + ' wrong-ooc-to-nooc-text: ' + str(score_c1) + str(score_c2) + '\n'
                # ooc_to_nooc_textsim_file.write(log_text)
                pass

        elif pred_context == 1:
            ours_uncorrect_nooc_to_ooc += 1
            # log_text = str(i) + ' wrong-nooc-to-ooc: ' + str(score_c1) + str(score_c2) + '\n'
            # nooc_to_ooc_file.write(log_text)
            
        # language moel
        if language_context == actual_context:
            lang_correct += 1
        elif language_context == 0:
            lang_uncorrect_ooc_to_nooc += 1
        elif language_context == 1:
            lang_uncorrect_nooc_to_ooc += 1


    print("Cosmos Accuracy", ours_correct / len(test_samples))
    print("Language Baseline Accuracy", lang_correct / len(test_samples))

    print("Ours Uncorrect OOC to NOOC", ours_uncorrect_ooc_to_nooc / len(test_samples))
    print("Ours Uncorrect NOOC to OOC", ours_uncorrect_nooc_to_ooc / len(test_samples))

    print("Language Uncorrect OOC to NOOC", lang_uncorrect_ooc_to_nooc / len(test_samples))
    print("Language Uncorrect NOOC to OOC", lang_uncorrect_nooc_to_ooc / len(test_samples))


    # mimi for eval log
    # ooc_file.close()
    # ooc_to_nooc_file.close()
    # nooc_to_ooc_file.close()
    # ooc_to_nooc_textsim_file.close()