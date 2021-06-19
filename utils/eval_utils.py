"""
    Helper functions used for evaluation of loss and other metrics
"""

def bb_intersection_over_union(boxA, boxB):
    """
        Computes IoU (Intersection over Union for 2 given bounding boxes)

        Args:
            boxA (list): A list of 4 elements holding bounding box coordinates (x1, y1, x2, y2)
            boxB (list): A list of 4 elements holding bounding box coordinates (x1, y1, x2, y2)

        Returns:
            iou (float): Overlap between 2 bounding boxes in terms of overlap
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of both boxes
    # intersection area / areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def top_bbox_from_scores(bboxes, scores):
    """
        Returns the top matching bounding box based on scores

        Args:
            bboxes (list): List of bounding boxes for each object
            scores (list): List of scores corresponding to bounding boxes given by bboxes

        Returns:
            matched_bbox: The bounding box with the maximum score
    """
    bbox_scores = [(bbox, score) for bbox, score in zip(bboxes, scores)]
    sorted_bbox_scores = sorted(bbox_scores, key=lambda x: x[1], reverse=True)
    matched_bbox = sorted_bbox_scores[0][0]
    return matched_bbox


def is_bbox_overlap(bbox1, bbox2, iou_overlap_threshold):
    """
        Checks if the two bounding boxes overlap based on certain threshold

        Args:
            bbox1: The coordinates of first bounding box
            bbox2: The coordinates of second bounding box
            iou_overlap_threshold: Threshold value beyond which objects are considered overlapping

        Returns:
            Boolean whether two boxes overlap or not
    """
    iou = bb_intersection_over_union(boxA=bbox1, boxB=bbox2)
    if iou >= iou_overlap_threshold:
        return True
    return False
