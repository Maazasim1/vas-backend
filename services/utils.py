import uuid

def generate_unique_id():
    return str(uuid.uuid4())

def xyxy_xywh(x1, y1, x2, y2):
    x= x1
    y= y1
    w = x2-x1
    h = y2-y1

    return x,y,w,h

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = xyxy_xywh(box1[0], box1[1], box1[2], box1[3])
    x2, y2, w2, h2 = xyxy_xywh(box2[0], box2[1], box2[2], box2[3])

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def filter_batches(bboxes, confidences, iou_threshold):
    sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)

    filtered_bboxes = []

    while len(sorted_indices) > 0:
        index = sorted_indices.pop(0)
        bbox = bboxes[index]
        filtered_bboxes.append(bbox)

        for i in reversed(range(len(sorted_indices))):
            iou = calculate_iou(bbox, bboxes[sorted_indices[i]])
            if iou > iou_threshold:
                sorted_indices.pop(i)

    return filtered_bboxes

def face_detection(face_results):
    face_bboxes = []
    face_confs = []

    if len(face_results) > 0:
        for face_result in face_results:
            try:
                face_bboxes.append(face_result.boxes.xyxy.tolist())
                if face_result.boxes.conf.numel() == 1:
                    face_confs.append(face_result.boxes.conf.item())
                elif face_result.boxes.conf.numel() > 1:
                    face_confs.extend(face_result.boxes.conf.tolist())
            except:
                continue

        face_bboxes = face_bboxes[0]
        
        if len(face_bboxes) > 1:
            face_bboxes = filter_batches(face_bboxes, face_confs, iou_threshold=0.6)

    return face_bboxes
