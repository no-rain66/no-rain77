import cv2
import numpy as np

# 提高了 OBJ_THRESH 以减少误报，降低了 NMS_THRESH 以减少同目标重叠框
OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.50, 0.35, 640

# 全新的 12 类标签
CLASSES = (
    "ganzhe1", "zacao1", "zacao2", "zacao3", "zacao4", "zacao5", "zacao6",
    "ganzhe2", "ganzhe3", "bing1", "bing2", "bing3"
)

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True)) 
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE // grid_h, IMG_SIZE // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

def draw_box_corner(draw_img, top, left, right, bottom, length, corner_color):
    cv2.line(draw_img, (top, left), (top + length, left), corner_color, thickness=3)
    cv2.line(draw_img, (top, left), (top, left + length), corner_color, thickness=3)
    cv2.line(draw_img, (right, left), (right - length, left), corner_color, thickness=3)
    cv2.line(draw_img, (right, left), (right, left + length), corner_color, thickness=3)
    cv2.line(draw_img, (top, bottom), (top + length, bottom), corner_color, thickness=3)
    cv2.line(draw_img, (top, bottom), (top, bottom - length), corner_color, thickness=3)
    cv2.line(draw_img, (right, bottom), (right - length, bottom), corner_color, thickness=3)
    cv2.line(draw_img, (right, bottom), (right, bottom - length), corner_color, thickness=3)

def draw_label_type(draw_img, top, left, label_str, label_color):
    font_scale = 0.6  
    thickness = 1
    (label_w, label_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    box_coords = (top, left, top + label_w + 6, left + label_h + baseline + 4)
    text_pos = (top + 3, left + label_h + 3)
    cv2.rectangle(draw_img, box_coords[0:2], box_coords[2:4], color=label_color, thickness=-1)
    text_color = (255, 255, 255) if label_color == (0, 0, 255) else (0, 0, 0)
    cv2.putText(draw_img, label_str, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness=thickness, lineType=cv2.LINE_AA)

def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        
        top = int((top - padding[0]) / ratio[0])       
        left = int((left - padding[1]) / ratio[1])     
        right = int((right - padding[0]) / ratio[0])   
        bottom = int((bottom - padding[1]) / ratio[1]) 
        
        class_name = CLASSES[cl]
        if class_name.startswith("ganzhe"):
            box_color = (0, 255, 0)       # 纯绿色
        elif class_name.startswith("zacao"):
            box_color = (0, 255, 255)     # 纯黄色
        elif class_name.startswith("bing"):
            box_color = (0, 0, 255)       # 纯红色
        else:
            box_color = (255, 0, 255)     
        
        cv2.rectangle(image, (top, left), (right, bottom), box_color, 2)
        draw_box_corner(image, top, left, right, bottom, 15, box_color)
        label_text = f"{class_name} {score:.2f}"
        draw_label_type(image, top, left, label_text, box_color)

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r 
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad: 
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top)

def myFunc(rknn_lite, IMG):
    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG2, ratio, padding = letterbox(IMG2)
    IMG2 = np.expand_dims(IMG2, 0)
    outputs = rknn_lite.inference(inputs=[IMG2], data_format=['nhwc'])
    boxes, classes, scores = yolov8_post_process(outputs)
    
    coords = []
    target_cls = 0  
    
    if boxes is not None:
        draw(IMG, boxes, scores, classes, ratio, padding)
        if len(boxes) > 0:
            box = boxes[0]
            target_cls = int(classes[0]) 
            
            top = (box[0] - padding[0]) / ratio[0]
            left = (box[1] - padding[1]) / ratio[1]
            right = (box[2] - padding[0]) / ratio[0]
            bottom = (box[3] - padding[1]) / ratio[1]
            cx = int((top + right) / 2)
            cy = int((left + bottom) / 2)
            coords = [cx, cy]

    return {"img": IMG, "coords": coords, "class_id": target_cls}
