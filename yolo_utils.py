import numpy as np
from tensorflow.keras import backend as k
from image_utils import non_max_suppression


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = k.maximum(pred_mins, true_mins)
    intersect_maxes = k.minimum(pred_maxes, true_maxes)
    intersect_wh = k.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = k.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = k.arange(0, stop=conv_dims[0])
    conv_width_index = k.arange(0, stop=conv_dims[1])
    conv_height_index = k.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = k.tile(
        k.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = k.flatten(k.transpose(conv_width_index))
    conv_index = k.transpose(k.stack([conv_height_index, conv_width_index]))
    conv_index = k.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = k.cast(conv_index, k.dtype(feats))

    conv_dims = k.cast(k.reshape(conv_dims, [1, 1, 1, 1, 2]), k.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


def build_yolo_loss(n_classes=20, n_boxes=2, grid_w=7, grid_h=7):
    def yolo_loss(y_true, y_pred):
        label_class = y_true[..., :n_classes]  # ? * 7 * 7 * 20
        label_box = y_true[..., n_classes:n_classes + 4]  # ? * 7 * 7 * 4
        response_mask = y_true[..., n_classes + 4]  # ? * 7 * 7
        response_mask = k.expand_dims(response_mask)  # ? * 7 * 7 * 1

        predict_class = y_pred[..., :n_classes]  # ? * 7 * 7 * 20
        predict_trust = y_pred[..., n_classes:n_classes + 2]  # ? * 7 * 7 * 2
        predict_box = y_pred[..., n_classes + 2:]  # ? * 7 * 7 * 8

        _label_box = k.reshape(label_box, [-1, grid_h, grid_w, 1, 4])
        _predict_box = k.reshape(predict_box, [-1, grid_h, grid_w, n_boxes, 4])

        label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        label_xy = k.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_wh = k.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

        predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
        predict_xy = k.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_wh = k.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_xy_min, predict_xy_max = xywh2minmax(predict_xy,
                                                     predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

        iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
        best_ious = k.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
        best_box = k.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

        box_mask = k.cast(best_ious >= best_box, k.dtype(best_ious))  # ? * 7 * 7 * 2

        no_object_loss = 0.5 * (1 - box_mask * response_mask) * k.square(0 - predict_trust)
        object_loss = box_mask * response_mask * k.square(1 - predict_trust)
        confidence_loss = no_object_loss + object_loss
        confidence_loss = k.sum(confidence_loss)

        class_loss = response_mask * k.square(label_class - predict_class)
        class_loss = k.sum(class_loss)

        _label_box = k.reshape(label_box, [-1, 7, 7, 1, 4])
        _predict_box = k.reshape(predict_box, [-1, 7, 7, 2, 4])

        label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

        box_mask = k.expand_dims(box_mask)
        response_mask = k.expand_dims(response_mask)

        box_loss = 5 * box_mask * response_mask * k.square((label_xy - predict_xy) / 448)
        box_loss += 5 * box_mask * response_mask * k.square((k.sqrt(label_wh) - k.sqrt(predict_wh)) / 448)
        box_loss = k.sum(box_loss)

        loss = confidence_loss + class_loss + box_loss

        return loss

    return yolo_loss


def decode_yolo_output(image, output, labels, n_boxes=2, box_conf_threshold=0.5):
    decoded = []

    grid_h, grid_w = output.shape[:2]

    img_h, img_w = image.shape[:2]

    for row in range(grid_h):
        for col in range(grid_w):
            class_matrix = output[row][col][:len(labels)]
            boxes_matrix = output[row][col][len(labels):]

            label_index = int(np.argmax(class_matrix))

            info = {'label': labels[label_index], "label_conf": class_matrix[label_index]}
            boxes = []
            for b in range(n_boxes):
                cen_x, cen_y, width, height, conf = boxes_matrix[b * 5: (b + 1) * 5]

                if box_conf_threshold <= conf:
                    width *= img_w
                    height *= img_h

                    x = ((cen_x + col) / grid_w) * img_w
                    y = ((cen_y + row) / grid_h) * img_h

                    box = {"conf": conf, "xmin": x, "ymin": y, "xmax": width + x, "ymax": height + y}
                    boxes.append(box)

            if 0 < len(box):
                box = non_max_suppression(boxes, 0.3)[0]
                info["xmin"] = box["xmin"]
                info["ymin"] = box["ymin"]
                info["xmax"] = box["xmax"]
                info["ymax"] = box["ymax"]

            if len(info) > 2:
                decoded.append(info)

    return decoded