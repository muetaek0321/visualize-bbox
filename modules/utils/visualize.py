from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


__all__ = ["visualize_bbox"]

# 定数
COLORS = [
    [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
    [255, 128, 0, 255], [128, 255, 0, 255], [128, 0, 255, 255],
    [255, 0, 128, 255], [0, 255, 128, 255], [0, 128, 255, 255], 
]

def visualize_bbox(
    img: np.ndarray,
    bboxes: np.ndarray | list[list[int]],
    labels: np.ndarray | list[int],
    classes: list[str],
    dataset_type: str = "voc"
) -> None:
    """BBoxを可視化した画像を出力
    
    Args:
        img (np.ndarray): 画像データ
        bboxes (np.ndarray,list): BBoxが格納された配列
    """
    img_vis = deepcopy(img)
    plt.figure(figsize=(7, 7*(img.shape[0]/img.shape[1])))
    plt.imshow(img_vis)
    currentAxis = plt.gca()
    
    # BBoxを可視化
    for bbox, label in zip(bboxes, labels):
        if dataset_type == "coco":
            xy = (int(bbox[0]), int(bbox[1]))
            width = int(bbox[2])
            height = int(bbox[3])
        elif dataset_type == "voc":
            xy = (int(bbox[0]), int(bbox[1]))
            width = int(bbox[2]) - int(bbox[0])
            height = int(bbox[3]) - int(bbox[1])
        
        # ラベルに応じて表示するテキストと矩形の色を決定
        label_text = classes[label]
        color = np.array(COLORS[label]) / 255
        
        # 長方形を描画する
        currentAxis.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor=color, linewidth=2))

        # 長方形の枠の左上にラベルを描画する
        currentAxis.text(xy[0], xy[1], label_text, bbox={
                         'facecolor': color, 'alpha': 0.5})
        
        plt.tight_layout()
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
