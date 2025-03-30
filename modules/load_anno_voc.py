"""
参考：
・アノテーション読み込みクラス
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


__all__ = ["AnnoXmlToList"]


class AnnoXmlToList(object):
    """VOC形式のデータセットのXMLファイルの読み込みクラス"""

    def __init__(
        self, 
    ) -> None:
        """コンストラクタ
        
        Args:
            classes (list): 検出するクラス名のリスト
        """
        self.classes = []

    def __call__(
        self, 
        xml_path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """アノテーションファイルを読み込み
        
        Args:
            xml_path (str,Path): アノテーションファイルのパス
            
        Returns:
            tuple: アノテーションデータ(bbox, label)
        """
        bbox_datas = []
        label_datas = []

        # xmlファイル（アノテーションファイル）を読み込む
        xml = ET.parse(xml_path).getroot()
        
        # 画像ファイル名を取得
        img_name = xml.find('filename').text

        # 画像内にある物体（object）の数だけループする
        for obj in xml.iter('object'):

            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            name = obj.find('name').text.lower().strip()  # 物体名
            bbox = obj.find('bndbox')  # バウンディングボックスの情報

            # アノテーションの xmin, ymin, xmax, ymaxを取得
            # VOCは原点が(1,1)なので1を引き算して（0, 0）に
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = [int(bbox.find(pt).text)-1 for pt in pts]
            bbox_datas.append(bndbox)

            # アノテーションのクラス名のindexを取得して追加
            if name in self.classes:
                label_idx = self.classes.index(name)
            else:
                label_idx = len(self.classes)
                self.classes.append(name)
            label_datas.append(label_idx)

        return img_name, np.array(bbox_datas), np.array(label_datas)
    
    def show_classes_info(
        self
    ) -> None:
        """アノテーションのクラス情報を表示
        """
        for num, class_name in enumerate(self.classes):
            print(f"{num}: {class_name}")
    
    