from pathlib import Path

import toml
import cv2
import matplotlib.pyplot as plt

from modules.utils import imread_jpn, visualize_bbox
from modules.load_anno_voc import AnnoXmlToList


def main() -> None:
    """物体検出のアノテーションデータを可視化するプログラム
    """
    # 設定ファイルの読み込み
    with open("settings.toml", mode='r', encoding='utf-8') as f:
        settings = toml.load(f)
        
        data_type = settings["data_type"]
        if data_type == "voc":
            ext = "xml"
            load_anno = AnnoXmlToList()
        else:
            raise ValueError("無効な\"data_type\"です。")
        
        anno_path = Path(settings["anno_path"])
        if anno_path.is_dir():
            anno_file_list = list(anno_path.glob(f"*.{ext}"))
        elif anno_path.is_file():
            anno_file_list = [anno_path]
        else:
            raise FileNotFoundError("アノテーションファイルまたはアノテーションファイルが入ったディレクトリを指定してください。")
        
        img_path = Path(settings["img_path"])
        
        show = settings["show"]
        save = settings["save"]
        output_path = Path(settings["output_path"])
        if not output_path.is_dir():
            output_path.mkdir(parents=True)
        
    for anno_file_path in anno_file_list:
        # アノテーションデータの読み込み
        img_name, bboxes, labels = load_anno(anno_file_path)
        
        # 画像の読み込み
        img_file_path = img_path.joinpath(img_name)
        img = imread_jpn(img_file_path)
        
        # bboxを可視化
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualize_bbox(img_rgb, bboxes, labels, load_anno.classes)
        
        # 可視化画像を別Windowで表示
        if show:
            plt.show()
        
        # 可視化画像を保存
        if save:
            plt.savefig(output_path.joinpath(img_name))
            
        plt.close()
    
    # 読み込んだクラスの一覧を表示
    load_anno.show_classes_info()


if __name__ == "__main__":
    main()
