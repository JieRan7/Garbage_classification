import xml.etree.ElementTree as ET
import argparse
import os
from tqdm import tqdm


def get_label_set(opt):
    all_labels_dict = {}

    # 检查是否是YOLO格式目录
    if os.path.exists(os.path.join(opt.dir, "labels")):
        # YOLO格式处理
        labels_dir = os.path.join(opt.dir, "labels")
        for txt_name in tqdm(os.listdir(labels_dir)):
            if not txt_name.endswith(".txt"):
                continue
            txt_path = os.path.join(labels_dir, txt_name)
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    class_id = line.strip().split()[0]  # YOLO格式第一个值是类别ID
                    # 读取classes.txt获取类别名称
                    with open(os.path.join(opt.dir, "classes.txt"), 'r') as cls_file:
                        classes = [line.strip() for line in cls_file.readlines()]
                    class_name = classes[int(class_id)]
                    all_labels_dict[class_name] = all_labels_dict.get(class_name, 0) + 1
    else:
        # 原始XML格式处理（保留原有逻辑）
        xmls_file_list = os.listdir(opt.dir)
        for xml_file_name in xmls_file_list:
            xml_file_dir = os.path.join(opt.dir, xml_file_name, 'Annotations')
            if not os.path.exists(xml_file_dir):
                continue
            xmls_list = os.listdir(xml_file_dir)
            for xml_name in tqdm(xmls_list):
                if not xml_name.endswith('.xml'):
                    continue
                xml_path = os.path.join(xml_file_dir, xml_name)
                tree = ET.parse(xml_path)
                roots = tree.getroot()
                for element in roots.findall('object'):
                    bndname = element.find('name').text
                    all_labels_dict[bndname] = all_labels_dict.get(bndname, 0) + 1

    # 打印统计结果
    print("\nLabel Statistics:")
    total = 0
    for key, value in all_labels_dict.items():
        print(f"{key} : {value}")
        total += value
    print(f"Total examples: {total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='D:\project_1\datarubbish', help='data dir')
    parser.add_argument('--get_labels', type=bool, default=True, help='get dataset data labels')
    opt = parser.parse_args()
    get_label_set(opt)