import os
import shutil
import random
import cv2
from tqdm import tqdm
import argparse


def verify_yolo_dataset(data_dir):
    """验证YOLO格式数据集完整性"""
    required_dirs = ['images', 'labels']
    required_files = ['classes.txt']

    # 检查必需目录和文件
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            raise ValueError(f"缺少必需目录: {dir_path}")

    for file_name in required_files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise ValueError(f"缺少必需文件: {file_path}")

    # 检查图片和标签文件对应关系
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    # 获取不带扩展名的文件名
    image_names = {os.path.splitext(f)[0] for f in image_files}
    label_names = {os.path.splitext(f)[0] for f in label_files}

    # 检查是否有图片没有对应的标签文件
    missing_labels = image_names - label_names
    if missing_labels:
        print(f"警告: {len(missing_labels)}张图片没有对应的标签文件")

    # 检查是否有标签文件没有对应的图片
    missing_images = label_names - image_names
    if missing_images:
        print(f"警告: {len(missing_images)}个标签文件没有对应的图片")

    return True


def split_yolo_dataset(data_dir, output_dir, val_ratio=0.2):
    """分割YOLO格式数据集为训练集和验证集"""
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    for dir_path in [train_dir, val_dir]:
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # 读取classes.txt
    classes_file = os.path.join(data_dir, 'classes.txt')
    shutil.copy(classes_file, os.path.join(train_dir, 'classes.txt'))
    shutil.copy(classes_file, os.path.join(val_dir, 'classes.txt'))

    # 获取所有图片文件
    images_dir = os.path.join(data_dir, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = [f for f in image_files if
                   os.path.exists(os.path.join(data_dir, 'labels', os.path.splitext(f)[0] + '.txt'))]

    # 随机打乱并分割
    random.shuffle(image_files)
    split_idx = int(len(image_files) * val_ratio)
    val_files = image_files[:split_idx]
    train_files = image_files[split_idx:]

    # 复制文件到相应目录
    print("正在分割数据集...")
    for file_list, dest_dir in [(train_files, train_dir), (val_files, val_dir)]:
        for img_file in tqdm(file_list):
            base_name = os.path.splitext(img_file)[0]

            # 复制图片
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(dest_dir, 'images', img_file)
            shutil.copy(src_img, dst_img)

            # 复制标签
            src_label = os.path.join(data_dir, 'labels', base_name + '.txt')
            dst_label = os.path.join(dest_dir, 'labels', base_name + '.txt')
            shutil.copy(src_label, dst_label)

    print(f"数据集分割完成:\n训练集: {len(train_files)}张\n验证集: {len(val_files)}张")
    return train_dir, val_dir


def create_yolo_data_files(output_dir, train_dir, val_dir):
    """创建YOLO格式的data文件"""
    # 创建train.txt和val.txt
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for img_file in os.listdir(os.path.join(train_dir, 'images')):
            img_path = os.path.join(train_dir, 'images', img_file)
            f.write(img_path + '\n')

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for img_file in os.listdir(os.path.join(val_dir, 'images')):
            img_path = os.path.join(val_dir, 'images', img_file)
            f.write(img_path + '\n')

    # 创建data.yaml
    classes = []
    with open(os.path.join(train_dir, 'classes.txt'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    data = {
        'train': os.path.join(output_dir, 'train.txt'),
        'val': os.path.join(output_dir, 'val.txt'),
        'nc': len(classes),
        'names': classes
    }

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"train: {data['train']}\n")
        f.write(f"val: {data['val']}\n")
        f.write(f"nc: {data['nc']}\n")
        f.write("names: [")
        f.write(", ".join([f"'{name}'" for name in data['names']]))
        f.write("]\n")

    print("YOLO数据文件创建完成")


def process_yolo_dataset(data_dir, output_dir, val_ratio=0.1):
    """处理YOLO格式数据集的完整流程"""
    # 1. 验证数据集
    print("验证数据集完整性...")
    verify_yolo_dataset(data_dir)

    # 2. 分割数据集
    print("\n分割数据集...")
    train_dir, val_dir = split_yolo_dataset(data_dir, output_dir, val_ratio)

    # 3. 创建YOLO数据文件
    print("\n创建YOLO数据文件...")
    create_yolo_data_files(output_dir, train_dir, val_dir)

    print("\n数据集处理完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='处理YOLO格式数据集')
    parser.add_argument('--data_dir', type=str, required=True, help='原始YOLO格式数据集目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    args = parser.parse_args()

    process_yolo_dataset(args.data_dir, args.output_dir, args.val_ratio)
