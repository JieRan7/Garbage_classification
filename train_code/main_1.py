import argparse
from utilis_yolo import process_yolo_dataset

def main(opt):
    """
    主处理函数，用于处理YOLO格式数据集

    参数:
    - opt: 包含命令行参数的对象
      - train_data: 原始YOLO格式数据集目录
      - output_dir: 处理后的输出目录
      - val_ratio: 验证集比例(默认0.2)
    """
    print("开始处理YOLO格式数据集...")

    # 处理数据集
    process_yolo_dataset(
        data_dir=opt.train_data,
        output_dir=opt.output_dir,
        val_ratio=opt.val_ratio
    )

    print("数据集处理完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='D:\project_1\datarubbish',
                        help='原始YOLO格式数据集目录，包含images/labels/classes.txt')
    parser.add_argument('--output_dir', type=str, default='D:\project_1\datarubbish_train',
                        help='处理后的输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例(0-1之间)')
    opt = parser.parse_args()

    main(opt)