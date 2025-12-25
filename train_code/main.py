import argparse
import xml.etree.ElementTree as ET
from utilis_yolo import *
import argparse

label_list = ['mask']

def get_image_txt(opt):

    #　阶段一：对于数据集进行清洗梳理　
    # 第一步：根据images_label_split中的图像删除多余的xml
    print("V1")
    compare_image_label_remove_xml(opt.train_data)
    # # # 第二步：根据images_label_split中的图像删除多余的image
    print("V2")
    compare_image_label_remove_image(opt.train_data)
    # # 第三步：将各个文件夹中的xml不满足条件的文件删除
    print("V3")
    remove_not_satisfied_xml(opt.train_data)
    # # 第四步：查找xml是否为空，空的话删除xml,也删除对应的image
    print("V4")
    remove_image_null_xml(opt.train_data,label_list)
    # # 第五步：对照image和xml中数据，显示图片看画得框是否正确
    # show_label(opt.train_data,label_list)

    #　阶段二：将数据按照一定比例分成训练和验证集　
    # 将train和test随机分开，将image和xml分别保存到train和test所在的文件夹中
    # 根据前面可以得到xml和image,每个场景下选择10%的数据,作为验证集, 生成train和test两个文件夹

    # 在调用 yolov3_get_train_test_file 前添加：
    print("\n=== 准备数据分割 ===")
    data_dir = os.path.join(opt.train_data, "images_label_split", "classroom")
    print(f"数据目录: {data_dir}")
    print(f"Annotations 文件数: {len(os.listdir(os.path.join(data_dir, 'Annotations')))}")
    print(f"JPEGImages 文件数: {len(os.listdir(os.path.join(data_dir, 'JPEGImages')))}")

    # 调用分割函数
    yolov3_get_train_test_file(opt.train_data, 0.2)

    # 检查结果
    print("\n=== 分割结果 ===")
    train_dir = os.path.join(opt.train_data, "train")
    test_dir = os.path.join(opt.train_data, "test")
    print(f"train 文件夹是否存在: {os.path.exists(train_dir)}")
    print(f"test 文件夹是否存在: {os.path.exists(test_dir)}")
    if os.path.exists(train_dir):
        print(f"train/Annotations 文件数: {len(os.listdir(os.path.join(train_dir, 'Annotations')))}")
    if os.path.exists(test_dir):
        print(f"test/Annotations 文件数: {len(os.listdir(os.path.join(test_dir, 'Annotations')))}")

    # yolov3_get_train_test_file(opt.train_data,0.2)

    # 阶段三：将train和test的xml，转换成txt
    # 第一步：将train和test中的xml文件生成txt文件，都放到image_txt文件夹中
    yolov3_get_txt(opt.train_data,label_list)
    # #  第二步：将所有的image文件一起移动到image_txt中
    yolov3_move_image(opt.train_data)
    # # 第三步：将train/Annotations和test/Annotations的xml自动生成train.txt和test.txt文件，并保存到train_test_txt中
    yolov3_get_train_test_txt(opt.train_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='D:\download\mask_train_data', help='data dir')
    opt = parser.parse_args()
    get_image_txt(opt)