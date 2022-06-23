import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import pickle


# rename files in image data folder
def rename_folder_files(image_folders_path):
    folders = os.listdir(image_folders_path)
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v',
              'w', 'x', 'y', 'z']

    for i, fd in enumerate(folders):
        inner_folders = os.listdir(image_folders_path + fd)
        for ifd in inner_folders:
            if ifd == 'anns':
                new_ann_folder_path = image_folders_path + fd + "/" + "anns2"

                if not os.path.exists(new_ann_folder_path):
                    os.mkdir(new_ann_folder_path)

                annotations = os.listdir(image_folders_path + fd + '/' + ifd)
                for j, ann in enumerate(annotations):
                    source_ann_path = image_folders_path + fd + '/' + ifd + '/' + ann
                    shutil.copy(source_ann_path, new_ann_folder_path)
                    os.rename(new_ann_folder_path + "/" + ann, new_ann_folder_path + "/" + labels[i] + ann)

            if ifd == "IMAGES_0":
                new_img_folder_path = image_folders_path + fd + "/" + "imgs2"

                if not os.path.exists(new_img_folder_path):
                    os.mkdir(new_img_folder_path)

                imgs = os.listdir(image_folders_path + fd + '/' + ifd)
                for j, img in enumerate(imgs[0:len(annotations)]):
                    source_ann_path = image_folders_path + fd + '/' + ifd + '/' + img
                    shutil.copy(source_ann_path, new_img_folder_path)
                    os.rename(new_img_folder_path + "/" + img, new_img_folder_path + "/" + labels[i] + img)


# Generate radar chirp magnitude
def magnitude(chirp, radar_data_type):
    """
    Calculate magnitude of a chirp
    :param chirp: radar data of one chirp (w x h x 2) or (2 x w x h)
    :param radar_data_type: current available types include 'RI', 'RISEP', 'AP', 'APSEP'
    :return: magnitude map for the input chirp (w x h)
    """
    c0, c1, c2 = chirp.shape
    if radar_data_type == 'RI' or radar_data_type == 'RISEP':
        if c0 == 2:
            chirp_abs = np.sqrt(chirp[0, :, :] ** 2 + chirp[1, :, :] ** 2)
        elif c2 == 2:
            chirp_abs = np.sqrt(chirp[:, :, 0] ** 2 + chirp[:, :, 1] ** 2)
        else:
            raise ValueError
    elif radar_data_type == 'AP' or radar_data_type == 'APSEP':
        if c0 == 2:
            chirp_abs = chirp[0, :, :]
        elif c2 == 2:
            chirp_abs = chirp[:, :, 0]
        else:
            raise ValueError
    else:
        raise ValueError
    return chirp_abs


# Generate RF map
def draw_rf_image(chirp_data, radar_data_type, save_path=None, normalized=True):
    """
    Visualize radar data of one chirp
    :param chirp_data: (w x h x 2) or (2 x w x h)
    :param radar_data_type: current available types include 'RI', 'RISEP', 'AP', 'APSEP'
    :param save_path: save figure path
    :param normalized: is radar data normalized or not
    :return:
    """

    chirp_abs = magnitude(chirp_data, radar_data_type)

    if normalized:
        plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
    else:
        plt.imshow(chirp_abs, origin='lower')
        plt.colorbar()

    if save_path is None:
        plt.show()
    else:
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


# Filter radar_data in folders
def filter_radar_data(radar_folders_path):
    radar_folder = os.listdir(radar_folders_path)
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

    for i, r in enumerate(radar_folder):
        filtered_path = radar_folders_path + r + "/" + "filtered_" + r
        rad_files = os.listdir(radar_folders_path + r + "/RADAR_RA_H")

        if not os.path.exists(filtered_path):
            os.mkdir(filtered_path)

        for rad in rad_files:
            ext = rad.split("_")
            if ext[1] == "0000.npy":
                shutil.copy(radar_folders_path + r + "/RADAR_RA_H/" + rad, filtered_path)
                os.rename(filtered_path + "/" + rad, filtered_path + "/" + labels[i] + rad)


def save_rf_maps(data_path, save_path):
    files = os.listdir(data_path)
    for f in files:
        rad_file_path = data_path + f
        chirp = np.load(rad_file_path)
        draw_rf_image(chirp, "RI", save_path + f + ".png")


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    Args:
        bbox (~numpy.ndarray): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    .. csv-table::
        xmin,ymin,xmax,ymax
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """

    # 0-ymin, 1-xmin, 2-ymax, 3-xmax
    # 1-ymin, 0-xmin, 3-ymax, 2-xmax - ours

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]

    bbox[1] = y_scale * bbox[1]
    bbox[3] = y_scale * bbox[3]

    bbox[0] = x_scale * bbox[0]
    bbox[2] = x_scale * bbox[2]
    return bbox


def resize_image(image_path, size, dest_path):
    image_array = cv2.imread(image_path)
    img_array = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(dest_path, img_array)


def resize_data(orig_size, new_size, image_folder, annotation_path, columns, dest_image_folder,
                dest_ann_path, ):  # columns format(image file, xmin,ymin,xmax,ymax)
    new_bboxes = []
    df = pd.read_csv(annotation_path)
    images = df[columns[0]].tolist()
    unique_images = df[columns[0]].unique().tolist()
    bbox = df[columns[1:5]].to_numpy()
    classes = df[columns[5]].tolist()

    print("resizing images")
    print(len(unique_images))
    for i in range(len(unique_images)):
        image_path = image_folder + unique_images[i]
        resize_image(image_path, new_size, dest_image_folder + unique_images[i])
        print("resized: " + image_path)

    print("resizing boxes")
    for j in range(len(images)):
        h, w = orig_size
        new_bbox = resize_bbox(bbox[j], (h, w), new_size)
        new_bboxes.append([images[j], new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], classes[j]])

    new_df = pd.DataFrame(new_bboxes, columns=['image', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    new_df.to_csv(dest_ann_path, index=False)


def xml_to_csv(path, dest_path, classes, image_type=".jpg"):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        name_ = xml_file.split('\\')
        file_ = name_[1].split(".")
        file_name_ = file_[0] + image_type

        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text
            class_id = label  # classes.index(label) + 1

            value = (
                # root.find('filename').text,
                # int(root.find('size')[0].text),
                # int(root.find('size')[1].text),
                file_name_,
                xmin,
                ymin,
                xmax,
                ymax,
                class_id
            )
            xml_list.append(value)
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(dest_path, index=False)


def rf_maps_to_csv(image_csv_path, dest_path):
    df = pd.read_csv(image_csv_path)
    images = df['image'].tolist()
    names = []

    for i in tqdm(images):
        x = i.split(".")

        base = x[0]
        first_char = i[0]
        last_six = base[-6:]

        rf_name = first_char + last_six + "_0000.npy.png"
        names.append(rf_name)

    bb = df[['xmin', 'ymin', 'xmax', 'ymax', 'class']]
    bb.insert(0, column='image', value=names)
    bb.to_csv(dest_path, index=False)


def cv2_draw_box(img_array, xmin, ymin, xmax, ymax, color, line_width):
    img = cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, line_width)
    return img


def cv2_draw_box_with_labels(img_array, xmin, ymin, xmax, ymax, class_, conf, bb_color, line_width):
    label = class_ + " " + str(conf)
    img = cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), bb_color, line_width)

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.6, 1)

    img = cv2.rectangle(img, (xmin, ymin - 15), (xmin + w, ymin), bb_color, -1)

    img = cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)

    return img


# def shuffle_data(csv_path, dest_path, dataset_name, test_size=None, val_size=None, splits=False):
#     df = pd.read_csv(csv_path)
#     df = df.sample(frac=1)
#
#     if splits:
#         if test_size is None or val_size is None:
#             print("Please specify a test and validation size.")
#         else:
#             test_ = int(len(df) * 0.2)
#             val_ = int(len(df) * 0.1)
#
#             train = df[test_ + val_:]
#             valid = df[test_: test_ + val_]
#             test = df[0: test_]
#
#             print("Train shape: ", train.shape)
#             print("Train counts:", train.value_counts('class'))
#             print('-------------------------------')
#             print("Valid shape: ", valid.shape)
#             print("Valid counts:", valid.value_counts('class'))
#             print('-------------------------------')
#             print("Test shape: ", test.shape)
#             print("Test counts:", test.value_counts('class'))
#
#             print(dest_path + dataset_name + "_train.csv")
#
#             train.to_csv(dest_path + dataset_name + "_train.csv", index=False)
#             test.to_csv(dest_path + dataset_name + "_test.csv", index=False)
#             valid.to_csv(dest_path + dataset_name + "_valid.csv", index=False)
#     else:
#         df.to_csv(dest_path, index=False)

def split_data(csv_path, image_column, test_size, val_size, dest_path, shuffle):
    test_data_list = []
    val_data_list = []
    train_data_list = []

    df = pd.read_csv(csv_path)
    if shuffle:
        df = df.sample(frac=1)

    print("Data class counts")
    print(df.value_counts('class'))
    print("")

    images = df[image_column].unique()

    test_ = int(len(images) * test_size)
    val_ = int(len(images) * val_size)

    test_images = images[0:test_]
    val_images = images[test_:test_ + val_]
    train_images = images[test_ + val_:]

    print("Test images:", len(test_images))
    print("Val images:", len(val_images))
    print("Train images:", len(train_images))
    print("Total images:", test_ + val_ + len(train_images))

    # get all test, val and train annotations
    for tt in test_images:
        test_data = df[df[image_column] == tt]  # [['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        test_data_list.append(test_data)

    for vl in val_images:
        val_data = df[df[image_column] == vl]
        val_data_list.append(val_data)

    for tr in train_images:
        train_data = df[df[image_column] == tr]
        train_data_list.append(train_data)

    #     Concatenate all annotations
    test_df = pd.concat(test_data_list)
    test_df = test_df.sample(frac=1)

    val_df = pd.concat(val_data_list)
    val_df = val_df.sample(frac=1)

    train_df = pd.concat(train_data_list)
    train_df = train_df.sample(frac=1)

    print("")
    print("Test Data class counts")
    print(test_df.value_counts('class'))
    print("")
    print("Val Data class counts")
    print(val_df.value_counts('class'))
    print("")
    print("Train Data class counts")
    print(train_df.value_counts('class'))

    test_df.to_csv(dest_path + "cruw_test_data.csv", index=False)
    val_df.to_csv(dest_path + "cruw_val_data.csv", index=False)
    train_df.to_csv(dest_path + "cruw_train_data.csv", index=False)


def load_pickle(file_path):
    file_ = open(file_path, 'rb')
    data = pickle.load(file_)
    file_.close()

    return data
