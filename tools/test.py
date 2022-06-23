import os
import shutil
import cv2
import tools as t
import pandas as pd
from tqdm import tqdm

"""
save rf maps
"""
# data_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/radar/"
# save_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/rf_maps/"
# t.save_rf_maps(data_path, save_path)

"""
Convert XML files to csv
"""
# anns_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/anns/"
# save_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/all_data2.csv"
# classes = ["car", "cyclist", "pedestrian"]
#
# t.xml_to_csv(anns_path, save_path, classes)

"""
Resize images and annotations
"""
# print("resize images and annotations")
# orig_size = (864, 1440)
# new_size = (448, 448)
# image_folder = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/images/"
# annotation_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/all_data.csv"
# columns = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id']
# dest_image_folder = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/images/"
# dest_ann_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/resized_data_448.csv"
#
# t.resize_data(orig_size, new_size, image_folder, annotation_path, columns, dest_image_folder, dest_ann_path)

"""
Resize rf maps
"""
# print("resize rf maps")
# new_size = (448, 448)
# image_folder = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/rf_maps/"
# dest_folder_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/rf_maps/"
#
# for i in os.listdir(image_folder):
#     image_path = image_folder + i
#     t.resize_image(image_path, new_size, dest_folder_path + i)
#     print("resized: " + image_path)

"""
Draw bounding box with detections
"""
# df = pd.read_csv('C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/resized_data_448_shuffled.csv')
# test_df = pd.read_csv("C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/cruw_test_rgb.csv")
#
# base_path = 'C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/images/'
# dest_path = 'C:/Users/Administrator/Desktop/my_detections/cruw/multi/lab_gts/'
#
# pickle_path = "C:/Users/Administrator/Desktop/multi_preds_500.sav"
#
# gt_color = (255, 255, 255)
# classes = ["bg", "car", "cyclist", "pedestrian"]
# colors = [(), (76, 153, 0), (35, 207, 233), (153, 76, 0)]
#
# show_gts = True
#
# test_images = test_df['image'].unique()
#
# dets = t.load_pickle(pickle_path)
# print(len(dets))
#
# for i, tt in enumerate(test_images[0:20]):
#     gt_boxes = df[df['image'] == tt][['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
#     img = cv2.imread(base_path + tt)
#
#     # Draw detection bboxes
#     for j in range(len(dets[i])):
#         d_xmin = int(dets[i][j][0])
#         d_ymin = int(dets[i][j][1])
#         d_xmax = int(dets[i][j][2])
#         d_ymax = int(dets[i][j][3])
#
#         cl = classes[(int(dets[i][j][4]))]
#         color = colors[int(dets[i][j][4])]
#         conf = round(dets[i][j][5], 2)
#
#         img = t.cv2_draw_box(img, d_xmin, d_ymin, d_xmax, d_ymax, color, line_width=1)
#
#         img = t.cv2_draw_box_with_labels(img_array=img,
#                                          xmin=d_xmin,
#                                          ymin=d_ymin,
#                                          xmax=d_xmax,
#                                          ymax=d_ymax,
#                                          class_=cl,
#                                          conf=conf,
#                                          bb_color=color,
#                                          line_width=1)
#
#     # Draw ground truth bboxes
#     if show_gts:
#         for gt in gt_boxes:
#             gt_xmin = gt[0]
#             gt_ymin = gt[1]
#             gt_xmax = gt[2]
#             gt_ymax = gt[3]
#
#             img = t.cv2_draw_box(img, gt_xmin, gt_ymin, gt_xmax, gt_ymax, gt_color, line_width=1)
#
#     cv2.imwrite(dest_path + tt, img)

"""
Generate rf maps csv path
"""
csv_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/cruw_train_data_rgb.csv"
rf_folder_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/cruw_train_data_rf.csv"
t.rf_maps_to_csv(csv_path, rf_folder_path)

"""
Shuffle data
"""
# csv_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/resized_data_448.csv"
# dest_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/"
#
# t.shuffle_data(csv_path, dest_path, "cruw", test_size=0.2, val_size=0.1, splits=True)

"""
Split data
"""
# csv_path = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/CRUW_kaggle/resized_data_448_shuffled.csv"
# dest_path = '../data/'
# t.split_data(csv_path, image_column='image',
#              test_size=0.2,
#              val_size=0.1,
#              dest_path=dest_path,
#              shuffle=True)
