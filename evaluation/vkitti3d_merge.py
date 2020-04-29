#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import plyfile
import numpy as np
import argparse
import h5py

reduced_length_dict = {"MarketplaceFeldkirch": [10538633, "marketsquarefeldkirch4-reduced"],
                       "0006_00521": [14608690, "0006_00521"],
                       "0006_00542": [28931322, "sg27_10-reduced"],
                       "0006_00564": [24620684, "sg28_2-reduced"]}

full_length_dict = {"0005_00080": [333206, "0005_00080"],
                    "0005_00106": [333293, "0005_00106"],
                    "0005_00132": [334383, "0005_00132"],
                    "0005_00158": [328194, "0005_00158"],
                    "0005_00184": [338761, "0005_00184"],
                    "0005_00210": [345188, "0005_00210"],
                    "0005_00236": [347744, "0005_00236"],
                    "0005_00262": [345501, "0005_00262"],
                    "0005_00288": [357484, "0005_00288"],
                    "0005_00314": [419277, "0005_00314"],
                    "0005_00340": [347610, "0005_00340"],
                    "0005_00366": [348642, "0005_00366"],
                    "0005_00392": [350265, "0005_00392"],
                    "0005_00418": [359940, "0005_00418"],
                    "0005_00444": [366115, "0005_00444"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d',
                        help='Path to input *_pred.h5', required=True)
    parser.add_argument('--version', '-v',
                        help='full or reduced', type=str, required=True)
    args = parser.parse_args()
    print(args)

    if args.version == 'full':
        length_dict = full_length_dict
    else:
        length_dict = reduced_length_dict

    categories_list = [category for category in length_dict]
    print(categories_list)

    for category in categories_list:
        output_path = os.path.join(
            args.datafolder, "results", length_dict[category][1]+".labels")
        if not os.path.exists(os.path.join(args.datafolder, "results")):
            os.makedirs(os.path.join(args.datafolder, "results"))
        pred_list = [pred for pred in os.listdir(args.datafolder)
                     if category in pred and pred.split(".")[0].split("_")[-1] == 'pred']

        label_length = length_dict[category][0]
        merged_label = np.zeros((label_length), dtype=int)
        merged_confidence = np.zeros((label_length), dtype=float)

        for pred_file in pred_list:
            print(os.path.join(args.datafolder, pred_file))
            data = h5py.File(os.path.join(args.datafolder, pred_file))
            labels_seg = data['label_seg'][...].astype(np.int64)
            indices = data['indices_split_to_full'][...].astype(np.int64)
            confidence = data['confidence'][...].astype(np.float32)
            data_num = data['data_num'][...].astype(np.int64)

            for i in range(labels_seg.shape[0]):
                temp_label = np.zeros((data_num[i]), dtype=int)
                pred_confidence = confidence[i][:data_num[i]]
                temp_confidence = merged_confidence[indices[i][:data_num[i]]]

                temp_label[temp_confidence >= pred_confidence] = merged_label[indices[i]
                                                                              [:data_num[i]]][temp_confidence >= pred_confidence]
                temp_label[pred_confidence > temp_confidence] = labels_seg[i][:data_num[i]
                                                                              ][pred_confidence > temp_confidence]

                merged_confidence[indices[i][:data_num[i]][pred_confidence >
                                                           temp_confidence]] = pred_confidence[pred_confidence > temp_confidence]
                merged_label[indices[i][:data_num[i]]] = temp_label

        np.savetxt(output_path, merged_label+1, fmt='%d')


if __name__ == '__main__':
    main()
