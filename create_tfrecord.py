import tensorflow as tf
import numpy as np
# import PIL.Image as Image
import io
import os
import copy
import shutil
import random
import math
import argparse
from tqdm import tqdm
import sys
import cv2
import os
import json
from create_dates import create_maps, label2num

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

'''

Example of use:

python create_tfrecord.py --output=./tfrecords/ --num_shards=2

'''

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_example(image, label, name):

    encoded_jpg = image.tobytes()

    height = image.shape[0]
    width = image.shape[1]

    bboxes = label[name]["boxes"]
    labels = label[name]["labels"]

    clss = [label2num[i] for i in labels]
    bboxes = np.stack(bboxes, axis=0)
    ymin, xmin, ymax, xmax = bboxes[:,1], bboxes[:,0], bboxes[:,3], bboxes[:,2]

    if len(clss) == 0:
        return None

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(encoded_jpg),
        'height': int64_feature(int(height)),
        'width': int64_feature(int(width)),
        'num': int64_feature(len(clss)),
        'ymin': float_list_feature(ymin),
        'xmin': float_list_feature(xmin),
        'ymax': float_list_feature(ymax),
        'xmax': float_list_feature(xmax),

        'class': int64_list_feature(clss),

    }))

    return example


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-n', '--num_shards', type=int, default=1)
    return parser.parse_args()


def main(num_examples=100):
    ARGS = make_args()

    num_shards = ARGS.num_shards
    shard_size = math.ceil(num_examples / num_shards)
    print('Number of images per shard:', shard_size)

    output_dir = ARGS.output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    shard_id = 0
    num_examples_written = 0
    for _ in tqdm(range(num_examples)):
        if num_examples_written == 0:
            shard_path = os.path.join(output_dir, f'maps-train-iou-0.25-5-clz-{shard_id:04}.tfrecords')
            writer = tf.io.TFRecordWriter(shard_path)

        output_name = f"{_:08}.jpg"
        image, label = create_maps(output_name, _)
        tf_example = dict_example(image, label, output_name)
        if tf_example is None:
            continue

        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != shard_size and num_examples % num_shards != 0:
        writer.close()

    print('Result is here:', ARGS.output)


if __name__ == "__main__":

    main(1000)