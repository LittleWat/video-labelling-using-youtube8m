# -*- coding: utf-8 -*-
import argparse
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf

from feature_extractor import get_feature_mat_from_video
from my_pca import get_pcaed_feature
from yt8m_utils import Dequantize


def print_predicted_label(feature, topn=10, latest_checkpoint='./yt8m_model/model.ckpt-2833',
                          id2label_csv='./label_names.csv'):
    id2label_ser = pd.read_csv(id2label_csv, index_col=0)
    id2label = id2label_ser.to_dict()['label_name']

    meta_graph_location = latest_checkpoint + ".meta"

    sess = tf.InteractiveSession()

    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    saver.restore(sess, latest_checkpoint)

    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
        init_op_list = []
        for variable in list(variables):
            if "train_input" in variable.name:
                init_op_list.append(tf.assign(variable, 1))
                variables.remove(variable)
        init_op_list.append(tf.variables_initializer(variables))
        return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
            tf.GraphKeys.LOCAL_VARIABLES)))

    padded_feature = np.zeros([300, 1024])
    padded_feature[:feature.shape[0], :] = Dequantize(feature)
    video_batch_val = padded_feature[np.newaxis, :, :].astype(np.float32)
    num_frames_batch_val = np.array([feature.shape[0]], dtype=np.int32)

    predictions_val, = sess.run([predictions_tensor], feed_dict={input_tensor: video_batch_val,
                                                                 num_frames_tensor: num_frames_batch_val})

    predictions_val = predictions_val.flatten()

    top_idxes = np.argsort(predictions_val)[::-1][:topn]

    pprint.pprint([(id2label[x], predictions_val[x]) for x in top_idxes])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video(< 3 min) label prediction')
    parser.add_argument('video_filename')

    args = parser.parse_args()

    feature_2048_mat = get_feature_mat_from_video(args.video_filename)
    print ("Got separated image features using keras model")

    pca_feature = get_pcaed_feature(feature_2048_mat)
    print ("Got PCA feature")

    print_predicted_label(pca_feature)
