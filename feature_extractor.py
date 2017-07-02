import cv2
import imageio
import numpy as np
from keras.models import Model

from inception_v3 import InceptionV3, preprocess_input
from util import mkdir_if_not_exist


def get_img_list_from_vid_reader(vid, extension, reshape_size=(299, 299), MAX_N_FRAME=300):
    fps = int(round(vid.get_meta_data()['fps']))
    img_list = []

    for i in range(MAX_N_FRAME):
        num = i * fps
        try:
            img = vid.get_data(num)
        except:
            return np.array(img_list)

        if extension == '3gp':
            img = img[25:120, :]  # cropping

        img = cv2.resize(img, reshape_size).astype(np.float32)
        img_list.append(img)

    return np.array(img_list)


def get_feature_mat(mymodel, input_data, batch_size=128):
    idx = 0
    preds_mat = None
    while (idx < len(input_data)):
        x = input_data[idx:idx + batch_size]
        x = preprocess_input(x)
        preds = mymodel.predict(x)

        if preds_mat is None:
            preds_mat = preds
        else:
            preds_mat = np.concatenate([preds_mat, preds])
        idx += batch_size

    return preds_mat.astype(np.float32)


def get_feature_mat_from_video(video_filename, output_dir='output'):
    yt_vid, extension = video_filename.split('/')[-1].split('.')

    assert extension in ['webm', 'mp4', '3gp']

    mkdir_if_not_exist(output_dir, False)

    output_filename = output_dir + '/' + yt_vid + '.npy'

    vid_reader = imageio.get_reader(video_filename, 'ffmpeg')

    img_list = get_img_list_from_vid_reader(vid_reader, extension)

    base_model = InceptionV3(include_top=True, weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    feature_mat = get_feature_mat(model, img_list)

    np.save(output_filename, feature_mat)

    return feature_mat
