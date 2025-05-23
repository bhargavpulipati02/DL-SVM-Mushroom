import pandas as pd
import argparse
import tensorflow as tf
import numpy as np
import os
# -----------run using :python proj3_classification_test.py --model trained_cnn_model.keras  --test_csv sample_test_data/mushrooms_test.csv -------
def load_model_weights(model):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model

def get_images_labels(df, classes, img_height=224, img_width=224):
    class_to_idx = {name: idx for idx, name in enumerate(sorted(classes))}
    image_paths = df['image_path'].tolist()
    labels = df['label'].map(class_to_idx).tolist()

    images = []
    for path in image_paths:
        full_path = os.path.join('sample_test_data', path)
        img_raw = tf.io.read_file(full_path)
        img_processed = decode_img(img_raw, img_height, img_width)
        images.append(img_processed)

    test_images = tf.stack(images)
    test_labels = np.array(labels)
    return test_images, test_labels

def decode_img(img_raw, img_height, img_width):
    img = tf.io.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Classification Test")
    parser.add_argument('--model', type=str, default='trained_cnn_model.keras', help='Saved CNN model')
    parser.add_argument('--test_csv', type=str, default='sample_test_data/mushrooms_test.csv', help='CSV file with image_path and label')

    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    classes = {'Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma',
               'Hygrocybe', 'Lactarius', 'Russula', 'Suillus'}

    test_images, test_labels = get_images_labels(test_df, classes)

    my_model = load_model_weights(args.model)
    loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print('CNN softmax model accuracy: Test model, accuracy: {:5.5f}%'.format(100 * acc))
