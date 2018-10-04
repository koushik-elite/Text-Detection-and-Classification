import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import numpy as np
import os
import pandas as pd
import re
import csv
import logging

LOG_FILENAME = 'C:\\author_detection\\tensor.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
logging.getLogger('tensorflow')

from sklearn.preprocessing import MultiLabelBinarizer

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["author"] = []
  # data["filename"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(str(f.read()))
      data["author"].append(int(re.match("\d+_(\d+)\.txt", file_path).group(1)))
      # data["filename"].append(str(file_path))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(directory)
  return pos_df.sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  train_df = load_dataset(os.path.join(os.path.dirname("C:/author_detection/"), "stories"))
  test_df = load_dataset(os.path.join(os.path.dirname("C:/author_detection/"), "test_stories"))  
  # test_df = []
  return train_df, test_df

train_writer = tf.summary.FileWriter('C:\\author_detection\\train_dir')

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.INFO)

train_df, test_df = download_and_load_datasets()
# train_df.head()
# test_df.head()

input_x = train_df.drop('author',axis=1)
input_y = train_df['author']

test_input_x = test_df.drop('author',axis=1)
# print(test_input_x)


# print(train_df['filename'])
# label = pd.DataFrame(np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], ...]))
# label.head()

# exit();
# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(input_x, input_y, batch_size=10, num_epochs=1000, shuffle=True)
# train_input_fn = tf.estimator.input.numpy_input_fn(data, labels, shuffle=True)
# exit()

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(input_x, input_y, shuffle=False)

# Prediction on the test set.
# predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, test_df["author"], shuffle=False)

raw_test = [
    "Catesby, if I had ten such captains as Sir Richard, I would march forthright on London. But now, sir, claim your reward, ", # 8
    "Your statement is most interesting, said Sherlock Holmes. Has anything else occurred to you? Yes", # 1-2
    "but not quite so unaffectedly happy as she had been some days earlier. The prince redoubled his attentive study of her symptoms., It was a most curious circumstance, in his opinion, that she never spoke of Rogojin", # 0
]

# predict_input_fn = tf.estimator.inputs.pandas_input_fn(pd.DataFrame.from_dict({"sentence": np.array(raw_test).astype(np.str)}), shuffle=False)

predict_input_fn = tf.estimator.inputs.pandas_input_fn(test_input_x, shuffle=False)

def train_and_evaluate_with_module(hub_module, train_module=False):      
  
  embedded_text_feature_column = hub.text_embedding_column(key="sentence", module_spec=hub_module, trainable=train_module)

  print(hub_module)

  estimator = tf.estimator.DNNClassifier(
      hidden_units=[64,10],
      model_dir='models',
      feature_columns=[embedded_text_feature_column],
      n_classes=10,
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

  print('--------- Train your Estimators')
  estimator.train(input_fn=train_input_fn, max_steps=1000)

  # exit()

  print('--------- Test our model on some raw description data')
  train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
  # test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
  
  print('--------- Generate predictions')
  results = estimator.predict(predict_input_fn)

  print('Display predictions')
  count=0
  with open("C:\\author_detection\\result.csv", "w", newline='') as myfile:
    writer = csv.DictWriter(myfile, delimiter=',', lineterminator='\n', fieldnames=['author'], restval=0)
    writer.writeheader()
    print('--------- counting')
    for movie_genres in results:
      wrow = {}
      count = count + 1
      print('--------- counting {0}'.format(count))
      top_2 = movie_genres['probabilities'].argsort()[-1:][::-1]
      # wrow['author'] = ','.join(map(str, top_2))
      wrow['author'] = ','.join(map(str, top_2))
      writer.writerow(wrow)

  training_set_accuracy = train_eval_result["accuracy"]
  # test_set_accuracy = test_eval_result["accuracy"]

  return {
      "Training accuracy": training_set_accuracy,
      # "Test accuracy": test_set_accuracy
  }

# print(train_df)

results = {}

# print('-------------------- nnlm-en-dim128')
# results["nnlm-en-dim128"] = train_and_evaluate_with_module("https://tfhub.dev/google/nnlm-en-dim128/1")
# print('--------------------- nnlm-en-dim128-with-module-training')

# tf.reset_default_graph()
# ops.reset_default_graph()

# print('Wiki-words-250-with-normalization')
# results["Wiki-words-250-with-normalization"] = train_and_evaluate_with_module("https://tfhub.dev/google/Wiki-words-250-with-normalization/1", False)

# print('universal-sentence-encoder')
# results["universal-sentence-encoder"] = train_and_evaluate_with_module("C:\\author_detection\\universal-sentence-encoder_2", True)

print('Wiki-words-250-with-normalization')
# results["universal-sentence-encoder"] = train_and_evaluate_with_module("C:\\author_detection\\Wiki-words-250-with-normalization_1", True)
results["universal-sentence-encoder"] = train_and_evaluate_with_module("https://tfhub.dev/google/Wiki-words-250-with-normalization/1", True)


# results["random-nnlm-en-dim128"] = train_and_evaluate_with_module("https://tfhub.dev/google/random-nnlm-en-dim128/1")
# print('random-nnlm-en-dim128-with-module-training')
# results["random-nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module("https://tfhub.dev/google/random-nnlm-en-dim128/1", True)

print(pd.DataFrame.from_dict(results, orient="index"))