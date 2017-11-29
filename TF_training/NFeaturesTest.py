#https://www.tensorflow.org/tutorials/wide

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf
import numpy as np

def run_linear_classifier (N , titles, dir):
    CSV_COLUMNS = [
        titles [1:len(titles)]
    ]

    # Continuous base columns.
    group = tf.feature_column.categorical_column_with_vocabulary_list("group",['0','1'])
    label_column = [group]
    base_columns = []
    for title in titles:
        base_columns.append(tf.feature_column.numeric_column(title))



def build_estimator(model_dir, base_columns):
    """Build an estimator."""
    m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns)
    return m


def input_fn(data_file, num_epochs, shuffle):
    """Input builder function."""
    df_data = pd.read_csv(
        tf.gfile.Open(data_file),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    if (data_file == "training.csv"):
        print ("meow\n")
        labels = pd.read_csv(
            tf.gfile.Open("training_labels.csv"),
            names=label_column,
            skipinitialspace=True,
            engine="python",
            skiprows=1)
        labels = labels.dropna(how="any", axis=0)

    else:
          print ("wuf\n")
          labels = pd.read_csv(
              tf.gfile.Open("test_labels.csv"),
              names=label_column,
              skipinitialspace=True,
              engine="python",
              skiprows=1)
          labels = labels.dropna(how="any", axis=0)
    return tf.estimator.inputs.pandas_input_fn(
          x=df_data,
          y=labels,
          batch_size=100,
          num_epochs=num_epochs,
          shuffle=shuffle,
          num_threads=5)


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
      """Train and evaluate the model."""
      train_file_name, test_file_name = maybe_download("training.csv", "test.csv")
      # Specify file path below if want to find the output easily
      model_dir = tempfile.mkdtemp() if not model_dir else model_dir

      m = build_estimator(model_dir, model_type)
      # set num_epochs to None to get infinite stream of data.
      m.train(
          input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
          steps=train_steps)
      # set steps to None to run evaluation until all data consumed.
      results = m.evaluate(
          input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
          steps=None)
      print("model directory = %s" % model_dir)
      for key in sorted(results):
        print("%s: %s" % (key, results[key]))
      # Manual cleanup
      shutil.rmtree(model_dir)


    FLAGS = None


    def main(_):
      train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                     FLAGS.train_data, FLAGS.test_data)


    if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.register("type", "bool", lambda v: v.lower() == "true")
      parser.add_argument(
          "--model_dir",
          type=str,
          default="",
          help="Base directory for output models."
      )
      parser.add_argument(
          "--model_type",
          type=str,
          default="wide_n_deep",
          help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
      )
      parser.add_argument(
          "--train_steps",
          type=int,
          default=1000,
          help="Number of training steps."
      )
      parser.add_argument(
          "--train_data",
          type=str,
          default="",
          help="Path to the training data."
      )
      parser.add_argument(
          "--test_data",
          type=str,
          default="",
          help="Path to the test data."
      )
      FLAGS, unparsed = parser.parse_known_args()
      tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)