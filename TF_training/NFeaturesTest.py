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
import os

FLAGS = None


def main(argv=None):
    print (FLAGS)
    train_and_eval(FLAGS.model_dir, FLAGS.train_steps, FLAGS.csv_files_dir,
        FLAGS.CSV_COLUMNS,FLAGS.label_column,FLAGS.base_columns)

def run_linear_classifier (titles, dir):
    CSV_COLUMNS = [
        *titles [1:len(titles)]
    ]

    # Continuous base columns.
    group = tf.feature_column.categorical_column_with_vocabulary_list(titles[0],['0','1'])
    label_column = [group]
    base_columns = []
    for title in titles:
        base_columns.append(tf.feature_column.numeric_column(title))


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
    parser.add_argument(
        "--CSV_COLUMNS",
        type=list,
        default=CSV_COLUMNS,
        help="nope."
    )

    parser.add_argument(
        "--label_column",
        type=list,
        default=label_column,
        help="nope."
    )
    parser.add_argument(
        "--base_columns",
        type=list,
        default=base_columns,
        help="nope."
    )

    parser.add_argument(
        "--csv_files_dir",
        type=str,
        default=dir,
        help="nope."
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    train_and_eval(FLAGS.model_dir, FLAGS.train_steps, FLAGS.csv_files_dir,
                   FLAGS.CSV_COLUMNS, FLAGS.label_column, FLAGS.base_columns)
    #tf.app.run(main=main, argv=[FLAGS] + unparsed)

def build_estimator(model_dir, base_columns):
    """Build an estimator."""
    m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns)
    return m


def input_fn(data_file,CSV_COLUMNS,label_column,csv_files_dir, num_epochs, shuffle ):
    """Input builder function."""
    df_data = pd.read_csv(
        tf.gfile.Open(data_file),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    if (data_file == os.path.join(csv_files_dir,"training_data.csv")):
        print ("meow\n")
        labels = pd.read_csv(
            tf.gfile.Open(os.path.join(csv_files_dir,"training_labels.csv")),
            names=label_column,
            skipinitialspace=True,
            engine="python",
            skiprows=1)
        print (labels)


    else:
          print ("wuf\n")
          labels = pd.read_csv(
              tf.gfile.Open(os.path.join(csv_files_dir,"test_labels.csv")),
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


def train_and_eval(model_dir, train_steps, csv_files_dir, CSV_COLUMNS,label_column,base_columns):
    train_file_name, test_file_name = os.path.join(csv_files_dir,"training_data.csv"), os.path.join(csv_files_dir,"test_data.csv")
      # Specify file path below if want to find the output easily
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir, base_columns)
      # set num_epochs to None to get infinite stream of data.
    m.train(
        input_fn=input_fn(train_file_name, CSV_COLUMNS,label_column,csv_files_dir, num_epochs=None, shuffle=True),
        steps=train_steps)
    # set steps to None to run evaluation until all data consumed.
    results = m.evaluate(
        input_fn=input_fn(test_file_name, CSV_COLUMNS,label_column,csv_files_dir, num_epochs=1, shuffle=False),
        steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
      # Manual cleanup
    shutil.rmtree(model_dir)






