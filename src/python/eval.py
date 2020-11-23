#!/usr/bin/env python3

#######################################################################################################################
# eval.py - python script for running eval on a breath rate detection algorithm
# 
# usage: ./eval.py <data-dir>
#######################################################################################################################
import argparse
import os
import random
from sklearn.metrics import mean_squared_error


class RandomBreathRatePredictor(object):
    def predict_rate(self, vid_path):
        return (60.0 * random.uniform(0, 1))

class StaticBreathRatePredictor(object):
    def predict_rate(self, vid_path):
        return 20.0

class Eval(object):
    def __init__(self, data_dir, label_file, predictor=RandomBreathRatePredictor()):
        self._data_dir = data_dir
        self._label_file = os.path.join(data_dir, label_file)
        self._predictor = predictor

    def evaluate_rmse(self):
        lines = []
        with open(self._label_file) as f:
            lines = list(f)

        labels = [float(x.split(',')[1]) for x in lines]
        vids = [os.path.join(self._data_dir, x.split(',')[0]) for x in lines]

        vids_predict = [self._predictor.predict_rate(v) for v in vids]

        rmse = mean_squared_error(labels, vids_predict, squared=False)

        return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an algorithm for detecting breath rates.')
    parser.add_argument('data_dir', metavar='DATA-DIR', type=str, help='directory containing videos and annotation file')
    parser.add_argument('label_file', metavar='LABEL-FILE', type=str, help='path to a csv file containing <video-file>,<breath-rate> entries for each file in the test set. assumed to be in the DATA-DIR.')
    args = parser.parse_args()

    print('Using data dir: {} and labels from {}'.format(args.data_dir, args.label_file))
    
    ev = Eval(args.data_dir, args.label_file)
    rmse = ev.evaluate_rmse()

    print('Default predictor, RMSE = {}'.format(rmse))

    ev = Eval(args.data_dir, args.label_file, predictor=StaticBreathRatePredictor())
    rmse = ev.evaluate_rmse()

    print('Static predictor, RMSE = {}'.format(rmse))