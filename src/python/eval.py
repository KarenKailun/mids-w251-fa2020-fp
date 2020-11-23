#!/usr/bin/env python3

#######################################################################################################################
# eval.py - python script for running eval on a breath rate detection algorithm
# 
# usage: ./eval.py <data-dir>
#######################################################################################################################
import argparse
import random


class RandomBreathRatePredictor(object):
    def predict_rate(self, vid):
        return (60.0 * random.uniform(0, 1))

class Eval(object):
    def __init__(self, data_dir, predictor=RandomBreathRatePredictor()):
        self._data_dir = data_dir

    def evaluate_rmse(self):
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an algorithm for detecting breath rates.')
    parser.add_argument('data_dir', metavar='DATA-DIR', type=str, help='directory containing videos and annotation file')
    args = parser.parse_args()
    print('Using data dir: ' + args.data_dir)
    
    ev = Eval(args.data_dir)
    rmse = ev.evaluate_rmse()

    print('RMSE = {}'.format(rmse))