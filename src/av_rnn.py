"""
An RNN for the aviation data
"""

import argparse
import csv
import numpy as np
import pandas as pd

class FlightPair:
    def __init__(self, data1, data2, dlen):
        self._data1 = data1 # lat, long, alt
        self._data2 = data2
        self._dlen = dlen

class AvData:
    def __init__(self):
        self._batch_size = 16
        self._max_len = 56 
        self._num_records = None
        self._pair_dict = None 

    def load_data_and_labels(self, data_path):
        self._data_path = data_path
        self._data = pd.read_csv( self._data_path )
        self._train_idx = 0
        self._test_idx = 0

        event_times = self._data["Time"].unique()
        self._pair_dict = {}
        df = self._data
        dlens = []
        for evt in event_times:
            tcodes = df["TCode"][ (df["Time"]==evt) ].unique()
            if len(tcodes) != 2:
                continue
            vdata = []
            ev_name = ".".join(tcodes)
            for tcode in tcodes:
                x = df["Longitude"][ \
                        (df["TCode"]==tcode) & (df["Time"]==evt) ].values
                y = df["Latitude"][ \
                        (df["TCode"]==tcode) & (df["Time"]==evt) ].values
                z = df["Altitude"][ \
                        (df["TCode"]==tcode) & (df["Time"]==evt) ].values
                dlen = len(x)
                if dlen > self._max_len:
                    print("Too long: %d" % (dlen))
                    continue
                dlens.append(dlen)
                data = np.zeros((self._max_len, 3))
                data[:dlen, 0] = x
                data[:dlen, 1] = y
                data[:dlen, 2] = z
                vdata.append(data)
            if len(vdata) != 2:
                continue
            min_len = np.min([dlens[-1], dlens[-2]])
            flp = FlightPair(vdata[0], vdata[1], min_len)
            self._pair_dict[ev_name] = flp
        print("Mean len: %.2f, Median: %.2f, std: %.2f" % (np.mean(dlens),
            np.median(dlens), np.std(dlens)))

        self._num_records = len(self._pair_dict)

    def num_records(self):
        return self._num_records
            

    # For every trajectory, a 'real' pair and 'fake' pair are created, with the 
    # fake pair essentially equivalent to two randomly selected trajectories
    def next_train_batch(self):
        pass


def main(args):
    av_data = AvData()
    av_data.load_data_and_labels("/home/matt.phillips/Repos/mattphillipsphd/" \
            "AviationNearMiss/data/nmacs_08252017.csv")
    print("Number of flight pairs: %d" % (av_data.num_records()))

    n_steps = 20
    n_inputs = 6
    n_neurons = 150
    n_outputs = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

