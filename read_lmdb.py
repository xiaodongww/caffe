import caffe
import lmdb
import numpy as np

lmdb_env = lmdb.open('mnist_test_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)  # data is relevant to a sample in the lmldb
