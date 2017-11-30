# a script for saving features into lmdb
# 'prob' is the layer expected to be extracted
# '704' is the num of batches  704 * batch_size is the num of all samples
CUDA_VISIBLE_DEVICES=3 ./build/tools/extract_features.bin \
    ./examples/snapshots/***.caffemodel \
    ./examples/***.prototxt \
    prob \
    ./examples/features/prob \
    704 lmdb GPU
