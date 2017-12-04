# This is a script for 10-crop test in caffe
# The path in the script is not modified due to tight time


import caffe
import numpy as np
import cPickle as pickle
import os 
caffe.set_mode_gpu()
caffe.set_device(3)

model_name = 'Places2-365-CNN-step2_lr_6_iter_5000.caffemodel'
net_name = 'Places2-365-CNN-deploy'
# step = ''
step = 'step2'
dataset = 'testa'



def multi_crop_predict(model_name, net_name, dataset):
    caffe_root = '/home/wuxiaodong/caffe-augmentation-master/'
    IMAGE_ROOT = '/home/wuxiaodong/data/ai_challenge_linked/' + dataset + '_imgs/'
    IMAGE_NAME_FLIE = '/home/wuxiaodong/data/ai_challenge_linked/' + dataset + '.txt'

    model_file = caffe_root + '/examples/ai_challenge/' + net_name + '.prototxt'   # unprepared
    pretrained = caffe_root + '/examples/ai_challenge/snapshots/' + model_name
    print(model_file)
    print(pretrained)
    print(IMAGE_NAME_FLIE)
    if os.path.exists(pretrained):
        pass
    else:
        pretrained = caffe_root + '/examples/ai_challenge/' + model_name
    # image_file = '00a58de1e260033ed972a7e322a2d8fd315cece6.jpg'
    mean = np.ones([3,224,224], dtype=np.float)
    mean[0,:,:] = 114.005
    mean[1,:,:] = 121.639
    mean[2,:,:] = 125.972

    net = caffe.Classifier(model_file, pretrained, mean=mean, channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))
    # net.set_phase_test()

    predictions = []
    i = 0
    with open(IMAGE_NAME_FLIE) as f:
        for line in f:
            image_file = line.split(' ')[0].strip()
            # print(image_file)
            input_image = caffe.io.load_image(IMAGE_ROOT + image_file)
            prediction = net.predict([input_image], oversample=True)
            predictions.append(list(prediction.reshape(80)))
            i += 1
            if i%100==0:
                print(i)
    predictions = np.array(predictions)
    with open('./results/' + net_name + step + '-' + dataset + '-10crop.pickle', 'w') as f:
        pickle.dump(predictions, f)


model_name = 'ResNet152-places365-lre7_iter_28000.caffemodel'
net_name = 'ResNet152-places365-deploy'
# step = ''
step = '1203'
# multi_crop_predict(model_name, net_name, 'testa')
multi_crop_predict(model_name, net_name, 'testb')
