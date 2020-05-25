import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt  
#%matplotlib inline
import pandas as pd
from tensorflow.keras.preprocessing import image   # for preprocessing the images
import numpy as np 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

from skimage.transform import resize
from datetime import datetime
import scipy
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.debugging.set_log_device_placement(True)
#tf.logging.set_verbosity(tf.logging.ERROR)
#import tensorflow.contrib.eager as tfe
tf.compat.v1.disable_eager_execution()

#vgg = VGG16(input_shape=None,weights='imagenet',include_top=True)

def load_preprocess_img(p,shape = None):
    Img = image.load_img(p, target_size=shape)
    X = image.img_to_array(Img)
    X = np.expand_dims(X,axis=0)    
    X = preprocess_input(X)
    return X

def preprocess_img(frame,shape = None):
    X = np.expand_dims(frame,axis=0)    
    X = preprocess_input(X.astype(('float64')))
    return X
    

#Loading style image
    
style_img = load_preprocess_img(p = './Style_Image(starrynight).jpg', shape=(224,224))
batch_shape = style_img.shape
shape = style_img.shape[1:]

#shape = (224,224,3)

#Content model define
def vgg_avg_pooling(shape):
    vgg = VGG16(input_shape=shape,weights='imagenet',include_top=False)
    model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
        # replace it with average pooling    
            model.add(AveragePooling2D())
        else:
            model.add(layer)
    return model   

def vgg_cutoff(shape,num_conv):
    if num_conv<1|num_conv>13:
        print('Error layer must be with in [1,13]')
    model = vgg_avg_pooling(shape)
    new_model = Sequential()
    n=0
    for layer in model.layers:
        new_model.add(layer)
        if layer.__class__ == Conv2D:
            n+=1
        if n >= num_conv:
            break
    return new_model

#Style loss comutation graph

def gram_matrix(img):
    # input is (H, W, C) (C = # feature maps)
    # we first need to convert it to (C, H*W)
   
    #X = tf.transpose(tf.reshape(img,[h*w,c]))
    
    X = K.batch_flatten(K.permute_dimensions(img,(2,0,1)))
    
    # now, calculate the gram matrix
    # gram = XX^T / N
    gram_mat = K.dot(X,K.transpose(X))/img.get_shape().num_elements()
    return gram_mat 

def style_loss(y,t):
    channels = 3
    img_nrows,img_ncols,_ = y.shape
    size = img_nrows * img_ncols
    x =  tf.cast(K.sum(K.square(gram_matrix(y)-gram_matrix(t))),tf.float32)
    ans = x/tf.cast(4.0*9*(size*size),tf.float32)
    return ans

def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale(x):
    x = x-x.min()
    x=x/x.max()
    return x

def total_variation_loss(x):
    #assert K.ndim(x) == 4
    _,img_nrows,img_ncols,_ = x.shape
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


#Function to minimize loss
def min_loss(fn,epochs,batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = scipy.optimize.fmin_l_bfgs_b(func=fn,x0=x,maxfun=20)
    # bounds=[[-127, 127]]*len(x.flatten())
    #x = np.clip(x, -127, 127)
    # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
    print("duration:", datetime.now() - t0)
    #plt.plot(losses)
    #plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]

def main():

    # Create a VideoCapture object
    batch_shape = style_img.shape
    shape = style_img.shape[1:]
    cap = cv2.VideoCapture('./input_video.avi')
    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4)) 
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    count = 0 
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #out = cv2.VideoWriter('outpy.avi',fourcc,20.0,(224,224))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break

        print(str(count),'th frame processing')
        
        frame = cv2.resize(frame,(224,224))
        X = preprocess_img(frame)
        vgg = vgg_avg_pooling(shape=shape)
        content_model = Model(vgg.input,vgg.layers[13].get_output_at(0))
        content_target = content_model.predict(X)
        symb_conv_outputs = [layer.get_output_at(1) for layer in \
                            vgg.layers if layer.name.endswith('conv1')]
        multi_output_model = Model(vgg.input, symb_conv_outputs)
        symb_layer_out = [K.variable(y) for y in multi_output_model.predict(style_img)]
        weights = [0.2,0.4,0.3,0.5,0.2]
        
        
        loss=K.sum(K.square(content_model.output-content_target)) 
        for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
            loss += 0.03 * w * style_loss(symb[0],actual[0])
        
        loss +=  0.1*total_variation_loss(symb_conv_outputs[-1])
        grad = K.gradients(loss,vgg.input)
        get_loss_grad = K.function(inputs=[vgg.input], outputs=[loss] + grad)
        def get_loss_grad_wrapper(x_vec):
            l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
            return l.astype(np.float64), g.flatten().astype(np.float64)
        
        final_img = min_loss(fn=get_loss_grad_wrapper,epochs=100,batch_shape=batch_shape)
        #plt.imshow(scale(final_img))
        #plt.show()
        
        
        
        
        filename ="./style_images_test/frame%d.jpg" % count;count+=1 
        cv2.imwrite(filename, final_img)

    cv2.destroyAllWindows()

    #Converting images to video clipping

    image_folder = './style_images_test'
    video_name = 'output_videos/output_video_test.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(video_name, fourcc, count+1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

#main()


def get_image(path = './Stata_Center1.jpg'):

    frame = load_preprocess_img(p = path, shape=(224,224))
    X = preprocess_img(frame)
    X = X[0,:,:,:,:]
    print('the shape of image is', str(X.shape))
    vgg = vgg_avg_pooling(shape=shape)
    content_model = Model(vgg.input,vgg.layers[13].get_output_at(0))
    content_target = content_model.predict(X)
    symb_conv_outputs = [layer.get_output_at(1) for layer in \
                            vgg.layers if layer.name.endswith('conv1')]
    multi_output_model = Model(vgg.input, symb_conv_outputs)
    symb_layer_out = [K.variable(y) for y in multi_output_model.predict(style_img)]
    weights = [0.2,0.4,0.3,0.5,0.2]
        
        
    loss=K.sum(K.square(content_model.output-content_target)) 
    for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
        loss += 0.03 * w * style_loss(symb[0],actual[0])
    
    loss+= 0.001*total_variation_loss(symb_conv_outputs[-1])
        
    grad = K.gradients(loss,vgg.input)
    get_loss_grad = K.function(inputs=[vgg.input], outputs=[loss] + grad)
    def get_loss_grad_wrapper(x_vec):
        l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    print('start synthesizing stylized image')
        
    final_img = min_loss(fn=get_loss_grad_wrapper,epochs=1000,batch_shape=batch_shape)
        
        
    filename ="./stylized_img.jpg" 
    cv2.imwrite(filename, final_img)

get_image()







