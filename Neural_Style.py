# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
from PIL import Image
import imageio
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import tensorflow.compat.v1 as tf
import streamlit.components.v1 as components
tf.disable_v2_behavior()

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <a class="btn btn-outline-secondary float-left" href="https://github.com/DhirajNavale/Neural-Styling" target="_blank" role="button"><i class="fa fa-github" style="font-size:48px;" aria-hidden="true"></i>Click Here For Code</a>
    """,
    height=100,
)
st.title("Play with Neural Style Network")
def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size = (img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(
     x[:, :img_height - 1, :img_width - 1, :] -
     x[:, 1:, :img_width - 1, :])
    b = K.square(
     x[:, :img_height - 1, :img_width - 1, :] -
     x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

col1, col2 = st.beta_columns(2)
width, height = 0
target_image_path = st.sidebar.file_uploader("Upload Target Image", type=['png', 'jpeg', 'jpg'])
style_reference_image_path = st.sidebar.file_uploader("Upload Source Image", type=['png', 'jpeg', 'jpg'])
epochs = st.sidebar.selectbox("Number Of Epochs", [None, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
iterations = st.sidebar.selectbox("After how many iteration you want to see the image ", [None, 2, 6, 8, 10, 20])
j = 0
if target_image_path and style_reference_image_path:
    target_image = Image.open(target_image_path)
    width, height = target_image.size
    source_file = Image.open(style_reference_image_path)
    col1.image(target_image, channels='RGB', width=200, caption="Target Image")
    col2.image(source_file, channels='RGB', width=300, caption="Source Image")

result_prefix = 'my_result'


if target_image_path and style_reference_image_path and epochs and iterations:
    if st.sidebar.button("Start"):
       # width, height = load_img(target_image_path).size
        img_height = height
        img_width = int(width * img_height / height)
        target_image = K.constant(preprocess_image(target_image_path, img_height=img_height, img_width=img_width))
        style_reference_image = K.constant(preprocess_image(style_reference_image_path, img_height=img_height, img_width=img_width))
        combination_image = K.placeholder((1, img_height, img_width, 3))
        input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)
        model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        st.sidebar.write('Model loaded.')
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        content_layer = 'block5_conv2'
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        total_variation_weight = 1e-2
        style_weight = 1.
        content_weight = 0.025
        loss = K.variable(0.)

        layer_features = outputs_dict[content_layer]
        target_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + content_weight * content_loss(target_image_features,
                                                    combination_features)
        for layer_name in style_layers:
            layer_features = outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(style_layers)) * sl
        loss += total_variation_weight * total_variation_loss(combination_image)

        grads = K.gradients(loss, combination_image)[0]
        fetch_loss_and_grads = K.function([combination_image], [loss, grads])
        evaluator = Evaluator()
        x = preprocess_image(target_image_path, img_height=img_height, img_width=img_width)
        x = x.flatten()
        my_bar = st.sidebar.progress(0)
        col_1, col_2, col_3 = st.beta_columns(3)
        loss_val = st.sidebar.write()

        for i in range(epochs):
            st.sidebar.write('Start of iteration', i + 1)
            #start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                            x,
                                            fprime=evaluator.grads,
                                            maxfun=20)
            st.sidebar.write('Current loss value:', min_val)
            if i % iterations == 0:
                j +=1
                img = x.copy().reshape((img_height, img_width, 3))
                img = deprocess_image(img)
                fname = result_prefix + '_at_iteration_%d.png' % i
                imageio.imwrite(fname, img)
                st.sidebar.write('Image saved as', fname)
                img_converted = Image.open(fname)
                if j % 3 == 0:
                    col_3.image(img_converted, channels='RGB', width=200, caption="Image at Iteration" + str(i + 1))
                elif j % 2 == 0:
                    col_2.image(img_converted, channels='RGB', width=200, caption="Image at Iteration" + str(i + 1))
                else:
                    col_1.image(img_converted, channels='RGB', width=200, caption="Image at Iteration" + str(i + 1))

            my_bar.progress(round((i % epochs) * 10, 2))
        st.balloons()
