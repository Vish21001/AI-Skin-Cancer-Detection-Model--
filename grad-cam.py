import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("skin_cancer_model.h5")

def make_gradcam(img_path, class_index):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128,128))
    img_input = img_resized.reshape(1,128,128,3)/255.0

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-3).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs.numpy()[0]
    heatmap = np.mean(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)/np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
