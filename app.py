from bleach import clean
import streamlit as st
import os
import util

# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
# ----------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title('Chest X-Ray Analysis')
# def delete_imgs(dirname):
#     for files in os.listdir(dirname):
#         os.remove(os.path.join(dirname, files))
# try:        
#     dirname = os.path.join('tempDir')
#     delete_imgs(dirname)
# except FileNotFoundError:
#     pass

def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
# -------------------------------------------------------------
pos_weights = np.array([0.02 , 0.013, 0.128, 0.002, 0.175, 0.045, 0.054, 0.106, 0.038,
        0.021, 0.01 , 0.014, 0.016, 0.033])

neg_weights = np.array([0.98 , 0.987, 0.872, 0.998, 0.825, 0.955, 0.946, 0.894, 0.962,
        0.979, 0.99 , 0.986, 0.984, 0.967])

pos_contribution = pos_weights * pos_weights 
neg_contribution = neg_weights * neg_weights
# --------------------------------------------------------------
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            # for each class, we add average weighted loss for that class 
            pos_loss = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            neg_loss = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += pos_loss + neg_loss            
        return loss
    return weighted_loss

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file is not None:
        save_uploadedfile(uploaded_file)
        st.write(f"{uploaded_file.name} uploaded successfully")
        analyse = st.button("Analyse")
        if analyse:
            with st.spinner('Please wait while the image is being processed..'):
                # create the base pre-trained model
                base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)
                x = base_model.output
                # add a global spatial average pooling layer
                x = GlobalAveragePooling2D()(x)
                # and a logistic layer
                predictions = Dense(len(labels), activation="sigmoid")(x)
                model = Model(inputs=base_model.input, outputs=predictions)
                model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))
                # --------------------------------------------------------------
                model.load_weights("./nih/pretrained_model.h5")
                df = pd.read_csv("nih/train-small.csv")
                # IMAGE_DIR = "nih/images-small/"
                IMAGE_DIR = os.path.join('tempDir')
                labels_to_show = ['Cardiomegaly', 'Edema', 'Mass', 'Pneumothorax']
                # ---------------------------------------------------------------
                util.compute_gradcam(model, f'{uploaded_file.name}', IMAGE_DIR+'/', df, labels, labels_to_show)
                img = st.image('results.png')
                filename = 'results.png'
                label = 'Download Results'
                with open (filename, 'rb') as f:
                    # encoded = base64.b64encode(f.read())
                    st.download_button(label, data=f.read(), file_name=filename)
        
