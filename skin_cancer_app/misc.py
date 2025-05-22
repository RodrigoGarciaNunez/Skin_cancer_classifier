import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from keras import models
import numpy as np

def procesa_data(dict_data: dict):
    #print(dict_data['age_approx'])
    cols_order = ['isic_id','age_approx','benign_malignant','concomitant_biopsy','melanocytic','sex',
                  'anatom_site_general_anterior torso','anatom_site_general_head/neck','anatom_site_general_lower extremity',
                  'anatom_site_general_oral/genital','anatom_site_general_palms/soles','anatom_site_general_posterior torso',
                  'anatom_site_general_upper extremity','diagnosis_1_benign','diagnosis_1_indeterminate','diagnosis_1_malignant',
                  'diagnosis_2_benign epidermal proliferations','diagnosis_2_benign melanocytic proliferations',
                  'diagnosis_2_benign soft tissue proliferations - fibro-histiocytic','diagnosis_2_benign soft tissue proliferations - vascular',
                  'diagnosis_2_indeterminate epidermal proliferations','diagnosis_2_malignant adnexal epithelial proliferations - follicular',
                  'diagnosis_2_malignant epidermal proliferations','diagnosis_2_malignant melanocytic proliferations (melanoma)',
                  'diagnosis_3_basal cell carcinoma','diagnosis_3_dermatofibroma',"diagnosis_3_melanoma, nos",'diagnosis_3_nevus',
                  'diagnosis_3_pigmented benign keratosis','diagnosis_3_solar or actinic keratosis',"diagnosis_3_squamous cell carcinoma, nos",
                  'diagnosis_confirm_type_confocal microscopy with consensus dermoscopy','diagnosis_confirm_type_histopathology',
                  'diagnosis_confirm_type_serial imaging showing no change','diagnosis_confirm_type_single image expert consensus']
    

    standar_scaler = joblib.load('models/standar_scaler.pkl')
    dummy_columns = joblib.load('models/dummy_columns.pkl')
   
    
    data = pd.DataFrame()

    data = pd.concat([data, pd.DataFrame([dict_data])], ignore_index=True)

    #print(data.dtypes)    


    data['concomitant_biopsy'] =(data['concomitant_biopsy'] == "True").astype(int)
    data['melanocytic'] =(data['melanocytic'] == "True").astype(int)
    data["sex"] = (data["sex"] == "male").astype(int) 
    data['age_approx'] =  standar_scaler.transform(data[["age_approx"]])

    #one_hot_encode 
    columns_to_ignore  = ["concomitant_biopsy", "melanocytic", "sex", 
                        "age_approx"]
    
    data = pd.get_dummies(data, columns=data.drop(columns = columns_to_ignore).columns[0:].tolist(), dtype=int)
    for col in dummy_columns:
        if col not in data.columns:
            data[col] = 0  # Agregar las que falten


    data = data[cols_order]  #se ponen en orden
    data.drop(columns=['isic_id', 'benign_malignant'], inplace=True)

    
    #data.to_csv('example.csv', index=False)  
    
    return data

    

def predict_(metadata, image):

    dict_diagnosis= {0:'actinic keratosis', 1:'basal cell carcinoma', 2:'dermatofibroma', 3:'melanoma', 4:'nevus', 
     5:'pigmented benign keratosis', 6:'squamous cell carcinoma', 7:'vascular lesion'}
    
    skin_cancer_model =  models.load_model('models/skin_cancer_model.h5')
    print(f'{type(metadata)} y {type(image)} y {type(skin_cancer_model)}')

    print(f'{image.shape} y {metadata.shape}')
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Ahora (1, 32, 32, 3)

    predictions = skin_cancer_model.predict(x = {
        'image_input': image,
        'metadata': metadata
        })
    
    predicted_probabilities = tf.nn.softmax(predictions, axis=1)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)  

    return dict_diagnosis[predicted_labels[0]]