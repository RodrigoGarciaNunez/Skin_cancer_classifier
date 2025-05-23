import tensorflow as tf
from keras import Sequential, layers, Input, models, utils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import cv2
import gc
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from misc import graficador_bar_pie, plot_history
import time

l_encoder = LabelEncoder()
hot_encoder = OneHotEncoder()

def normalizar_images(data_df:pd.DataFrame, images:list):
    
    images_ids = (data_df['isic_id'])
    for i, id in enumerate(images_ids):
        img  = cv2.imread(f'ISIC-images/{id}.jpg',cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img,(32, 32))
            img = img.astype('float32') / 255.0 #normalizar   
            images.append(img)
            
        del img
        if i % 100 == 0:
            gc.collect()
            cv2.waitKey(1)

def preparar_data(data_df:pd.DataFrame, objetivo:str):
    
    X = data_df.drop(columns=[objetivo, 'benign_malignant' if objetivo == 'diagnosis' else 'diagnosis'])
    y  = data_df[objetivo]
    images=[]

    if y.dtype == 'object':
        y = l_encoder.fit_transform(y)

        y = utils.to_categorical(y)



    normalizar_images(data_df, images)

    X.drop(columns='isic_id', inplace=True)
    return X, y, images


if __name__ == '__main__':


    for objetivo in ['diagnosis']: 
        loss='categorical_crossentropy'
        metrics=['categorical_accuracy']
        output_activation = 'softmax'


        ###### Preparación de los Datos ######################


        print(f'\n----Preparando Datos para {objetivo}----\n')
        data =  pd.read_csv(f'data/ham10000_metadata_balanced_{objetivo}.csv', engine = 'c')
        test_data = pd.read_csv(f'data/ham10000_metadata_leftover_{objetivo}.csv', engine = 'c').sample(frac=0.5)
        
        graficador_bar_pie(data, objetivo, f'balanceo/{objetivo}/train_plus_val')
        #se toma una muestra del dataset balanceado y esa muestra se elimina del dataset original
        val_data =data.sample(frac=0.5, random_state=42)
        data = data.drop(val_data.index)

        # data = generador_de_registros_images(data)
        # val_data = generador_de_registros_images(val_data)
        # test_data = generador_de_registros_images(test_data)

        missing_classes = val_data[~val_data['diagnosis'].isin(test_data['diagnosis'])]
        missing = missing_classes['diagnosis'].unique()

        if missing.size > 0:
            print(f'Hay clases que no estan en test: {missing} ')
            for clase in missing:
                test_data = pd.concat([test_data, val_data[val_data['diagnosis']==clase]], ignore_index=True)

        repeated_registers = data[data['isic_id'].isin(val_data['isic_id'])]
        repeated_registers = repeated_registers['isic_id'].unique()
        print(f'\nNúmero de ids repetidos en datos de entrenamietno y validación: {repeated_registers}\n')

        data.to_csv(f'data/data_img_{objetivo}.csv', index=False)
        val_data.to_csv(f'data/val_data_{objetivo}.csv', index=False)

        print("\nDisposicion de los datos: \n")
        print(f'data columns: {data.columns.size}, test_columns: {test_data.columns.size} y val_columns: {val_data.columns.size}')
        print(f'data shape: {data.shape}, test_shape: {test_data.shape} y val_shape: {val_data.shape}')
        
        if objetivo == 'benign_malignant':
            loss= 'binary_crossentropy'
            metrics= ['accuracy']
            output_activation = 'sigmoid'
            data = pd.get_dummies(data, columns=['diagnosis'])
            test_data = pd.get_dummies(test_data, columns=['diagnosis'])
            val_data = pd.get_dummies(val_data, columns=['diagnosis'])

        # concat = pd.concat([data, val_data])
        # graficador_bar_pie(concat, objetivo, f'balanceo/{objetivo}/train_plus_val')

        X_train, y_train, images_train = preparar_data(data, objetivo)
        images_train = np.array(images_train)
        y_train = np.array(y_train)

        clase_a_num = dict(zip(l_encoder.classes_, l_encoder.transform(l_encoder.classes_)))
        print("Diccionario de Clases: ", clase_a_num)
        
        X_val, y_val, images_val = preparar_data(val_data, objetivo)
        images_val = np.array(images_val)
        y_val= np.array(y_val)

        X_tests, y_test, images_test = preparar_data(test_data, objetivo)
        images_test = np.array(images_test)
        y_test = np.array(y_test)

        num_output = np.unique(y_train).size
        
        if num_output <= 2:
            num_output = 1
        print(num_output)

        print("\nX_train shape: ", X_train.shape)
        print("images_train shape: ", images_train.shape)
        print("y_train shape :", y_train.shape)

        print("\nX_val shape:", X_val.shape)
        print("images_val shape:", images_val.shape)
        print("y_val shape:", y_val.shape)

        print("\nX_test shape: ", X_tests.shape)
        print("images_test shape: ", images_test.shape)
        print("y_test shape: ", y_test.shape)

        print(f'{set(X_train) - set(X_tests)}')
        num_input_columns = (X_train.columns.size)

        ####### definición del modelo #########################3
     
        image_input =  Input(shape=(32, 32, 3), name= "image_input")
        metadata_input = Input(shape=(num_input_columns,), name = "metadata")

        x = layers.Conv2D(16, (3, 3), activation='leaky_relu')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        # x = layers.Conv2D(32, (3, 3), activation='leaky_relu')(x)
        # x = layers.MaxPooling2D((2, 2))(x)
        # x = layers.Conv2D(16, (3, 3), activation='leaky_relu')(x)
        # x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)  
        
        x = layers.Concatenate()([x, metadata_input])        

        x = layers.Dense(16, activation='leaky_relu')(x)
        x = layers.Dropout(0.5)(x)
        # x = layers.Dense(16, activation='leaky_relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(8, activation='leaky_relu')(x)
        # x = layers.Dropout(0.5)(x)

        output = layers.Dense(8, activation=output_activation)(x)

        model = models.Model(inputs = [image_input, metadata_input], outputs = output)
        
        model.compile(optimizer='adam',
              loss=loss,
              metrics=metrics)
        
        model.summary()

        utils.plot_model(model, to_file='graficos/model_performance/model.png',show_shapes=True, show_layer_names=True)
        
        ##################### entrenamiento #######################3
        
        trainning = model.fit(
            x = {
                    'image_input': images_train,
                    'metadata': X_train                
            },
            y = y_train,
            epochs=60,
            batch_size= 64, 
            validation_data=(
                {
                    'image_input': images_val,
                    'metadata': X_val
                
            },
            y_val)
        )


        model.save('../skin_cancer_app/models/skin_cancer_model.h5')
        plot_history(trainning, [metrics, loss], objetivo)

        ################## evaluación ############################

        print("Evaluate on test data")
        results = model.evaluate(
            x={
                'image_input': images_test,
                'metadata': X_tests 
            },
            y=y_test)

        print("test loss, test acc:", results)



        ################## TEST ##################################33

        print("\nGenerar predicciones de 30 registros aleatorios de test")


        random_samples_ids = np.random.choice(len(images_test), size=1000, replace=False)
        start_time = time.time()
        predictions = model.predict(x ={
            'image_input': images_test[random_samples_ids],
            'metadata': X_tests.iloc[random_samples_ids] 
        })

        end_time = time.time()

        if output_activation == 'softmax':
            predicted_probabilities = tf.nn.softmax(predictions, axis=1)
            predicted_labels = np.argmax(predicted_probabilities, axis=1)  
            original_prob = tf.nn.softmax(y_test[random_samples_ids], axis=1)
            original_labels = np.argmax(original_prob,axis=1)     
            print(f"\n comparacion predicciones {predicted_labels} vs. \noriginal: {original_labels}\n")
            print(f"\n tiempo total de prediccion {end_time - start_time} y tiempo por prediccion {(end_time - start_time)/1000}\n")
            
            # Precisión general
            accuracy = accuracy_score(original_labels, predicted_labels)
            print(f"Precisión (accuracy): {accuracy:.4f}")

            # Reporte completo: precisión, recall (sensibilidad) y F1 por clase
            print(classification_report(original_labels, predicted_labels, digits=4))
            
            cm = confusion_matrix(original_labels, predicted_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix= cm)
            disp.plot()
            plt.savefig(f'graficos/model_performance/confusion_matrix_{objetivo}.png')
            plt.show()
            plt.close()

            specificities = []
            for i in range(len(cm)):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                specificity = tn / (tn + fp)
                specificities.append(specificity)

            for idx, spec in enumerate(specificities):
                print(f"Especificidad clase {idx}: {spec:.4f}")
            
        else:
            y_pred_classes = (predictions > 0.5).astype("int")
            print(f"comparacion {y_pred_classes} vs. {y_test[random_samples_ids]} \n")
        
        print("predictions shape:", predictions.shape)