from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.utils import secure_filename
import cv2 
import numpy as np
from misc import procesa_data, predict_

UPLOAD_FOLDER = 'images'
app = Flask(__name__)
app.secret_key = 'zwertf7yg8uhio'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#model = jb.load('model.pkl')

@app.route("/")
def root():
    return render_template("index.html")


@app.route('/formulario', methods=['POST'])
def formulario():
    # Guarda los datos del formulario en la sesión
    session['datos_formulario'] = request.form.to_dict()

    # Redirige a la página para subir la imagen
    return redirect(url_for('subir_imagen'))

@app.route('/upload_image')
def subir_imagen():
    return render_template('upload_image.html')


@app.route('/predict', methods= ['POST'] )
def predict():
    file = request.files['foto']
    filename = secure_filename(file.filename)
    file_bytes = file.read()

    np_buff = np.frombuffer(file_bytes, np.uint8)
    file_matlike = cv2.imdecode(np_buff, cv2.IMREAD_COLOR)

    file_matlike = cv2.resize(file_matlike, (32,32))
    img = file_matlike.astype('float32') / 255.0 #normalizar  

    metadata= procesa_data(session['datos_formulario'])
    
    pred = predict_(metadata, img)
    session['pred'] = pred
    # for column in session['datos_formulario']:
    #     pass
    #     #print(column.)

    return redirect(url_for('diagnosis'))    

@app.route('/load_data')
def diagnosis():
    pred = session.get('pred', 'Sin predicción')
    return render_template('diagnostico.html', pred = pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)