from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import base64
import requests
import json
import io

app = Flask(__name__)
baseUrl = 'https://shark-app-mo2mz.ondigitalocean.app'
url = f'{baseUrl}/v1/models/cars_model:predict'

# Habilitar CORS para todos los orígenes


@app.after_request
def enable_cors(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


@app.route('/prediction', methods=['POST'])
def predict():
    # Obtener la imagen base64 del cuerpo de la solicitud
    image_base64 = request.json['image']

    # Decodificar la imagen base64 a bytes
    image_bytes = base64.b64decode(image_base64)

    # Crear una imagen PIL desde los bytes decodificados
    image = Image.open(io.BytesIO(image_bytes))

    # Cambiar el tamaño de la imagen a 224x224
    image = image.resize((224, 224))

    # Convertir la imagen en un arreglo numpy
    image_array = np.array(image)

    # Normalizar la imagen
    image_array = image_array / 255.0

    # Convertir la imagen en una lista y agregar una dimensión adicional
    image_list = [image_array.tolist()]

    # Preparar los datos de entrada en el formato esperado por el modelo
    payload = {
        'instances': image_list
    }
    headers = {'content-type': 'application/json'}

    # Realizar la solicitud HTTP para la predicción
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # Analizar la respuesta JSON
    response_json = json.loads(response.text)

    # Comprobar si la clave 'predictions' está presente en la respuesta
    if 'predictions' in response_json:
        predictions = response_json['predictions']
    elif 'outputs' in response_json:
        predictions = response_json['outputs']
    else:
        raise KeyError(
            "No se pudieron recuperar las predicciones del JSON de respuesta.")

    # Retornar las predicciones como respuesta
    return jsonify(predictions)


if __name__ == '__main__':
    app.run()
