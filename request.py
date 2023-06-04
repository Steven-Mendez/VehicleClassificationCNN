import requests
import json
from PIL import Image
import numpy as np

# URL del servidor TensorFlow Serving
baseUrl = 'https://shark-app-mo2mz.ondigitalocean.app'
url = f'{baseUrl}/v1/models/cars_model:predict'

# Cargue la imagen de entrada y cambie su tamaño a 224x224
image = Image.open('imagen.jpg')
image = image.resize((224, 224))
image_array = np.array(image)

# Normalizar la imagen
image_array = image_array / 255.0

# Convierta la imagen en una lista y agregue una dimensión adicional
image_list = image_array.tolist()
image_list = [image_list]

# Preparar los datos de entrada en el formato esperado por el modelo
payload = {
    'instances': image_list
}
headers = {'content-type': 'application/json'}

# Realice la solicitud HTTP para la predicción
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Analizar la respuesta JSON
response_json = json.loads(response.text)

# Comprueba si la clave de 'predictions' está presente en la respuesta
if 'predictions' in response_json:
    predictions = response_json['predictions']
elif 'outputs' in response_json:
    predictions = response_json['outputs']
else:
    raise KeyError("No se pudieron recuperar las predicciones del JSON de respuesta.")

# Imprime las predicciones
print(predictions)
