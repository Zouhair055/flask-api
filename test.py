import requests

url = 'http://localhost:5000/predict'  # Update to /predict
files = {'image': open('1.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.text)  # Affiche la catégorie détectée

