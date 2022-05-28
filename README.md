# neural-nala-challenge

En el siguiente repositorio se responde a las actividades y preguntas propuestas por parte de neural design y nala.
### ---> notebooks/eda.ipynb

    EDA: Explore the data and present a thorough analysis

### ---> notebooks/categorization_customer.ipynb

    Propose a categorization of the different customers

### ---> notebooks/machine_learning_model.ipynb

    Build a model that is capable of detecting fraudsters
    Explain how good your model is, which are the most important features, and how the values of the variables influence in the model
    Explain the trade-off of using your model to detect fraudsters

### ---> Link para visualizar la evidencia en producción:

    Put this model into production (locally or preferably in the cloud)

## Lineas de ejecucion para probar el modelo predictivo

Para situarse en el environment
```
pipenv shell
```
Instalar librerias y dependencias
```
pipenv install
```
Para referenciar los modulos desarrollados en analytics
```
export PYTHONPATH=.
```
Luego podemos ver las metricas del modelo random forest tras el entrenamiento y validacion con el archivo ```run.py```
```
python run.py
```
Ejecuta el siguiente comando en una nueva terminal para poder ejecutar la api (interfaz entre servidor y aplicacion)
```
python api/app.py
```
En una nueva terminal ejecutar el siguiente comando, el que nos permitira hacer peticiones a nuestra api y obtener la prediccion de fraude (True/False)
```
curl -d '{"ID_USER": 0, "genero": "F", "monto": "608.3456335", "fecha": "21/01/2020", "hora": 20, "dispositivo": "ANDROID", "establecimiento": "Super", "ciudad": "Merida", "tipo_tc": "Fisica", "linea_tc": 71000, "interes_tc": 51, "status_txn": "Aceptada", "is_prime": false, "dcto": "60.83456335", "cashback": "5.475110702"}' -H 'Content-Type: application/json'  http://localhost:8000/v1/predict
```
# Happy end
