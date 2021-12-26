curl -d '{"ID_USER": 0, "genero": "F", "monto": "608,3456335", "fecha": "21/01/2020", "hora": 20, 
"dispositivo": "{'model': 2020; 'device_score': 3; 'os': 'ANDROID'}", "establecimiento": "Super", 
"ciudad": "Merida", "tipo_tc": "F\u00c3\u00adsica", "linea_tc": 71000, "interes_tc": 51, 
"status_txn": "Aceptada", "is_prime": false, "dcto": "60,83456335", "cashback": "5,475110702", 
"fraude": false}' -H 'Content-Type: application/json'  http://localhost:8000/v1/predict


curl -d '{"ID_USER": 0, "genero": "F", "monto": "608.3456335", "fecha": "21/01/2020", "hora": 20, "dispositivo": "{'model': 2020; 'device_score': 3; 'os': 'ANDROID'}", "establecimiento": "Super", "ciudad": "Merida", "tipo_tc": "Fisica", "linea_tc": 71000, "interes_tc": 51, "status_txn": "Aceptada", "is_prime": false, "dcto": '60.83456335', "cashback": "5.475110702"}' -H 'Content-Type: application/json'  http://localhost:8000/v1/predict

curl -d '{"ID_USER": 0, "genero": "F", "monto": "608.3456335", "fecha": "21/01/2020", "hora": 20, "dispositivo": "ANDROID", "establecimiento": "Super", "ciudad": "Merida", "tipo_tc": "Fisica", "linea_tc": 71000, "interes_tc": 51, "status_txn": "Aceptada", "is_prime": false, "dcto": "60.83456335", "cashback": "5.475110702"}' -H 'Content-Type: application/json'  http://localhost:8000/v1/predict


