# -*- coding: utf-8 -*-
"""
Created on April 27 2024
@authors: E. ROJAS - K. QUISPE 
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from Student import Student
#import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import csv
import mysql.connector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Configurar la conexión a la base de datos
config = {
  'user': 'Enrrique',
  'password': 'Clave123@',
  'host': 'Enrrique.mysql.pythonanywhere-services.com',
  'database': 'Enrrique$PredictionDesertion',
}

app = FastAPI()

@app.post('/information')
def get_information():

    try:
        conn = mysql.connector.connect(**config)
        print("Conexión exitosa a MySQL")
    except mysql.connector.Error as err:
        print("Error de conexión:", err)
        return {
            "sucess": False,
            "message": "Error de conexión"
        }
    cursor = conn.cursor()
    # Lista de campos en la tabla Prediction
    fields = [
        'gender',
        'age',
        'birthDistrict',
        'currentDistrict',
        'civilStatus',
        'typeInstitution',
        'provenanceLevel',
        'disability',
        'enrolledCycle',
        'repeatedCourses',
        'languageLevel',
        'computingLevel',
        'isForeign'
    ]

    # Consulta SQL para obtener el valor que más se repite en cada campo
    queries = {}
    for field in fields:
        query = f"""
            SELECT {field}, COUNT(*) AS count
            FROM prediction
            WHERE outcome=1
            GROUP BY {field}
            ORDER BY count DESC
            LIMIT 1;
        """
        queries[field] = query

    # Ejecutar las consultas
    results = {}
    for field in fields:
        cursor.execute(queries[field])
        result = cursor.fetchone()
        field_name = result[0]  # Tipo
        count = result[1]       # Total de dicho tipo
        results[field] = {
            field_name:count
        }

    max_field_count = {}
    max_count = 0

    for field, field_count in results.items():
        key = list(field_count.keys())[0]
        count = field_count[key]
        if(max_count < count):
            max_count = count
            max_field_count = {
                field: field_count
            }
    result_order = sorted(results.items(), key=lambda x: list(x[1].values())[0], reverse=True)

    cursor.close()
    conn.close()

    response = {
        "sucess": True,
        "message": "Correcto",
        "max_count": max_field_count,
        "result": result_order
    }

    return response


@app.post('/')
def index():
    df=pd.read_csv('Student.csv')

    X = df.drop(columns='outcome', axis=1)
    y = df['outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=0)

    classifiers_score = {}
    ### Implement Classifier SVC
    classifier=svm.SVC(kernel='linear')
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifiers_score['SVC'] = score_train

    ### Implement Classifier KNeighborsClassifier
    classifier= KNeighborsClassifier()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifiers_score['KNeighborsClassifier'] = score_train

    ### Implement Classifier LogisticRegression
    classifier=LogisticRegression()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifiers_score['LogisticRegression'] = score_train

    ### Implement Classifier RandomForestClassifier
    classifier=RandomForestClassifier()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifiers_score['RandomForestClassifier'] = score_train

    ### Implement Classifier DecisionTreeClassifier
    classifier= DecisionTreeClassifier()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)

    #Metrics
    matriz = confusion_matrix(y_train, y_pred_train )
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    auc = roc_auc_score(y_train, y_pred_train)

    '''### Create a Pickle file using serialization 
    pickle_out = open("classifier.pkl","wb")
    pickle.dump(classifier, pickle_out)
    pickle_out.close()''' 
    return JSONResponse(content={
        "confusion_matrix": matriz.tolist(),
        'precision' : precision,
        'recall': recall,
        'f1': f1,
        'score_train': score_train, 
        'classifiers_score': classifiers_score,
        'auc': auc
        })

@app.post('/save')
def save(data: Student):
    try:
        conn = mysql.connector.connect(**config)
        print("Conexión exitosa a MySQL")
    except mysql.connector.Error as err:
        print("Error de conexión:", err)
    cursor = conn.cursor()
    student_model = data.model_dump()
    name=student_model['name']
    gender=student_model['gender']
    age=student_model['age']
    birthDistrict=student_model['birthDistrict']
    currentDistrict=student_model['currentDistrict']
    civilStatus=student_model['civilStatus']
    typeInstitution=student_model['typeInstitution']
    provenanceLevel=student_model['provenanceLevel']
    disability=student_model['disability']
    enrolledCycle=student_model['enrolledCycle'] 
    repeatedCourses=student_model['repeatedCourses']
    languageLevel=student_model['languageLevel']
    computingLevel=student_model['computingLevel']
    isForeign=student_model['isForeign']

    data_insert = (name,gender,age,birthDistrict,currentDistrict,civilStatus,typeInstitution,provenanceLevel,disability,enrolledCycle,repeatedCourses,languageLevel,computingLevel,isForeign, None)
    insert_data(conn,cursor,data_insert)
    cursor.close()
    conn.close()
    response = {
        'success': True,
        'message': "Correcto",
    }
    return response

@app.post('/predict')
def predict_dropout():
    # Establecer la conexión
    try:
        conn = mysql.connector.connect(**config)
        print("Conexión exitosa a MySQL")
    except mysql.connector.Error as err:
        print("Error de conexión:", err)

    # Crear un cursor para ejecutar consultas
    cursor = conn.cursor()

    df=pd.read_csv('Student.csv')

    X = df.drop(columns='outcome', axis=1)
    y = df['outcome']

    scaler = StandardScaler()
    scaler.fit(X)

    standardized_data = scaler.transform(X)
    X = standardized_data
    y = df['outcome']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

    ### Implement Classifier
    classifier=svm.SVC(kernel='linear')
    classifier.fit(X_train,y_train)

    ### Check Accuracy on the training data
    y_pred=classifier.predict(X_train)
    score_train=accuracy_score(y_pred,y_train)

    ### Check Accuracy on the test data
    y_pred=classifier.predict(X_test)
    score_test=accuracy_score(y_pred,y_test)

    data = select_data_where_prediction_false(cursor)
    if not data:
        response = {
        'sucess': False,
        'message': "No existe datos"
        }
        cursor.close()
        conn.close()
        return response
    
    students = []
    for student in data:
        gender=student[2]
        age=student[3]
        birthDistrict=student[4]
        currentDistrict=student[5]
        civilStatus=student[6]
        typeInstitution=student[7]
        provenanceLevel=student[8]
        disability=student[9]
        enrolledCycle=student[10]

        repeatedCourses=student[11]
        languageLevel=student[12]
        computingLevel=student[13]
        isForeign=student[14]
        students.append([gender,age,birthDistrict,currentDistrict,civilStatus,
                         typeInstitution,provenanceLevel,disability,enrolledCycle,
                         repeatedCourses,languageLevel,computingLevel,isForeign])


    predictions = classifier.predict(students)
    
    responseStudents = []
    for i, prediction in enumerate(predictions):
        students[i].append(int(prediction))
        studentPrediction = {
            "name":data[i][1],
            "prediction": int(prediction),
        }
        responseStudents.append(studentPrediction)
        update_prediction_by_id(conn, cursor, data[i][0], prediction)
       
    append_data_to_csv('Student.csv',students)
    cursor.close()
    conn.close()
    response = {
        'sucess': True,
        'message': "Correcto",
        'prediction': responseStudents,
        'score_train': score_train,
        'score_test': score_test
    }

    return response

def append_data_to_csv(csv_file, new_data):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for row in new_data:
            writer.writerow(row)
    
def insert_data(conn, cursor, data):
    try:
        query = ("INSERT INTO prediction "
                "(nameStudent, gender, age, birthDistrict, currentDistrict,civilStatus,typeInstitution,provenanceLevel, "
                "disability, enrolledCycle, repeatedCourses, languageLevel, computingLevel, isForeign, outcome) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        cursor.execute(query, data)
        conn.commit() 
        print("Datos insertados exitosamente")
    except mysql.connector.Error as err:
        print("Error al insertar datos:", err)

def select_data_where_prediction_false(cursor):
    try:
        query = ("SELECT *"
        "FROM prediction WHERE prediction = FALSE")
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except mysql.connector.Error as err:
        print("Error al obtener datos:", err)
        return None
    
def update_prediction_by_id(conn, cursor, id, outcome):
    try:
        query = "UPDATE prediction SET prediction = TRUE, outcome = %s WHERE id = %s"
        cursor.execute(query, (int(outcome),id,))
        conn.commit()
        print(f"Registro con id {id} actualizado exitosamente")
    except mysql.connector.Error as err:
        print("Error al actualizar el registro:", err)