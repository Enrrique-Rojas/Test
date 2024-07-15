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
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import csv
import mysql.connector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Configurar la conexión a la base de datos
config = {
  'user': 'freedb_Enrrique',
  'password': 'MJCfD8mM&H2yvs!',
  'host': 'sql.freedb.tech',
  'database': 'freedb_PredectionDesertion',
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
    classifiers_precision = {}
    classifiers_recall = {}
    classifiers_f1 = {}
    ### Implement Classifier SVC
    classifier=svm.SVC(kernel='linear')
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifier_name = 'SVC'
    classifiers_score[classifier_name] = score_train
    ### Metrics
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    classifiers_precision[classifier_name] = precision
    classifiers_recall[classifier_name] = recall
    classifiers_f1[classifier_name] = f1
    classifiers_f1[classifier_name+'_2'] = calculate_f1(precision, recall)

    ### Implement Classifier KNeighborsClassifier
    classifier= KNeighborsClassifier()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifier_name = 'KNeighborsClassifier'
    classifiers_score[classifier_name] = score_train
    ### Metrics
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    classifiers_precision[classifier_name] = precision
    classifiers_recall[classifier_name] = recall
    classifiers_f1[classifier_name] = f1
    classifiers_f1[classifier_name+'_2'] = calculate_f1(precision, recall)

    ### Implement Classifier LogisticRegression
    classifier=LogisticRegression()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifier_name = 'LogisticRegression'
    classifiers_score[classifier_name] = score_train
    ### Metrics
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    classifiers_precision[classifier_name] = precision
    classifiers_recall[classifier_name] = recall
    classifiers_f1[classifier_name] = f1
    classifiers_f1[classifier_name+'_2'] = calculate_f1(precision, recall)

    ### Implement Classifier DecisionTreeClassifier
    classifier= DecisionTreeClassifier()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)
    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifier_name = 'DecisionTreeClassifier'
    classifiers_score[classifier_name] = score_train
    ### Metrics
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    classifiers_precision[classifier_name] = precision
    classifiers_recall[classifier_name] = recall
    classifiers_f1[classifier_name] = f1
    classifiers_f1[classifier_name+'_2'] = calculate_f1(precision, recall)

    ### Implement Classifier RandomForestClassifier
    classifier=RandomForestClassifier()
    classifier.fit(X_train,y_train)
    y_pred_train =classifier.predict(X_train)

    precision, recall, thresholds = precision_recall_curve(y_train, y_pred_train)
    # Find umbral waited
    desired_recall = 0.90
    threshold_recall = thresholds[np.where(recall >= desired_recall)][0]
    y_pred_train_recall = (y_pred_train >= threshold_recall).astype(int)

    score_train=accuracy_score(y_pred_train ,y_train)
    y_pred_test=classifier.predict(X_test)
    score_test=accuracy_score(y_pred_test,y_test)
    classifier_name = 'RandomForestClassifier'
    classifiers_score[classifier_name] = score_train
    #Metrics
    matriz = confusion_matrix(y_train, y_pred_train )
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train_recall)
    f1 = calculate_f1(precision, recall)
    auc = roc_auc_score(y_train, y_pred_train)
    classifiers_precision[classifier_name] = precision
    classifiers_recall[classifier_name] = recall
    classifiers_f1[classifier_name] = f1_score(y_train, y_pred_train)
    classifiers_f1[classifier_name+'_2'] = f1

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
        'classifiers_precision': classifiers_precision,
        'classifiers_recall': classifiers_recall,
        'classifiers_f1': classifiers_f1,
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
    classifier=RandomForestClassifier()
    classifier.fit(X_train,y_train)
    ### Check Accuracy on the training data
    y_pred=classifier.predict(X_train)
    score_train=accuracy_score(y_pred,y_train)
    ### Check Accuracy on the test data
    y_pred=classifier.predict(X_test)
    score_test=accuracy_score(y_pred,y_test)

    data = select_data_where_prediction(cursor)
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
        """ repeatedCoursesStudent=data[i][11]
        languageLevelStudent=data[i][12]
        computingLevelStudent = data[i][13]
        if repeatedCoursesStudent > 2 and languageLevelStudent == 1 and computingLevelStudent == 1:
            prediction = 1 """
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

@app.post('/list')
def list_students():
    # Establecer la conexión
    try:
        conn = mysql.connector.connect(**config)
        print("Conexión exitosa a MySQL")
    except mysql.connector.Error as err:
        print("Error de conexión:", err)

    # Crear un cursor para ejecutar consultas
    cursor = conn.cursor()
    data = select_data_prediction(cursor)
    if not data:
        response = {
        'sucess': False,
        'message': "No existe datos"
        }
        cursor.close()
        conn.close()
        return response
    else:
        response = {
        'sucess': True,
        'message': "Lista de datos",
        'result': data
        }
        cursor.close()
        conn.close()
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

def select_data_where_prediction(cursor):
    try:
        query = ("SELECT *"
        "FROM prediction WHERE prediction = FALSE")
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except mysql.connector.Error as err:
        print("Error al obtener datos:", err)
        return None

def select_data_prediction(cursor):
    try:
        query = ("SELECT nameStudent,enrolledCycle,created_at,outcome FROM prediction ORDER BY created_at DESC")
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

def calculate_f1(precision_param, recall_param):
    divisor_plus = (precision_param+recall_param) 
    divisor = divisor_plus if divisor_plus>0 else 1
    dividendo = (2 * precision_param * recall_param)
    return  dividendo/divisor
