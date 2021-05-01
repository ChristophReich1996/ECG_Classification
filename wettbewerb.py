# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.
Funktionen zum Laden und Speichern der Dateien
@author: Maurice Rohr
"""
import csv
import scipy.io as sio
import os


### Achtung! Diese Funktion nicht verändern.

def load_references(folder='../training/'):
    '''
    Parameters
    ----------
    folder : TYPE, optional
        Ort der Trainingsdaten. The default is '../training/'.
    Returns
    -------
    ecg_leads : list of numpy arrays
        EKG Signale.
    ecg_labels : list of str
        gleiche Laenge wie ecg_leads. Werte: 'N','A','O','~'
    fs : float
        Sampling Frequenz.
    ecg_names : list of str
    '''
    ecg_leads = list()
    ecg_labels = list()
    ecg_names = list()
    fs = 300
    with open(folder + 'REFERENCE.csv') as csv_file:  # Einlesen der Liste mit Dateinamen und Zuordnung
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            data = sio.loadmat(folder + row[0] + '.mat')  # Import der EKG-Dateien
            ecg_lead = data['val'][0]
            label = row[1]
            ecg_leads.append(ecg_lead)
            ecg_labels.append(label)
            ecg_names.append(row[0])
    print(str(len(ecg_leads)) + "\t Dateien wurden geladen.")

    return ecg_leads, ecg_labels, fs, ecg_names


### Achtung! Diese Funktion nicht verändern.

def save_predictions(predictions):
    '''
    Speichert Prädiktion in CSV-Datei
    Parameters
    ----------
    predictions : list of tuples
        ("Name der Datei", Label). Label : ['N','A','O','~']
    Returns
    -------
    None.
    '''
    if os.path.exists("PREDICTIONS.csv"):
        os.remove("PREDICTIONS.csv")

    with open('PREDICTIONS.csv', mode='w', newline='') as predictions_file:
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        for prediction in predictions:
            predictions_writer.writerow([prediction[0], prediction[1]])
        print(str(len(predictions)) + "\t Labels wurden geschrieben.")