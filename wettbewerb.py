# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.
Funktionen zum Laden und Speichern der Dateien
"""
__author__ = "Maurice Rohr und Christoph Reich"

from typing import List, Tuple
import csv
import scipy.io as sio
import numpy as np
import os


### Achtung! Diese Funktion nicht veraendern.

def load_references(folder: str = '../training') -> Tuple[List[np.ndarray], List[str], int, List[str]]:
    """
    Parameters
    ----------
    folder : str, optional
        Ort der Trainingsdaten. Default Wert '../training'.
    Returns
    -------
    ecg_leads : List[np.ndarray]
        EKG Signale.
    ecg_labels : List[str]
        Gleiche Laenge wie ecg_leads. Werte: 'N','A','O','~'
    fs : int
        Sampling Frequenz.
    ecg_names : List[str]
        Name der geladenen Dateien
    """
    # Check Parameter
    assert isinstance(folder, str), "Parameter folder muss ein string sein aber {} gegeben".format(type(folder))
    assert os.path.exists(folder), 'Parameter folder existiert nicht!'
    # Initialisiere Listen für leads, labels und names
    ecg_leads: List[np.ndarray] = []
    ecg_labels: List[str] = []
    ecg_names: List[str] = []
    # Setze sampling Frequenz
    fs: int = 300
    # Lade references Datei
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            # Lade MatLab Datei mit EKG lead and label
            data = sio.loadmat(os.path.join(folder, row[0] + '.mat'))
            ecg_leads.append(data['val'][0])
            ecg_labels.append(row[1])
            ecg_names.append(row[0])
    # Zeige an wie viele Daten geladen wurden
    print("{}\t Dateien wurden geladen.".format(len(ecg_leads)))
    return ecg_leads, ecg_labels, fs, ecg_names


### Achtung! Diese Funktion nicht veraendern.

def save_predictions(predictions: List[Tuple[str, str, float]], folder: str = None) -> None:
    """
    Funktion speichert the gegebenen predictions in eine CSV-Datei mit dem name PREDICTIONS.csv
    Parameters
    ----------
    predictions : List[Tuple[str, str,float]]
        List aus Tuplen wobei jedes Tuple den Name der Datei und das vorhergesagte label ('N','A','O','~'), sowie die Unsicherheit enthaelt
        Beispiel [('train_ecg_03183.mat', 'N'), ('train_ecg_03184.mat', "~"), ('train_ecg_03185.mat', 'A'),
                  ('train_ecg_03186.mat', 'N'), ('train_ecg_03187.mat', 'O')]
	folder : str
		Speicherort der predictions
    Returns
    -------
    None.
    """
    # Check Parameter
    assert isinstance(predictions, list), \
        "Parameter predictions muss eine Liste sein aber {} gegeben.".format(type(predictions))
    assert len(predictions) > 0, 'Parameter predictions muss eine nicht leere Liste sein.'
    assert isinstance(predictions[0], tuple), \
        "Elemente der Liste predictions muss ein Tuple sein aber {} gegeben.".format(type(predictions[0]))
    assert isinstance(predictions[0][2], float), \
        "3. Element der Tupel in der Liste muss vom Typ float sein, aber {} gegeben".format(type(predictions[0][2]))

    if folder == None:
        file = "PREDICTIONS.csv"
    else:
        file = os.path.join(folder, "PREDICTIONS.csv")
    # Check ob Datei schon existiert wenn ja loesche Datei
    if os.path.exists(file):
        os.remove(file)

    with open(file, mode='w', newline='') as predictions_file:

        # Init CSV writer um Datei zu beschreiben
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        # Iteriere über jede prediction
        for prediction in predictions:
            predictions_writer.writerow([prediction[0], prediction[1], prediction[2]])
        # Gebe Info aus wie viele labels (predictions) gespeichert werden
        print("{}\t Labels wurden geschrieben.".format(len(predictions)))
