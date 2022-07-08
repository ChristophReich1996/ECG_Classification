# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Maurice Rohr
"""

from predict import predict_labels
from wettbewerb import load_references, save_predictions
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('model_directory', action='store',type=str)
    parser.add_argument('data_directory', action='store',type=str)
    parser.add_argument('output_directory', action='store',type=str)
    args = parser.parse_args()
    
    ecg_leads,ecg_labels,fs,ecg_names = load_references(args.data_directory) # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
    
    start_time = time.time()
    predictions = predict_labels(ecg_leads,fs,ecg_names, is_binary_classifier=False, return_probability=True)
    pred_time = time.time()-start_time
    
    save_predictions(predictions,folder=args.output_directory) # speichert Prädiktion in CSV Datei
    print("Runtime",pred_time,"s")
