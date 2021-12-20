import os
import sys
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def score(label_dir='../test/', output_dir='./'):
    if not os.path.exists(os.path.join(output_dir, "PREDICTIONS.csv")):
        sys.exit("Es gibt keine Predictions")

    if not os.path.exists(os.path.join(label_dir, "REFERENCE.csv")):
        sys.exit("Es gibt keine Ground Truth")

    df_pred = pd.read_csv(os.path.join(output_dir, "PREDICTIONS.csv"), header=None)  # Klassifikationen
    df_gt = pd.read_csv(os.path.join(label_dir, "REFERENCE.csv"), header=None)  # Wahrheit

    N_files = df_gt.shape[0]  # Anzahl an Datenpunkten

    ## für normalen F1-Score
    TP = 0  # Richtig Positive
    TN = 0  # Richtig Negative
    FP = 0  # Falsch Positive
    FN = 0  # Falsch Negative

    ## für Multi-Class-F1
    Nn = 0  # Wahrheit ist normal, klassifiziert als normal
    Na = 0  # Wahrheit ist normal, klassifiziert als Vorhofflimmern
    No = 0  # Wahrheit ist normal, klassifiziert als anderer Rhythmus
    Np = 0  # Wahrheit ist normal, klassifiziert als unbrauchbar
    An = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als normal
    Aa = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als Vorhofflimmern
    Ao = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als anderer Rhythmus
    Ap = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als unbrauchbar
    On = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als normal
    Oa = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als Vorhofflimmern
    Oo = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als anderer Rhythmus
    Op = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als unbrauchbar
    Pn = 0  # Wahrheit ist unbrauchbar, klassifiziert als normal
    Pa = 0  # Wahrheit ist unbrauchbar, klassifiziert als Vorhofflimmern
    Po = 0  # Wahrheit ist unbrauchbar, klassifiziert als anderer Rhythmus
    Pp = 0  # Wahrheit ist unbrauchbar, klassifiziert als unbrauchbar

    y_true = list()
    y_score = list()

    for i in range(N_files):
        gt_name = df_gt[0][i]
        gt_class = df_gt[1][i]

        pred_indx = df_pred[df_pred[0] == gt_name].index.values

        if not pred_indx.size:
            print("Prediktion für " + gt_name + " fehlt, nehme \"normal\" an.")
            pred_class = "N"
            pred_certainty = 0.5
        else:
            pred_indx = pred_indx[0]
            pred_class = df_pred[1][pred_indx]
            pred_certainty = df_pred[2][pred_indx]

        if gt_class == "A" and pred_class == "A":
            TP = TP + 1
            y_score.append(pred_certainty)
            y_true.append(1)
        if gt_class == "N" and pred_class != "A":
            TN = TN + 1
            y_score.append(pred_certainty)
            y_true.append(0)
        if gt_class == "N" and pred_class == "A":
            FP = FP + 1
            y_score.append(pred_certainty)
            y_true.append(0)
        if gt_class == "A" and pred_class != "A":
            FN = FN + 1
            y_score.append(pred_certainty)
            y_true.append(1)

        if gt_class == "N":
            if pred_class == "N":
                Nn = Nn + 1
            if pred_class == "A":
                Na = Na + 1
            if pred_class == "O":
                No = No + 1
            if pred_class == "~":
                Np = Np + 1

        if gt_class == "A":
            if pred_class == "N":
                An = An + 1
            if pred_class == "A":
                Aa = Aa + 1
            if pred_class == "O":
                Ao = Ao + 1
            if pred_class == "~":
                Ap = Ap + 1

        if gt_class == "O":
            if pred_class == "N":
                On = On + 1
            if pred_class == "A":
                Oa = Oa + 1
            if pred_class == "O":
                Oo = Oo + 1
            if pred_class == "~":
                Op = Op + 1

        if gt_class == "~":
            if pred_class == "N":
                Pn = Pn + 1
            if pred_class == "A":
                Pa = Pa + 1
            if pred_class == "O":
                Po = Po + 1
            if pred_class == "~":
                Pp = Pp + 1

    sum_N = Nn + Na + No + Np
    sum_A = An + Aa + Ao + Ap
    sum_O = On + Oa + Oo + Op
    sum_P = Pn + Pa + Po + Pp

    sum_n = Nn + An + On + Pn
    sum_a = Na + Aa + Oa + Pa
    sum_o = No + Ao + Oo + Po
    sum_p = Np + Ap + Op + Pp

    F1 = TP / (TP + 1 / 2 * (FP + FN))

    # Confusion Matrix zur Evaluation
    Conf_Matrix = {'N': {'n': Nn, 'a': Na, 'o': No, 'p': Np},
                   'A': {'n': An, 'a': Aa, 'o': Ao, 'p': Ap},
                   'O': {'n': On, 'a': Oa, 'o': Oo, 'p': Op},
                   'P': {'n': Pn, 'a': Pa, 'o': Po, 'p': Pp}}

    F1_mult = 0
    n_f1_mult = 0

    if (sum_N + sum_n) != 0:
        F1_mult += 2 * Nn / (sum_N + sum_n)
        n_f1_mult += 1

    if (sum_A + sum_a) != 0:
        F1_mult += 2 * Aa / (sum_A + sum_a)
        n_f1_mult += 1

    if (sum_O + sum_o) != 0:
        F1_mult += 2 * Oo / (sum_O + sum_o)
        n_f1_mult += 1

    if (sum_P + sum_p) != 0:
        F1_mult += 2 * Pp / (sum_P + sum_p)
        n_f1_mult += 1

    F1_mult = F1_mult / n_f1_mult

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    AUROC = roc_auc_score(y_true, y_score)
    AUPRC = average_precision_score(y_true, y_score)
    accuracy = (TP + TN) / (TP + TN + FN + FP)

    return F1, F1_mult, Conf_Matrix, AUROC, AUPRC, accuracy


#
def score_official_physionet(label_dir='../test/', output_dir='./'):
    if not os.path.exists(os.path.join(output_dir, "PREDICTIONS.csv")):
        sys.exit("Es gibt keine Predictions")

    if not os.path.exists(os.path.join(label_dir, "REFERENCE.csv")):
        sys.exit("Es gibt keine Ground Truth")

    df_pred = pd.read_csv(os.path.join(output_dir, "PREDICTIONS.csv"), header=None)  # Klassifikationen
    df_gt = pd.read_csv(os.path.join(label_dir, "REFERENCE.csv"), header=None)  # Wahrheit

    N_files = df_gt.shape[0]  # Anzahl an Datenpunkten

    ## für normalen F1-Score
    TP = 0  # Richtig Positive
    TN = 0  # Richtig Negative
    FP = 0  # Falsch Positive
    FN = 0  # Falsch Negative

    ## für Multi-Class-F1
    Nn = 0  # Wahrheit ist normal, klassifiziert als normal
    Na = 0  # Wahrheit ist normal, klassifiziert als Vorhofflimmern
    No = 0  # Wahrheit ist normal, klassifiziert als anderer Rhythmus
    Np = 0  # Wahrheit ist normal, klassifiziert als unbrauchbar
    An = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als normal
    Aa = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als Vorhofflimmern
    Ao = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als anderer Rhythmus
    Ap = 0  # Wahrheit ist Vorhofflimmern, klassifiziert als unbrauchbar
    On = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als normal
    Oa = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als Vorhofflimmern
    Oo = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als anderer Rhythmus
    Op = 0  # Wahrheit ist anderer Rhythmus, klassifiziert als unbrauchbar
    Pn = 0  # Wahrheit ist unbrauchbar, klassifiziert als normal
    Pa = 0  # Wahrheit ist unbrauchbar, klassifiziert als Vorhofflimmern
    Po = 0  # Wahrheit ist unbrauchbar, klassifiziert als anderer Rhythmus
    Pp = 0  # Wahrheit ist unbrauchbar, klassifiziert als unbrauchbar

    y_true = list()
    y_score = list()

    y_true_multi = np.zeros((N_files, 4))
    y_score_multi = np.zeros((N_files, 4))
    classes = {'N': 0, 'A': 1, 'O': 2, '~': 3}

    for i in range(N_files):
        gt_name = df_gt[0][i]
        gt_class = df_gt[1][i]

        pred_indx = df_pred[df_pred[0] == gt_name].index.values

        if not pred_indx.size:
            print("Prediktion für " + gt_name + " fehlt, nehme \"normal\" an.")
            pred_class = "N"
            pred_certainty = np.ones((4)) * 0.25
        else:
            pred_indx = pred_indx[0]
            pred_class = df_pred[1][pred_indx]
            pred_certainty = np.zeros((4))
            for i in range(4):
                pred_certainty[i] = df_pred[i + 2][pred_indx]
        y_true_multi[i, classes[gt_class]] = 1
        y_score_multi[i, :] = pred_certainty

        if gt_class == "A" and pred_class == "A":
            TP = TP + 1
            y_score.append(pred_certainty[1])
            y_true.append(1)
        if gt_class == "N" and pred_class != "A":
            TN = TN + 1
            y_score.append(pred_certainty[1])
            y_true.append(0)
        if gt_class == "N" and pred_class == "A":
            FP = FP + 1
            y_score.append(pred_certainty[1])
            y_true.append(0)
        if gt_class == "A" and pred_class != "A":
            FN = FN + 1
            y_score.append(pred_certainty[1])
            y_true.append(1)

        if gt_class == "N":
            if pred_class == "N":
                Nn = Nn + 1
            if pred_class == "A":
                Na = Na + 1
            if pred_class == "O":
                No = No + 1
            if pred_class == "~":
                Np = Np + 1

        if gt_class == "A":
            if pred_class == "N":
                An = An + 1
            if pred_class == "A":
                Aa = Aa + 1
            if pred_class == "O":
                Ao = Ao + 1
            if pred_class == "~":
                Ap = Ap + 1

        if gt_class == "O":
            if pred_class == "N":
                On = On + 1
            if pred_class == "A":
                Oa = Oa + 1
            if pred_class == "O":
                Oo = Oo + 1
            if pred_class == "~":
                Op = Op + 1

        if gt_class == "~":
            if pred_class == "N":
                Pn = Pn + 1
            if pred_class == "A":
                Pa = Pa + 1
            if pred_class == "O":
                Po = Po + 1
            if pred_class == "~":
                Pp = Pp + 1

    sum_N = Nn + Na + No + Np
    sum_A = An + Aa + Ao + Ap
    sum_O = On + Oa + Oo + Op
    sum_P = Pn + Pa + Po + Pp

    sum_n = Nn + An + On + Pn
    sum_a = Na + Aa + Oa + Pa
    sum_o = No + Ao + Oo + Po
    sum_p = Np + Ap + Op + Pp

    F1 = TP / (TP + 1 / 2 * (FP + FN))

    # Confusion Matrix zur Evaluation
    Conf_Matrix = {'N': {'n': Nn, 'a': Na, 'o': No, 'p': Np},
                   'A': {'n': An, 'a': Aa, 'o': Ao, 'p': Ap},
                   'O': {'n': On, 'a': Oa, 'o': Oo, 'p': Op},
                   'P': {'n': Pn, 'a': Pa, 'o': Po, 'p': Pp}}

    F1_mult = 0
    n_f1_mult = 0

    if (sum_N + sum_n) != 0:
        F1_mult += 2 * Nn / (sum_N + sum_n)
        n_f1_mult += 1

    if (sum_A + sum_a) != 0:
        F1_mult += 2 * Aa / (sum_A + sum_a)
        n_f1_mult += 1

    if (sum_O + sum_o) != 0:
        F1_mult += 2 * Oo / (sum_O + sum_o)
        n_f1_mult += 1

    if (sum_P + sum_p) != 0:
        F1_mult += 2 * Pp / (sum_P + sum_p)
        n_f1_mult += 1

    F1_mult = F1_mult / n_f1_mult

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    AUROC = roc_auc_score(y_true, y_score)
    AUPRC = average_precision_score(y_true, y_score)
    AUROC_macro = roc_auc_score(y_true_multi, y_score_multi, multi_class='ovr')
    AUPRC_macro = average_precision_score(y_true_multi, y_score_multi)
    accuracy_multi = (Nn + Aa + Oo + Pp) / N_files

    accuracy = (TP + TN) / (TP + TN + FN + FP)

    return F1, F1_mult, Conf_Matrix, AUROC, AUPRC, accuracy, AUPRC_macro, AUROC_macro, accuracy_multi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('label_directory', action='store', type=str)
    parser.add_argument('output_directory', action='store', type=str)
    args = parser.parse_args()
    F1, F1_mult, Conf_Matrix, AUROC, AUPRC, accuracy, AUPRC_macro, AUROC_macro, accuracy_multi = score_official_physionet(
        args.label_directory, args.output_directory)
    print("F1:", F1, "\t AUROC:", AUROC, "\t AUPRC:", AUPRC, "\t Accuracy:", accuracy, "\n Physionet2017 Score:",
          F1_mult, "\t AUROC macro:", AUROC_macro, "\t AUPRC macro:", AUPRC_macro, "\t Accuracy multi:", accuracy_multi)
