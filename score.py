import os
import sys
import pandas as pd


def score():
    if not os.path.exists("PREDICTIONS.csv"):
        sys.exit("Es gibt keine Predictions")

    if not os.path.exists("../test/REFERENCE.csv"):
        sys.exit("Es gibt keine Ground Truth")

    df_pred = pd.read_csv("PREDICTIONS.csv", header=None)  # Klassifikationen
    df_gt = pd.read_csv("../test/REFERENCE.csv", header=None)  # Wahrheit

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

    for i in range(N_files):
        gt_name = df_gt[0][i]
        gt_class = df_gt[1][i]

        pred_indx = df_pred[df_pred[0] == gt_name].index.values

        if not pred_indx.size:
            print("Prediktion für " + gt_name + " fehlt, nehme \"normal\" an.")
            pred_class = "N"
        else:
            pred_indx = pred_indx[0]
            pred_class = df_pred[1][pred_indx]

        if gt_class == "A" and pred_class == "A":
            TP = TP + 1
        if gt_class == "N" and pred_class == "N":
            TN = TN + 1
        if gt_class == "N" and pred_class == "A":
            FP = FP + 1
        if gt_class == "A" and pred_class == "N":
            FN = FN + 1

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

    return F1, F1_mult


if __name__ == '__main__':
    F1, F1_mult = score()
    print("F1:", F1, "\t MultilabelScore:", F1_mult)
