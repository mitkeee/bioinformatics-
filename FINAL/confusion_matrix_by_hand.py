
import math

def confusion_metrics(TP, FP, TN, FN):
    P = TP + FN
    N = TN + FP
    total = P + N

    BM = (TP / P) + (TN / N) - 1
    prevalence = P / (P + N)
    TPR = TP / P
    FNR = FN / P
    FPR = FP / N
    TNR = TN / N
    PPV = TP / (TP + FP)
    FOR = FN / (TN + FN)
    LR_plus = (TP / P) / (FP / N)
    LR_minus = (FN / P) / (TN / N)
    ACC = (TP + TN) / total
    FDR = FP / (TP + FP)
    NPV = TN / (TN + FN)
    MK = PPV + NPV - 1
    DOR = LR_plus / LR_minus
    BA = ((TP / P) + (TN / N)) / 2
    F1 = (2 * TP) / (2 * TP + FP + FN)
    FM = math.sqrt(PPV * TPR)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    Jaccard = TP / (TP + FP + FN)

    print("Informedness (BM):", BM)
    print("Prevalence:", prevalence)
    print("TPR:", TPR)
    print("FNR:", FNR)
    print("FPR:", FPR)
    print("TNR:", TNR)
    print("PPV:", PPV)
    print("FOR:", FOR)
    print("LR+:", LR_plus)
    print("LR-:", LR_minus)
    print("Accuracy:", ACC)
    print("FDR:", FDR)
    print("NPV:", NPV)
    print("Markedness (MK):", MK)
    print("DOR:", DOR)
    print("Balanced Accuracy:", BA)
    print("F1 Score:", F1)
    print("Fowlkes–Mallows:", FM)
    print("MCC:", MCC)
    print("Jaccard Index:", Jaccard)

confusion_metrics(50, 10, 30, 5)
