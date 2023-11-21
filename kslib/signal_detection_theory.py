def confusion_matrix(sigs,dets):
    """_summary_

    Args:
        sigs (_type_): _description_
        dets (_type_): _description_

    1.
    if sigs==dets:
        decision = True
        label = TP if dets==p else TN
    else:
        decision = False
        label = FP if dets==p else FN

    Inputs     decision, label
    |sigs|dets|T/F|
    |--- |--- |---|
    | p  | p  | t | <- TP
    | p  | n  | f | <- FN
    | n  | p  | f | <- FP
    | n  | n  | t | <- TN
    sample size==N

    Confusion Matrix
    |dets\sig|p  |n  |
    |---     |---|---|
    |p       |TP |FP |
    |n       |FN |TN |

    2.
    - Accuracy(正解率) := count(decision==True)/N
    - Recall(TPR,検出率,再現率,Sensitivity,感度) := count(decision==True)/count(sig==positive)
    - Specificity(特異度) := count(decision==True)/count(sigs==negative)
    - Precision(精度,適合率) := count(decision==True)/count(dets==positive)
    - FPR(誤検出率,False Alarm) := count(decision==False)/count(sigs==negative)
    - FNR(未検出率,Miss Rate) := count(decision==False)/count(sigs==positive)
    - F1 := harmonic_mean(Precsion,Recall)

    3. curves
    ROC(Receiver Operative Characteristic)-curve := count(decision==True) vs count(decision==False) on count(dets==positive)
    PR(Precision)-curve := Precision vs Recall
    DET(Detection-Error-Tradeoff)-curve := FNR vs FPR

    4. AUC(area under curve)
    ROC-AUC
    PR-AUC

    5. EER(Equal Error Rate)
    DET-EER

    """
