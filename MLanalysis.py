import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing  import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score,roc_curve

              #Carico i dati e li divido in training e test set
dati=load_breast_cancer()
#print(dati.feature_names)
X = dati.data
y = dati.target

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)


              #Riscalo/Rinormalizzo i dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


              #Addestramento Regressione Logistica con features elimination e con cambio di treshold
modello_lineare = LogisticRegression()
rfe=RFE(modello_lineare,n_features_to_select=19,step=1)
rfe.fit(X_train,y_train)
#print(rfe.support_)
X_train_logreg_red=rfe.transform(X_train)
X_test_logreg_red=rfe.transform(X_test)
modello_lineare.fit(X_train_logreg_red,y_train)

              #Addestramento SVMs
modello_svm=SVC()
param_grid = {
    "C":[0.1,1,10,100],
    "kernel": ["linear", "rbf"]
}
grid_search = GridSearchCV(estimator=modello_svm, param_grid=param_grid, scoring='accuracy')
grid_search.fit(X_train,y_train)
print("i migliori parametri trovati dalla grid search sono:",grid_search.best_params_)

# --- CALCOLO AUTOMATICO DELLA SOGLIA OTTIMALE (INDICE DI YOUDEN) ---
# Estraiamo le probabilità per la Regressione Logistica
prob_maligno = modello_lineare.predict_proba(X_test_logreg_red)[:, 1]

# Calcoliamo la curva ROC grezza per ottenere tutte le soglie possibili
fpr, tpr, thresholds = roc_curve(y_test, prob_maligno)

# Applichiamo l'indice di Youden (massimizza Recall - FPR)
j_scores = tpr - fpr
best_index = np.argmax(j_scores)
soglia_ottimale = thresholds[best_index]
print(f"--> Soglia ottimale calcolata matematicamente (Youden): {soglia_ottimale:.4f}")

# Applichiamo la soglia ottimale trovata per generare le predizioni binarie
y_pred_logreg = (prob_maligno >= soglia_ottimale).astype(int)

# Predizioni SVM (con soglia standard di default a 0.5)
y_pred_svm = grid_search.predict(X_test)


#--------------- CONFRONTO DEI DUE MODELLI --------------#

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2= axes[0,0], axes[0,1]
ax3, ax4= axes[1,0], axes[1,1]

                #Matrice di confusione Regressione Logistica (con riduzione features e cambio tresholds) e matrice di confusione SVMs

#Matrice Regressione Logistica
cm=confusion_matrix(y_test, y_pred_logreg)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=dati.target_names)
disp.plot(ax=ax1,cmap=plt.cm.Blues,colorbar=True)

#Matrice SVMs
ConfusionMatrixDisplay.from_estimator(
    grid_search,
    X_test,
    y_test,
    ax=ax2,
    cmap=plt.cm.Blues,
    colorbar=False
)

# Per la Regressione Logistica
RocCurveDisplay.from_estimator(
    modello_lineare,
    X_test_logreg_red,
    y_test,
    ax=ax3, # Il riquadro in basso a sinistra
    name="Logistic Regression"
)

# Per la SVM
RocCurveDisplay.from_estimator(
    grid_search,
    X_test,
    y_test,
    ax=ax4, # Il riquadro in basso a destra
    name="SVM"
)
ax1.set_title("Matrice con Soglia Personalizzata")
ax1.set_xlabel("Predicted Lables")
ax1.set_ylabel("Actual Lables")

ax2.set_title("Confusion Matrix SVM")
ax2.set_xlabel("Predicted Lables")
ax2.set_ylabel("Actual Lables")

ax3.set_title("ROC Curve - Logistic Regression")
ax4.set_title("ROC Curve - SVM")

plt.tight_layout()
plt.show()


#Calcolo metriche e confronto dei modelli

acc_logreg = accuracy_score(y_test, y_pred_logreg)
acc_svm = accuracy_score(y_test, y_pred_svm)

prec_logreg = precision_score(y_test, y_pred_logreg)
prec_svm = precision_score(y_test, y_pred_svm)

rec_logreg = recall_score(y_test, y_pred_logreg)
rec_svm = recall_score(y_test, y_pred_svm)

f1_logreg = f1_score(y_test, y_pred_logreg)
f1_svm = f1_score(y_test, y_pred_svm)

# Calcolo AUC numerico
# Per la logreg usiamo le probabilità predette
auc_logreg = roc_auc_score(y_test, prob_maligno)
# Per la SVM usiamo la distance function (decision_function) o le probabilità se abilitate
auc_svm = roc_auc_score(y_test, grid_search.decision_function(X_test))

#Stampo i risultati

print("=" * 60)
print(f"{'METRICA':<15} | {'REGRESSIONE LOGISTICA':<22} | {'SVM':<12}")
print("=" * 60)
print(f"{'Accuracy':<15} | {acc_logreg:<22.4f} | {acc_svm:<12.4f}")
print(f"{'Precision':<15} | {prec_logreg:<22.4f} | {prec_svm:<12.4f}")
print(f"{'Recall':<15} | {rec_logreg:<22.4f} | {rec_svm:<12.4f}")
print(f"{'F1-Score':<15} | {f1_logreg:<22.4f} | {f1_svm:<12.4f}")
print(f"{'AUC (Area)':<15} | {auc_logreg:<22.4f} | {auc_svm:<12.4f}")
print("=" * 60)

