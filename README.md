# Breast Cancer Classification & Model Evaluation Pipeline

Progetto di Machine Learning in Python per la classificazione binaria di tumori (benigni vs maligni) basato sul dataset *Breast Cancer Wisconsin*. Il progetto confronta due algoritmi avanzati, focalizzandosi non solo sulle performance pure, ma sulla gestione rigorosa delle soglie decisionali e sulla valutazione geometrico-probabilistica.

## Caratteristiche Tecniche
- Pre-processing & Scaling: Standardizzazione delle feature tramite `StandardScaler`.
- Feature Selection: Implementazione della *Recursive Feature Elimination (RFE)* sulla Regressione Logistica.
- Ottimizzazione degli Iperparametri: Utilizzo di `GridSearchCV` per la SVM (Kernel Lineare e RBF).
- Soglia Dinamica (Indice di Youden): Calcolo matematico della soglia di classificazione ottimale tramite la Curva ROC, superando il limite arbitrario di 0.5.
- Visualizzazione Avanzata: Dashboard grafica 2x2 integrata con Matrici di Confusione e Curve ROC.

## Tech Stack
- Linguaggio: Python
- Librerie: Scikit-Learn, NumPy, Matplotlib

## Risultati Principali
I modelli hanno raggiunto performance eccellenti con un'**AUC > 0.99**, evidenziando l'efficacia dell'ottimizzazione tramite GridSearchCV e della sintonizzazione della soglia di decisione per minimizzare i falsi negativi in ambito medico.