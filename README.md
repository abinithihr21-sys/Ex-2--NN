<H3>Name: ABINITHI H</H3>
<H3>Register no:212225240003</H3>
<H3>Date: 18.11.26</H3>
<H3>Experiment No. 2 </H3>
## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# STEP 1: IMPORTING THE DATASET

df = pd.read_csv("synthetic_diabetes_5000.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# STEP 2: DATA PREPROCESSING

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Replace invalid zero values with NaN
cols = ['Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI']

df[cols] = df[cols].replace(0, np.nan)

# Fill missing values with median
for col in cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

print("\nDataset Shape After Preprocessing:")
print(df.shape)


# STEP 3: DATA VISUALIZATION

# Class distribution
df['Outcome'].value_counts().plot(
    kind='bar',
    title='Diabetes Class Distribution'
)

plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')
plt.ylabel('Count')
plt.show()

# Scatter plot
plt.scatter(
    df[df['Outcome'] == 0]['Glucose'],
    df[df['Outcome'] == 0]['BMI'],
    label='Non-Diabetic'
)

plt.scatter(
    df[df['Outcome'] == 1]['Glucose'],
    df[df['Outcome'] == 1]['BMI'],
    label='Diabetic'
)

plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('Diabetes Prediction')
plt.legend()
plt.show()


# STEP 4: FEATURE SELECTION

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

print("\nInput Features:")
print(X[:5])

print("\nTarget Output:")
print(y[:5])


# STEP 5: SPLIT THE DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# STEP 6: FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaled Training Data:")
print(X_train[:5])


# STEP 7: DEFINE INPUT VECTOR X

# Add bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

print("\nInput Vector X:")
print(X_train[:5])


# STEP 8: DEFINE OUTPUT VECTOR Y AS +1 OR -1

# Diabetic = +1
# Non-Diabetic = -1

Y_train = np.where(y_train == 1, 1, -1)
Y_test = np.where(y_test == 1, 1, -1)

print("\nDesired Output Y:")
print(Y_train[:10])


# STEP 9: INITIALIZE WEIGHTS AND LEARNING RATE

W = np.zeros(X_train.shape[1])

learning_rate = 0.01
epochs = 100

print("\nInitial Weight Vector:")
print(W)

print("Learning Rate:", learning_rate)


# STEP 10: TRAIN THE PERCEPTRON

errors = []

for epoch in range(epochs):

    total_error = 0

    for i in range(len(X_train)):

        v = np.dot(W, X_train[i])

        T = 1 if v >= 0 else -1

        if Y_train[i] != T:

            W = W + learning_rate * (Y_train[i] - T) * X_train[i]

            total_error += 1

    errors.append(total_error)

print("\nFinal Weight Vector:")
print(W)


# STEP 11: PLOT TRAINING ERRORS

plt.plot(range(1, epochs + 1), errors, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Training Error')

plt.grid()
plt.show()


# STEP 12: TEST THE MODEL

y_pred = []

for i in range(len(X_test)):

    v = np.dot(W, X_test[i])

    T = 1 if v >= 0 else -1

    y_pred.append(T)

y_pred = np.array(y_pred)

accuracy = accuracy_score(Y_test, y_pred)

print("\nActual Output:")
print(Y_test[:20])

print("\nPredicted Output:")
print(y_pred[:20])

print("\nAccuracy:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    Y_test,
    y_pred,
    labels=[-1, 1],
    target_names=['Non-Diabetic', 'Diabetic'],
    zero_division=0
))


# STEP 13: TEST WITH NEW PATIENT DATA

print("\nNEW PATIENT PREDICTION")

# Feature order:
# Pregnancies, Glucose, BloodPressure,
# SkinThickness, Insulin, BMI,
# DiabetesPedigreeFunction, Age

new_patients = np.array([
    [2, 150, 80, 25, 120, 32.5, 0.5, 35],
    [1, 85, 65, 20, 50, 22.0, 0.2, 24],
    [5, 180, 90, 35, 200, 38.5, 0.8, 50],
    [0, 95, 70, 18, 60, 25.0, 0.3, 28]
])

# Apply the same feature scaling
new_patients_scaled = scaler.transform(new_patients)

# Add bias
new_patients_scaled = np.c_[
    np.ones(new_patients_scaled.shape[0]),
    new_patients_scaled
]

# Predict new patients
for i in range(len(new_patients_scaled)):

    v = np.dot(W, new_patients_scaled[i])

    T = 1 if v >= 0 else -1

    if T == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    print(f"Patient {i+1}: {result}")


print("\nPROGRAM EXECUTION COMPLETED")


    

# OUTPUT:

<img width="837" height="665" alt="image" src="https://github.com/user-attachments/assets/47d82139-282e-41e7-a4bc-0aa7cf426832" />
<img width="595" height="582" alt="image" src="https://github.com/user-attachments/assets/9614f09b-4620-46e6-aa80-41ceb1b0d1b8" />
<img width="666" height="772" alt="image" src="https://github.com/user-attachments/assets/d738d75a-5450-4724-9440-57978a46d6a7" />
<img width="797" height="207" alt="image" src="https://github.com/user-attachments/assets/541c909e-350a-44ea-9ef1-538cec39c838" />
<img width="360" height="60" alt="image" src="https://github.com/user-attachments/assets/3a5b6fcd-d0a4-4870-9202-55670989b30b" />
<img width="832" height="217" alt="image" src="https://github.com/user-attachments/assets/ddcbe6b0-406a-4d6b-99df-24a382855c9a" />
<img width="835" height="226" alt="image" src="https://github.com/user-attachments/assets/42ae4ec6-b8a4-4782-ba80-f156ca994667" />
<img width="447" height="56" alt="image" src="https://github.com/user-attachments/assets/95838a91-8426-418b-9418-f4cb63c964ec" />
<img width="342" height="90" alt="image" src="https://github.com/user-attachments/assets/f10840a6-7f9a-4e16-bee8-bdfb33895ff2" />
<img width="811" height="72" alt="image" src="https://github.com/user-attachments/assets/a1b8ea6e-1133-4631-bfc1-5451bf35813a" />
<img width="836" height="480" alt="image" src="https://github.com/user-attachments/assets/231cdcfc-56a4-45b8-98ca-5f9b556aa15c" />
<img width="787" height="490" alt="image" src="https://github.com/user-attachments/assets/0057557a-9f01-496c-871c-0105926acd8b" />
<img width="271" height="42" alt="image" src="https://github.com/user-attachments/assets/bbc0cb43-015e-47a6-ae17-fd94f68328f1" />













# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
