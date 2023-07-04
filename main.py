import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Read Files
gb4 = pd.read_csv("DriverDbGb4/WRC-drivers-attributes.csv")
f1 = pd.read_csv("KaggleFormula1(1950-2020)/drivers-with-attributes.csv")
fe = pd.read_csv("WikipediaFormulaE/Drivers-with-attributes.csv")
wrc = pd.read_csv("WRC2022/drivers-with-attributes.csv")
nascar = pd.read_csv("KaggleNascarAdvancedStats2022/nascar-with-attributes.csv")
from sklearn.model_selection import train_test_split

# Add Colums with value for International Car Racing Sports organization
# and Local and international Car Racing federations
gb4["ICRS"] = 'BRDC'
gb4["CRF"] = 'MotorSport Vision'

f1["ICRS"] = 'FIA'
f1["CRF"] = 'Formula One'

nascar["ICRS"] = 'ACCUS-FIA'
nascar["CRF"] = 'Nascar'

fe["ICRS"] = 'FIA'
fe["CRF"] = 'Formula E'

wrc["ICRS"] = 'FIA'
wrc["CRF"] = 'World Rally Championship'

# merge
merged = pd.concat([gb4, f1, fe, wrc, nascar]).reset_index(drop=True)
merged.describe()

# Iterate through the dataset
for index, row in merged.iterrows():
    if row['FinalPosition'] < 4:
        merged.at[index, 'IsTopThree'] = 1
    else:
        merged.at[index, 'IsTopThree'] = 0

# Split the dataset into features (X) and target variable (y)
# Dropping names as that should not affect the prediction
X = merged.drop(['FinalPosition', 'Name', "ICRS", "CRF", "IsTopThree"], axis=1)
y = merged['IsTopThree']


def run_classic_models(X, y, label):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # Scale the features for better performance
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Initialize and train models
    models = [
        ("k-Nearest Neighbors", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("Logistic Regression", LogisticRegression()),
        ("Support Vector Classifier", SVC(kernel='rbf', C=1.0, random_state=42)),
        ("Naive Bayes", GaussianNB())
    ]
    plt.figure()
    for name, model in models:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)  # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Precision: {precision:.4f}")
        print(f"{name} Recall: {recall:.4f}")
        print(f"{name} F1-Score: {f1:.4f}")
        print()
        # zero_division because logstic regression seems to always predict false for IsTopThree
        print(classification_report(y_test, y_pred, zero_division=1))
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)

        # Plot ROC Curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    if not os.path.exists('figures/' + label):
        os.makedirs('figures/' + label)
    plt.savefig(os.path.join('figures/' + label, label + "roc.png"))
    plt.show()
    # Plot Confusion Matrix
    plt.figure(figsize=(12, 8))
    for name, model in models:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"{name} - Confusion Matrix")
        if not os.path.exists('figures/' + label):
            os.makedirs('figures/' + label)
        plt.savefig(os.path.join('figures/' + label, label + name + "confusion.png"))
        plt.show()
    for model_name, model in models:
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        model.fit(X_train_pca, y_train)
        # Define the step size of the mesh
        h = 0.02

        # Get the minimum and maximum values of the transformed features
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

        # Create a meshgrid of points with the defined step size
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k')
        plt.title(f"Decision Boundaries - {model_name}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        if not os.path.exists('figures/' + label):
            os.makedirs('figures/' + label)
        plt.savefig(os.path.join('figures/' + label, label + model_name + "boundary.png"))
        plt.show()


run_classic_models(X, y, "before_prune")

# Reduce features to improve performance
print(X.shape)
# ExtraTreeClassifier relies on stochastic methods and might produce different results with each run without seeding
numpy.random.seed(42)
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
for feat, importance in zip(X.columns, clf.feature_importances_):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))
print(X_new)

run_classic_models(X_new, y, "after_prune")

print(X_new.shape)
print("X_new")
print(X_new.shape)
# Reshape the input data
x_shaped = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(x_shaped, y, test_size=0.4, random_state=42)
print("X_train")
print(X_train.shape)
# Reshape the input data


from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D

# Define the model
model = Sequential()

# Add a 1D convolutional layer
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(x_shaped.shape[1], 1)))

# Add max pooling layer
# CNN sometimes fails, probably due to issues with reshaping after dropping features, a rerun usually fixes it
model.add(MaxPooling1D(pool_size=2))

# Add LSTM layer
model.add(LSTM(units=64, activation='relu'))

# Add output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Predict class probabilities using the trained model
y_prob = model.predict(X_test)

# Extract probabilities for the positive class
y_prob_positive = y_prob[:, 0]

# Extract predicted classes
y_pred = np.argmax(y_prob, axis=1)

# Calculate the false positive rate (FPR), true positive rate (TPR), and threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_prob_positive)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Plot the ROC curve
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Plot the random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Plot the confusion matrix
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.tight_layout()

if not os.path.exists('figures/cnn'):
    os.makedirs('figures/cnn')
plt.savefig(os.path.join('figures/cnn', "CNN-LSTM-roc-confusion.png"))
plt.show()
