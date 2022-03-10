from sklearn.metrics import  accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest


class RandomForest:
    def __init__(self, dataX, dataY, k_fold, tree):
        self.dataX = dataX
        self.dataY = dataY
        self.k_fold = k_fold
        self.tree = tree

    def get_result(self):
        # skala
        sc = MinMaxScaler(feature_range=(0, 1))
        self.dataX = sc.fit_transform(self.dataX)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(self.dataX, self.dataY, test_size=0.15, shuffle=True, random_state=42)

        #Outlier
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        yhat = iso.fit_predict(X_train)
        mask = yhat != -1
        X_train, y_train = X_train[mask, :], y_train[mask]

        #from collections import Counter
        # counter1 = Counter(y_train)
        #print("Data setelah Outlier : ",counter1)

        accuracy_train = []
        f1Score_train = []

        accuracy_test = []
        f1Score_test = []


        # Membagi Data dengan K-fold
        kf = KFold(n_splits=self.k_fold)

        for index_train, index_validasi in kf.split(X_train):
            x_training, x_validasi = X_train[index_train], X_train[index_validasi]
            y_training, y_validasi = y_train[index_train], y_train[index_validasi]

        # Membangun Model
            model = RandomForestClassifier(n_estimators=self.tree, criterion='entropy', bootstrap=True, random_state=42)
            model.fit(x_training, y_training)

        # Melakukan Prediksi
            predicted_labels_train = model.predict(x_validasi)
            predicted_labels_test = model.predict(X_test)

        #Menyimpan nilai hasil f1-score dan akurasi  ke dalam list
            accuracy_train.append(accuracy_score(y_validasi, predicted_labels_train) * 100)
            f1Score_train.append(f1_score(y_validasi, predicted_labels_train, average='binary') * 100)
            accuracy_test.append(accuracy_score(y_test, predicted_labels_test) * 100)
            f1Score_test.append(f1_score(y_test, predicted_labels_test, average='binary') * 100)



        return accuracy_train, accuracy_test, f1Score_train, f1Score_test