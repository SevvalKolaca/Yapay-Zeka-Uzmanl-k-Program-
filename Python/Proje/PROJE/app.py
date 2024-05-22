import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class App:

    def __init__(self):
        self.preprocessed_df = None
        self.classifier_name = None
        self.params = dict()
        self.classifier = None
        self.X, self.y = None, None
        self.data_loader = False
        self.Init_Streamlit_Page()

    def Init_Streamlit_Page(self):
        # Görseli ekleyin
        st.image("title_image.png", use_column_width=True)

        st.write("""
        ### "Farklı Sınıflandırıcıları Keşfedin... En İyisi Hangisi?"
        """)
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")

    def run(self):
        # st.sidebar.title("Seçim Ekranı")
        try:
            st.sidebar.write(" ## Veri Seti")
            data_file = st.sidebar.file_uploader("Lütfen Bir Veri Seti Dosyası Seçiniz ↴", type=["csv"])

            if data_file is not None:
                df = self.load_data(data_file)

                st.write(f"##### Seçilen Veri Seti '{data_file.name}' ")
                st.write(f"Veri setinin ilk 10 satırı: ")
                self.show_data(df)

                st.image("mor.png", use_column_width=True)

                st.sidebar.write("## ────────────────────")
                st.sidebar.write(" ##  Normalizasyon & Sınıflandırıcı")

                self.preprocessed_df = self.preprocess_data(df)

                self.classifier_name = st.sidebar.selectbox(
                    'Sınıflandırıcı Seçiniz ↴',
                    ('KNN', 'SVM', 'Naive Bayes')
                )

                self.generate(self.preprocessed_df)
        except Exception as hata_mesaji:
            st.sidebar.error("Dosya yüklenirken bir hata oluştu: {}".format(hata_mesaji))

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def show_data(self, df):
        st.dataframe(df.head(10))  # İlk 10 satırı tablo olarak göster
        """st.write("## Sütunlar")
        columns = [f"{i}: {col}" for i, col in zip(range(len(df.columns)), df.columns)]
        st.write(columns)"""

    def preprocess_data(self, df):
        # DROP COLUMNS & LABEL ENCODER
        dropped_columns = ['id', 'Unnamed: 32']  # Eklenen sütunları saklamak için bir listeye atıyoruz
        df = df.drop(columns=['id', 'Unnamed: 32'], axis=1)
        # M -->, B --> 0
        # label_encoder = LabelEncoder()
        # df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

        # Lambda fonksiyonu kullanmak için label encoder'a alternatif
        df['diagnosis'] = df['diagnosis'].map(lambda x: 1 if x == 'M' else 0)

        # Etiket ve öznitelik sütunlarını ayir
        labels = df['diagnosis']
        features = df.drop(columns=['diagnosis'], axis=1)

        # Korelasyon matrisini hesapla
        korelasyon_matrisi = df.corr()
        # Korelasyon matrisinde 'diagnosis' ile negatif korelasyona sahip sütunları bul ve drop yap
        negatif_korelasyon_sutunlar = korelasyon_matrisi.loc[korelasyon_matrisi['diagnosis'] < 0, :].index
        features = features.drop(columns=negatif_korelasyon_sutunlar)

        # Sütun adlarını virgülle ayrılmış bir dizeye dönüştürme
        cikarilan_sutunlar_listesi = ', '.join(negatif_korelasyon_sutunlar)
        # diagnosis sütunu etiket verisine çevirdik
        # df = df.set_index('diagnosis')

        self.plot_scatter(df)
        st.image("mor.png", use_column_width=True)

        # NaN değerleri ortalama ile doldur
        features = features.fillna(df.mean())

        normalization_type = st.sidebar.selectbox("Normalizasyon Tipini Seçiniz ↴ ", ["MinMaxScaler", "StandardScaler"])
        if normalization_type == "MinMaxScaler":
            # MinMaxScaler seçildiğinde yapılacak işlemler
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
        elif normalization_type == "StandardScaler":
            # StandardScaler seçildiğinde yapılacak işlemler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

        # Normalleştirilmiş öznitelikler DataFrame
        scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

        # Etiketler DataFrame
        labels_df = pd.DataFrame(labels, columns=['diagnosis'])

        # Preprocessed Dataframe
        self.preprocessed_df = pd.concat([labels_df, scaled_features_df], axis=1)

        st.write("##### Yapılan Ön İşleme Adımları""")
        st.write(f"- Değersiz olan '{dropped_columns[0]}' ve tüm değerleri NaN olan '{dropped_columns[1]}'"
                 f" sütunları drop edildi.")
        st.write("- Diagnosis sütunundaki değerler M : 1 , B : 0 olacak şekilde güncellendi.")
        # st.write("- Korelasyon matriksi oluşturuldu ve 'Diagnosis' sütununa göre korelasyon değeri"
        #         " negatif olan sütunlar veri setinden çıkartıldı."
        #         f" Çıkarılan sütunlar →  '{cikarilan_sutunlar_listesi}'.")
        st.write("- Sütunlarda bulunan NaN değerler, sütun ortalamaları ile dolduruldu.")
        st.write("- Seçilmiş olan normalizasyon işlemine göre, normalizasyon yapıldı.")

        st.write(f" Ön işleme adımlarından sonra veri setinin son 10 satırı "
                 f"(1 : Melignant, 0 : Benign):")
        st.dataframe(self.preprocessed_df.tail(10))
        return self.preprocessed_df

    def split_data(self, df):

        self.y = df['diagnosis']
        self.X = df.drop(columns=['diagnosis'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            shuffle=True)
        return X_train, X_test, y_train, y_test

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            self.classifier = SVC()
            self.params = {'C': [2, 4, 6, 8, 10]}
        elif self.classifier_name == 'KNN':
            self.classifier = KNeighborsClassifier()
            self.params = {'n_neighbors': [1, 2, 4, 6, 8, 10]}
        elif self.classifier_name == 'Naive Bayes':
            self.classifier = GaussianNB()
            self.params = {}

    def generate(self, df):
        self.get_classifier()

        #### CLASSIFICATION ####
        X_train, X_test, y_train, y_test = self.split_data(df)

        ## Gridsearch
        grid_search = GridSearchCV(self.classifier, self.params)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        if self.classifier_name == 'Naive Bayes':
            best_params_text = " Naive Bayes sınıflandırıcısı parametre kullanmamaktadır"
        else:
            best_params_text = best_params

        self.classifier.set_params(**best_params)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        # MODEL PERFORMANSI
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y_test, y_pred)*100
        # precision tp / (tp + fp)
        precision = precision_score(y_test, y_pred)*100
        # recall: tp / (tp + fn)
        recall = recall_score(y_test, y_pred)*100
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test, y_pred)*100

        st.image("mor.png", use_column_width=True)

        st.write("##### Değerlendirmeler:")
        st.write(f'- Sınıflandırıcı: {self.classifier_name}')
        st.write(f'- Sınıflandırıcının accuracy(doğruluk) yüzdesi: %{accuracy: .3f}')
        st.write(f'- Sınıflandırıcının precision yüzdesi: %{precision: .3f}')
        st.write(f'- Sınıflandırıcının recall yüzdesi: %{recall: .3f}')
        st.write(f'- Sınıflandırıcının f1-score yüzdesi: %{f1: .3f}')
        st.write(f'- Sınıflandırıcı için en iyi parametre: {best_params_text}')

        # Cross validation scores overfit test için
        cv_scores,  cv_score_mean = self.cross_validation_score(self.classifier, self.X, self.y)
        st.write("Modelin overfit edip etmediğin test etmek için cross validation yapıyoruz ↴")
        st.write(f'- Cross Validation Skorlarının yüzdesi: %{cv_score_mean*100: .3f}')

        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)

    def cross_validation_score(self, model, X, y, cv=6): # veri seti 5 alt kümeye böler
        scores = cross_val_score(model, X, y, cv=cv)
        mean_score = np.mean(scores)
        return scores, mean_score

    def plot_scatter(self, df):
        # Malignant ve benign örnekleri ayrıştır
        malignant = df[df['diagnosis'] == 1]
        benign = df[df['diagnosis'] == 0]

        # Scatter plot çizdir
        fig = plt.figure(figsize=(8, 6))
        plt.rcParams['font.family'] = 'Serif'
        sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant, label='Kötü', color='deeppink', alpha=0.5)
        sns.scatterplot(x='radius_mean', y='texture_mean', data=benign, label='İyi', color='lightskyblue', alpha=0.5)
        plt.title('Yarıçap Ortalaması ve Doku Ortalamasının Dağılım Grafiği', color='darkslateblue', fontsize=16, fontname='Serif')
        plt.xlabel('Radius Mean', color='darkslateblue', fontsize=14, fontname='Serif')
        plt.ylabel('Texture Mean', color='darkslateblue', fontsize=14, fontname='Serif')
        plt.legend(edgecolor='darkslateblue')  # Çizginin rengini burada belirliyoruz
        st.pyplot(fig)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.rcParams['font.family'] = 'Serif'

        cm_fig = plt.figure(figsize=(6, 4))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', cbar=True,
                    xticklabels=['İyi', 'Kötü'], yticklabels=['İyi', 'Kötü'],
                    annot_kws={"size": 9})  # Sayıların boyutunu 8 olarak ayarla
        # Colorbar nesnesine erişim sağlama
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)  # Colorbar etiketlerinin boyutunu 8 olarak ayarla
        # x ve y ekseni etiketlerinin fontunu değiştirme
        ax.tick_params(axis='both', labelsize=6, labelcolor='darkslateblue', labelrotation=0)
        plt.xlabel('Tahmin Edilen Etiketler', fontsize=7, color='darkslateblue', fontname='Serif')
        plt.ylabel('Doğru Etiketler', fontsize=7, color='darkslateblue', fontname='Serif')
        plt.title('Confusion Matrix', fontsize=9, color='darkslateblue', fontname='Serif')

        ax.text(0.5, 0.8, "True Positive", ha="center", va="center", fontsize=6, color="white", fontname='Serif')
        ax.text(1.5, 0.8, "False Positive", ha="center", va="center", fontsize=6, color="indigo", fontname='Serif')
        ax.text(0.5, 1.8, "False Negative", ha="center", va="center", fontsize=6, color="indigo", fontname='Serif')
        ax.text(1.5, 1.8, "True Negative", ha="center", va="center", fontsize=6, color="white", fontname='Serif')

        st.pyplot(cm_fig)
