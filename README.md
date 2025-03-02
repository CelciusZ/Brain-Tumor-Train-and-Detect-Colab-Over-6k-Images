# Tumor Detection with Convolutional Neural Network (CNN)  
**Ad Soyad:** Enez Aykut GÜLIRMAK  
Bu proje, beyin MR görüntülerini **Tumor Positive** (Tümör Var) ve **Tumor Negative** (Tümör Yok) olarak sınıflandırmak için bir Convolutional Neural Network (CNN) modelini içermektedir.  
Model, Keras ve TensorFlow kullanılarak geliştirilmiş ve Google Drive üzerinde saklanan 5,899 görüntülük bir veri seti ile eğitilmiştir.  
Amaç, tıbbi görüntüleme alanında derin öğrenme tekniklerini kullanarak tümör tespitini otomatikleştirmektir.  
## Proje Özeti  
- **Amaç:** Beyin MR görüntülerinde tümör varlığını tespit etmek.  
- **Veri Seti:** Toplam 5,899 görüntü. `yes/`: Tümör içeren görüntüler. `no/`: Tümör içermeyen görüntüler.  
- **Donanım:** Google Colab ortamı.  
- **Yazılım:** Model: Keras ve TensorFlow. Veri işleme: NumPy, PIL, Scikit-learn. Görselleştirme: Matplotlib.  
## Sistem Nasıl Çalışır?  
### Veri Hazırlığı  
Görüntüler `/gdrive/MyDrive/Colab Notebooks/input1500/` yolundan alınır.  
Her görüntü 64x64 piksele küçültülür ve RGB formatında işlenir.  
Etiketleme: One-hot encoding ile 0 (Tumor Positive) ve 1 (Tumor Negative).  
Veri seti %85 eğitim (5,014 görüntü) ve %15 test (885 görüntü) olarak bölünmüştür.  
Örnek Çıktı: ```data = np.array(data)  print(data.shape[0])  # Çıktı: 5899```  
### Model Mimarisi  
- **Katmanlar:** `Conv2D` (8 filtre, 3x3 kernel, padding='Same') x2, `BatchNormalization`, `MaxPooling2D` (2x2), `Dropout` (0.5), `Conv2D` (32 filtre, 2x2 kernel, padding='Same') x2, `BatchNormalization`, `MaxPooling2D` (2x2, stride=2), `Dropout` (0.3), `Flatten`, `Dense` (64 birim, ReLU), `Dropout` (0.1), `Dense` (2 birim, softmax).  
- **Parametreler:** 530,634 toplam, 530,554 eğitilebilir.  
- **Optimizer:** SGD (öğrenme oranı eksponansiyel azalma ile).  
- **Loss:** Categorical Crossentropy.  
- **Metrik:** Accuracy.  
### Eğitim Süreci  
- **Epoch:** 50  
- **Batch Size:** 64  
Model, eğitim ve doğrulama verileriyle 50 epoch boyunca eğitildi.  
Örnek Çıktı (Son Epoch): ```Epoch 50/50  79/79 [==============================] - 33s 415ms/step - loss: 0.0472 - accuracy: 0.9826 - val_loss: 0.0746 - val_accuracy: 0.9774```  
- **Sonuçlar:** Eğitim Doğruluğu: %98.26, Doğrulama Doğruluğu: %97.74, Son Epoch Kayıp: 0.0746 (val_loss)  
#### Eğitim ve Doğrulama Kayıp Grafiği  
![Eğitim Grafiği](https://github.com/CelciusZ/Tumor-Detection-with-Convolutional-Neural-Network-CNN-/raw/main/egitim.png)  
### Tahmin Sistemi  
`predict_tumor2` fonksiyonu ile yeni görüntüler sınıflandırılır.  
Örnek Çıktılar:  
- **Tumor Positive Örneği:** ```99.97826218605042%    TUMOR POSITIVE```  
![Tumor Positive](https://github.com/CelciusZ/Tumor-Detection-with-Convolutional-Neural-Network-CNN-/raw/main/positive.png)  
- **Tumor Negative Örneği:** ```99.99998807907104%    TUMOR NEGATIVE```  
![Tumor Negative](https://github.com/CelciusZ/Tumor-Detection-with-Convolutional-Neural-Network-CNN-/raw/main/negative.png)  
- **Performans:** Tumor Positive tahmini: ~0.0129 saniye/görüntü, Tumor Negative tahmini: ~0.0181 saniye/görüntü  
## Kurulum ve Kullanım  
1. **Gereksinimler:** ```pip install tensorflow keras numpy pandas matplotlib pillow scikit-learn```  
2. **Google Drive Bağlantısı:** ```from google.colab import drive  drive.mount("/gdrive", force_remount=True)  %cd /gdrive/MyDrive/'Colab Notebooks'/input1500/```  
3. **Model Eğitimi:** ```history = model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1, validation_data=(x_test, y_test))```  
4. **Tahmin Yapma:** ```img = Image.open(r"/gdrive/Othercomputers/PC2/input1500/yes/y36.jpg")  predict_tumor2(img)```  
5. **Model Kaydetme ve Yükleme:** ```model.save('proje')  model = tf.keras.models.load_model('proje')```  
## Ekran Görüntüleri  
- **Eğitim Grafiği:** [egitim.png](https://github.com/CelciusZ/Tumor-Detection-with-Convolutional-Neural-Network-CNN-/raw/main/egitim.png)  
- **Tumor Positive Örneği:** [positive.png](https://github.com/CelciusZ/Tumor-Detection-with-Convolutional-Neural-Network-CNN-/raw/main/positive.png)  
- **Tumor Negative Örneği:** [negative.png](https://github.com/CelciusZ/Tumor-Detection-with-Convolutional-Neural-Network-CNN-/raw/main/negative.png)  
