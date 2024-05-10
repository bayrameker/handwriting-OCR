import shutil
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Rakamları Elle Yazma ve Dijital Ortama Aktarma
def capture_images():
    num_samples = 100  # Her rakam için toplam örnek sayısı
    output_dir = "dataset"  # Görüntülerin kaydedileceği dizin

    # Rakamları yakalamak için bir döngü oluşturun
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)  # Rakam dizinini oluşturun

        # Belirli sayıda örnek için bir döngü oluşturun
        for i in range(num_samples):
            print(f"Capture {digit}: {i+1}/{num_samples}")

            # Kamera görüntüsünü yakalayın
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            # Görüntüyü kaydedin
            img_path = os.path.join(output_dir, str(digit), f"{digit}_{i}.png")
            cv2.imwrite(img_path, frame)

            cap.release()
            cv2.destroyAllWindows()

# Görüntüleri Düzenleme ve Boyutlandırma
def preprocess_images(input_dir, output_dir, size=(28, 28)):
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)  # Rakam dizinini oluşturun

        for img_name in os.listdir(os.path.join(input_dir, str(digit))):
            img_path = os.path.join(input_dir, str(digit), img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü gri tonlamalı olarak yükleme
            img = cv2.resize(img, size)  # Görüntüyü belirtilen boyuta yeniden boyutlandırma
            cv2.imwrite(os.path.join(output_dir, str(digit), img_name), img)

# Veri Setini Oluşturma ve Etiketleme
# Bu adımda elle yazılan rakamları görüntülerle birleştirip, her bir görüntüye karşılık gelen etiketleri belirlemeniz gerekecek.

# Eğitim ve Test Setlerine Ayırma
def split_dataset(input_dir, train_dir, test_dir, train_split=0.8):
    for digit in range(10):
        os.makedirs(os.path.join(train_dir, str(digit)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, str(digit)), exist_ok=True)

        images = os.listdir(os.path.join(input_dir, str(digit)))
        num_train = int(len(images) * train_split)
        train_images = images[:num_train]
        test_images = images[num_train:]

        for img_name in train_images:
            shutil.copy(os.path.join(input_dir, str(digit), img_name), os.path.join(train_dir, str(digit), img_name))
        for img_name in test_images:
            shutil.copy(os.path.join(input_dir, str(digit), img_name), os.path.join(test_dir, str(digit), img_name))

# Modeli Oluşturma ve Eğitme
def train_model(train_dir, test_dir):
    # Veri setini yükleme
    X_train, y_train = load_dataset(train_dir)
    X_test, y_test = load_dataset(test_dir)

    # Modeli oluşturma
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Modeli derleme
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Modeli eğitme
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    # Modeli değerlendirme
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

# Özel veri setini yükleme işlevi
def load_dataset(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü gri tonlamalı olarak yükleme
            image = cv2.resize(image, (28, 28))  # Görüntüyü belirtilen boyuta yeniden boyutlandırma
            images.append(image)
            labels.append(int(label))  # Etiketi ekleyin
    # Görüntüleri ve etiketleri numpy dizisine dönüştürme
    X = np.array(images).reshape(-1, 28, 28, 1)
    y = to_categorical(labels)
    return X, y

# capture_images()  # Rakamları yakalamak için bu işlevi çağırın
# preprocess_images("dataset", "preprocessed_dataset")  # Görüntüleri ön işlemek için bu işlevi çağırın
# split_dataset("preprocessed_dataset", "train", "test")  # Veri setini eğitim ve test setlerine ayırmak için bu işlevi çağırın
# train_model("train", "test")  # Modeli eğitmek için bu işlevi çağırın
