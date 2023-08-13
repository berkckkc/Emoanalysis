import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from deepface import DeepFace

### Emoanalysis  v1.0 --- by Berk ÇIKIKCI --- don't forget to follow on Linkedin(Berk ÇIKIKCI) for further projects ----  ###

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection and Emotion Analysis App")
        self.setGeometry(100, 100, 800, 500)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setAlignment(Qt.AlignTop)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.import_button = QPushButton("Import Picture", self)
        self.import_button.clicked.connect(self.select_image)
        self.button_layout.addWidget(self.import_button)

        self.retry_button = QPushButton("Try Again", self)
        self.retry_button.clicked.connect(self.clear_and_select_image)
        self.retry_button.setEnabled(False)
        self.button_layout.addWidget(self.retry_button)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        self.button_layout.addWidget(self.exit_button)

        


    
    def clear_and_select_image(self):
        self.clear_result()
        self.select_image()


    def select_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Picture Files (*.jpg *.jpeg *.png)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.process_image(file_path)

    def process_image(self, file_path):
        img = cv2.imread(file_path)
        if img is not None:
            img = self.resize_image(img, 800)  # Limit 800 pixels
            faces = self.detect_faces(img)

            if len(faces) == 0:
                self.show_error_message("No Face Found", "No face found in the selected image. Please try again.")
                self.clear_result()
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_img = img[y:y+h, x:x+w]
                    emotions = DeepFace.analyze(face_img, actions=['emotion'])

                    if emotions and 'emotion' in emotions[0]:
                        emotion_values = emotions[0]['emotion']
                        self.show_result(img, emotion_values)
                    else:
                        self.show_error_message("No Emotion Found", "No emotion found in the selected image. Please try another picture.")
                        self.clear_result()
        else:
            self.show_error_message("Image Failed to Load", "The selected image failed to load. Please select a valid image.")

    def resize_image(self, img, max_size):
        height, width, _ = img.shape
        if height > max_size or width > max_size:
            if height > width:
                scale_factor = max_size / height
            else:
                scale_factor = max_size / width
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        return img

    def detect_faces(self, img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def show_result(self, img, emotion_values):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

        self.plot_emotion_graph(emotion_values)
        self.retry_button.setEnabled(True)
        

    def plot_emotion_graph(self, emotion_values):
        emotions = list(emotion_values.keys())
        values = list(emotion_values.values())

        total = sum(values)
        percentages = [value * 100 / total for value in values]

        # Distribution to 100%
        remaining_percentage = 100 - sum(percentages)
        max_index = np.argmax(percentages)
        percentages[max_index] += remaining_percentage

        plt.figure(figsize=(8, 5))
        plt.bar(emotions, percentages)
        plt.xlabel('Emotions')
        plt.ylabel('Percentage')
        plt.title('Emotion Analysis')
        plt.tight_layout()
        
        # Automatically save the graph
        file_number = 1
        while os.path.exists(f"analysis{file_number}.jpg"):
            file_number += 1
        plt.savefig(f"analysis{file_number}.jpg")
        plt.savefig('emotion_graph.png')
        plt.close()

        graph_image = cv2.imread('emotion_graph.png')
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_BGR2RGB)
        height, width, channel = graph_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(graph_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        graph_pixmap = QPixmap.fromImage(q_img)
        graph_label = QLabel(self)
        graph_label.setPixmap(graph_pixmap)
        self.layout.addWidget(graph_label)

    
    def clear_result(self):
        self.image_label.clear()
        self.layout.itemAt(2).widget().deleteLater()
        self.retry_button.setEnabled(False)

    def show_error_message(self, title, message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

### Emoanalysis  v1.0 --- by Berk ÇIKIKCI --- don't forget to follow on Linkedin(Berk ÇIKIKCI) for further projects ----  ###