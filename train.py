from preprocess import load_images
from model import create_model
from sklearn.model_selection import train_test_split

X, y, class_names = load_images()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(input_shape=(128,128,3), num_classes=len(class_names))
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

model.save("skin_cancer_model.h5")
