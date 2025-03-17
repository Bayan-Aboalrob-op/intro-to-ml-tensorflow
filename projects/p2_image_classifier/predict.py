import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# here is the Function to load the class names
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

import tensorflow as tf
from PIL import Image
import numpy as np

def process_image(image_path):
    """
    Takes in an image path, loads the image, and preprocesses it
    to make it ready for inference by the model (same preprocessing used during training).

    Args:
    - image_path (str): Path to the image to be processed.

    Returns:
    - numpy.ndarray: Preprocessed image in the form (224, 224, 3)
    """
    image = Image.open(image_path).convert('RGB')

    # Converting the image to a NumPy array
    image = np.asarray(image)

    # Converting the image to a tensor and resize it to 224x224
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))

    # Normalizing the image by scaling pixel values to the [0, 1] range
    image = image / 255.0

    image = tf.expand_dims(image, axis=0)  

    return image

# This is the Function for  making predictions
def predict(image_path, model, class_names, top_k=5):
    # Processing the input image
    img = process_image(image_path)

    # Performing prediction
    probs = model.predict(img)
    top_k_preds = probs[0].argsort()[-top_k:][::-1]  
    # Getting class names from the labels
    class_labels = [class_names[str(i)] for i in top_k_preds]

    return probs[0][top_k_preds], class_labels

# Here, I included the Main function to handle the CLI arguments and process the predictions
def main():
    parser = argparse.ArgumentParser(description="Predict the class of a flower image")
    parser.add_argument('image_path', type=str, help='Path to the image to be predicted')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to the category labels file')
    
    # Parsing the arguments
    args = parser.parse_args()

    model = load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    class_names = load_class_names(args.category_names)

    # Getting predictions using th epretrained saved model
    probs, classes = predict(args.image_path, model, class_names, top_k=args.top_k)

    # Printing the top K predictions
    print(f"Top {args.top_k} Predictions:")
    for i in range(args.top_k):
        print(f"{classes[i]}: {probs[i]:.4f}")

    img = Image.open(args.image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {classes[0]} ({probs[0]:.4f})")
    plt.show()

if __name__ == '__main__':
    main()
