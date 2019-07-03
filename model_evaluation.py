import tensorflow as tf
from tensorflow import keras

class ModelEvaluation(object):
    def __init__(self):
        pass
    
    def model_evaluation(self, model, testing_data):
        test_images, test_labels = testing_data
        evaluate_result = model.evaluate(test_images, test_labels)

        return evaluate_result

    def prediction_under_model(self, model, test_images):
        
        classification_result = model.predict(test_images)

        return classification_result