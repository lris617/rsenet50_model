from .cnnlstm import CNNLSTMRegressor, CNNLSTMRegressorResnetSeq
from .resnet import resnet50

model_dict = {
    'resnet50': resnet50,
}

def create_model(model_name, num_classes):   
    return model_dict[model_name](num_classes = num_classes)
