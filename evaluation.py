import torch
from sklearn.metrics import confusion_matrix, classification_report

from detection import PneumoniaDetectionCNN
from data_preprocessing import get_test_loader

def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds)
    print("Confusion matrix: \n", cm)
    print("Classification report: \n", cr)
    
model = PneumoniaDetectionCNN()
data_dir = "/home/anirudh/Desktop/Projects/cnn/chest_xray"
test_loader = get_test_loader(data_dir)
model.load_state_dict(torch.load("pneumonia_detection_model.pth"))
evaluate(model, test_loader)