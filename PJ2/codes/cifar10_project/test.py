from model import load_model
from data_loader import get_loaders
from train import evaluate

def test_model():
    model = load_model()
    _, _, test_loader = get_loaders()
    accuracy = evaluate(model, test_loader)
    print(f'Test Accuracy: {100*accuracy:.2f}%')