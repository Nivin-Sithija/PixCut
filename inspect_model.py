import torch
import sys

def main():
    try:
        model = torch.load('best_model.pt', map_location='cpu')
        print(type(model))
        if isinstance(model, dict):
            print("Model is a state_dict.")
            print("Keys:", list(model.keys())[:10])
        else:
            print("Object is arguably a full model instance:")
            print(model)
            
    except Exception as e:
        print("Error loading model:", e)

if __name__ == '__main__':
    main()
