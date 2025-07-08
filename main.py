"""
Face Mask Detection - Main Script
"""
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Face Mask Detection Project')
    parser.add_argument('operation', choices=['train', 'evaluate', 'detect', 'predict'], 
                       help='Operation to perform')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    
    args = parser.parse_args()
    
    if args.operation == 'train':
        print("Starting model training...")
        try:
            os.system("C:/Users/selay/anaconda3/python.exe quick_train.py")
            print("Training completed!")
        except Exception as e:
            print(f"Error during training: {e}")
    
    elif args.operation == 'evaluate':
        print("Evaluating model...")
        try:
            os.system("C:/Users/selay/anaconda3/python.exe quick_eval.py")
            print("Evaluation completed!")
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
    elif args.operation == 'detect':
        print("Starting real-time detection...")
        try:
            from src.detect import main as detect_main
            detect_main()
        except Exception as e:
            print(f"Error during detection: {e}")
    
    elif args.operation == 'predict':
        if not args.image:
            print("Please provide an image path using --image flag")
            return
        
        print(f"Predicting on image: {args.image}")
        try:
            from src.evaluate import predict_external_image
            predict_external_image(args.image)
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
