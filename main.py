from src.dataset_generator import generate_dataset
from src.eda import run_eda
from src.preprocessing import preprocess_data

def main():
    print("Starting Incident Prediction Pipeline...")

    generate_dataset()
    run_eda()
    preprocess_data()

    print("Pipeline completed.")

if __name__ == "__main__":
    main()