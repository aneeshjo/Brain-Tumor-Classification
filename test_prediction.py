
from cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline

def main():
    image_path = "E:\\Project_Works\\ML_Projects\\Deep Learning\\CNN\\Brain-Tumor-Classification\\artifacts\\data_ingestion\\Testing\\meningioma\\Te-aug-me_14.jpg"
   

    pipeline = PredictionPipeline()
    result = pipeline.run_pipeline(image_path)

    print(result)

if __name__ == "__main__":
    main()