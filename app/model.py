from transformers import pipeline

class ClinicalBertClassifier:
    def __init__(self):
        # We load the model from the local cache to ensure fast startup.
        # This assumes the model is downloaded during the Docker build phase.
        self.model_name = "bvanaken/clinical-assertion-negation-bert"
        self.classifier = None

    def load_model(self):
        print(f"Loading model {self.model_name}...")
        self.classifier = pipeline("text-classification", model=self.model_name)
        print("Model loaded successfully.")

    def predict(self, text: str):
        if not self.classifier:
            raise RuntimeError("Model is not loaded.")
        
        result = self.classifier(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"])
        }

# Global instance to be loaded once
bert_classifier = ClinicalBertClassifier()
