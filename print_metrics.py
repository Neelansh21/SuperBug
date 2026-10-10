import joblib
import glob
import json

for file in glob.glob("ml_pipeline/models/*_model.joblib"):
    try:
        data = joblib.load(file)
        sp = data["species"]
        ab = data["antibiotic"]
        mn = data["model_name"]
        print("\n---", sp, "/", ab, "---")
        print("Model:", mn)
        print(json.dumps(data["metrics"], indent=2))
    except Exception as e:
        print("Error loading", file, ":", e)
