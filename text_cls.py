# STEP 1: Load modules
from transformers import pipeline

# STEP 2: Load model
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# STEP 3: Input data
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text = "샤오미의 폴더블 폰의 점유율이 삼성전자보다 높아졌다."

# STEP 4: Run Inference
result = classifier(text)

# STEP 5: Check results
print(result)