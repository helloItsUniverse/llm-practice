# STEP 1: Load modules
from transformers import pipeline

# STEP 2: Load model
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# STEP 3: Input data
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text = "C쇼크에 멈춘 흑자비행…대한항공 1분기 영업적자 566억"

# STEP 4: Run Inference
result = classifier(text)

# STEP 5: Check results
print(result)