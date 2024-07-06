# STEP 1: Load modules
from transformers import pipeline

# STEP 2: Load model
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")

# STEP 3: Input data
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

# STEP 4: Run Inference
result = question_answerer(question=question, context=context)

# STEP 5: Check results
print(result)