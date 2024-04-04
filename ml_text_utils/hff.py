from transformers import pipeline

def hf_sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    classifier.save()
    x = classifier("I've been waiting for a HuggingFace course my whole life.")
    print(x)

def hf_zero_shot_classification():
    classifier = pipeline("zero-shot-classification")
    x =  classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    print(x)

def hf_text_generation():
    generator = pipeline("text-generation", model="distilgpt2")
    x = generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
    print(x)


hf_text_generation()