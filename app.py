from flask import Flask, render_template, request, jsonify
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from BERTClassifier import BERTClassifier

import torch

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# initialize pko-T5 model and tokenizer
t5_model_name = 'C:/Users/user/PycharmProjects/ITSAI/finetuning_model_sum'
t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# initialize koBERT model and tokenizer
ko_model_name = 'skt/kobert-base-v1'
ko_tokenizer = KoBERTTokenizer.from_pretrained(ko_model_name)
ko_bertmodel = BertModel.from_pretrained(ko_model_name, return_dict=False)
ko_model = BERTClassifier(ko_bertmodel, dr_rate=0.5)
ko_model.load_state_dict(torch.load('C:/Users/user/PycharmProjects/ITSAI/koBERT04.pt', torch.device('cpu')))


# T5_summarize.py routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    input_text = request.form["input_text"]
    summarized_text = summarize_text(input_text)
    return jsonify(summarized_text=summarized_text)


def summarize_text(text):
    # Tokenization
    input_ids = t5_tokenizer.encode(
        "Summary: " + text, return_tensors="pt", max_length=1024, truncation=True
    )

    # Generate a summary by inputting it to the model
    output = t5_model.generate(input_ids, max_length=300, num_beams=2, early_stopping=True)  # type: ignore

    # Decode and convert to text form
    summarized_text = t5_tokenizer.decode(output[0], skip_special_tokens=True)

    return summarized_text


# koBERT.py routes
@app.route("/summarize_and_evaluate", methods=["POST"])
def summarize_and_evaluate():
    input_text = request.form["input_text"]

    # T5 요약 실행
    summarized_text = summarize_text(input_text)

    # koBERT 긍부정 분석
    sentiment = classify_sentiment(input_text)

    return jsonify(summarized_text=summarized_text, sentiment=sentiment)


def classify_sentiment(title):
    max_length = 128  # 여기에 원하는 최대 길이를 설정
    inputs = ko_tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    token_ids = inputs['input_ids']
    valid_length = inputs['attention_mask'].sum(dim=1)
    segment_ids = inputs['token_type_ids']
    logits = ko_model(token_ids, valid_length, segment_ids)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment_labels = ["매우 부정", "부정", "중립", "긍정", "매우 긍정"]
    return sentiment_labels[predicted_class]


if __name__ == "__main__":
    app.run(debug=True)
