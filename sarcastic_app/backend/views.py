import traceback
from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from .model_loader import cnn_lstm_model, tokenizer
from .model_loader import (
    transformer_model,
    transformer_tokenizer,
    clean_text)
import torch


MAX_LEN = 50


def preprocess(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = np.zeros((len(sequences), MAX_LEN))
    for i, seq in enumerate(sequences):
        padded[i, :len(seq[:MAX_LEN])] = seq[:MAX_LEN]
    return padded


# def predict_ml(texts):
#     preds = ml_model.predict(texts)
#     return preds
#
#
def predict_cnn_lstm(texts):
    processed = preprocess(texts)
    preds = cnn_lstm_model.predict(processed)
    return preds


def predict_transformer(texts):
    results = []
    confidences = []

    for text in texts:
        cleaned = clean_text(text)

        inputs = transformer_tokenizer(
            cleaned,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128   # 🔥 FIX ADDED HERE
        )

        with torch.no_grad():
            outputs = transformer_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

        results.append(pred)
        confidences.append(confidence)

    return results, confidences


def index(request):
    return render(request,"sarcastic_detection.html")

@csrf_exempt
def predict_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))

        headlines = data.get("headlines", [])
        model = data.get("model")

        if not isinstance(headlines, list) or len(headlines) == 0:
            return JsonResponse({"error": "Invalid headlines"}, status=400)

        results = []

        def format_output(text, model_name, pred, confidence):
            return {
                "headline": str(text),
                "model": model_name,
                "prediction": "Sarcastic" if int(pred) == 1 else "Not Sarcastic",
                "confidence": round(float(confidence) * 100, 2)
            }

        # # ================= ML =================
        # if model == "ML":
        #     preds = predict_ml(headlines)
        #     for text, p in zip(headlines, preds):
        #         val = float(p[0]) if hasattr(p, "__len__") else float(p)
        #         results.append(format_output(text, "ML", val > 0.5, val))
        #
        # ================= CNN + LSTM =================
        if model == "CNN_LSTM":
            preds = predict_cnn_lstm(headlines)
            for text, p in zip(headlines, preds):
                val = float(p[0]) if hasattr(p, "__len__") else float(p)
                results.append(format_output(text, "CNN+LSTM", val > 0.5, val))

        # ================= TRANSFORMER =================
        elif model == "Transformer":
            preds, confs = predict_transformer(headlines)

            for text, p, c in zip(headlines, preds, confs):
                results.append(format_output(text, "Transformer", p, c))

        # ================= ALL =================
        elif model == "ALL":
            # ml_preds = predict_ml(headlines)
            cnn_preds = predict_cnn_lstm(headlines)
            tf_preds, tf_confs = predict_transformer(headlines)

            for i, text in enumerate(headlines):

                # ML
                # ml_val = float(ml_preds[i][0]) if hasattr(ml_preds[i], "__len__") else float(ml_preds[i])
                # results.append(format_output(text, "ML", ml_val > 0.5, ml_val))
                #
                # CNN
                cnn_val = float(cnn_preds[i][0]) if hasattr(cnn_preds[i], "__len__") else float(cnn_preds[i])
                results.append(format_output(text, "CNN+LSTM", cnn_val > 0.5, cnn_val))

                # Transformer
                results.append(format_output(text, "Transformer", tf_preds[i], tf_confs[i]))

        else:
            return JsonResponse({

                "error":"Invalid model selected"

            },status=400)

        return JsonResponse({"results": results}, status=200)

    except Exception as e:
        print("\n===== BACKEND ERROR =====")
        traceback.print_exc()
        print("=========================\n")

        return JsonResponse({
            "error": str(e),
            "type": str(type(e))
        }, status=500)