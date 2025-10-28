# Simple Naive Bayes (single feature) with probability output

data = [
    ("Sunny", "Play"),
    ("Rainy", "Don't Play"),
    ("Sunny", "Play"),
    ("Rainy", "Don't Play"),
    ("Sunny", "Play")
]

def predict(feature):
    labels = {label for _, label in data}
    probs = {}

    for label in labels:
        total = sum(l == label for _, l in data)
        p_label = total / len(data)
        p_feature_given_label = sum(f == feature and l == label for f, l in data) / total
        probs[label] = p_label * p_feature_given_label

    best_label = max(probs, key=probs.get)
    total_prob = sum(probs.values())
    normalized = {k: v / total_prob for k, v in probs.items()}  # make probabilities sum to 1

    print("\nProbabilities:")
    for label, p in normalized.items():
        print(f"  {label}: {p:.3f}")

    return best_label, normalized[best_label]

# --- Example predictions ---
label, prob = predict("Sunny")
print(f"\nPrediction for 'Sunny': {label} (probability: {prob:.3f})")

label, prob = predict("Rainy")
print(f"Prediction for 'Rainy': {label} (probability: {prob:.3f})")
