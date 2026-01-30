from flask import Flask, render_template, request, session
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load ML components
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Risky categories
risky_categories = ["E-commerce", "Food Delivery", "Online Gaming", "Travel", "Luxury"]

@app.route('/')
def home():
    return render_template(
        "index.html",
        card_types=encoder.categories_[0],
        banks=encoder.categories_[1],
        categories=encoder.categories_[2],
        states=encoder.categories_[3],
        history=session.get("history", [])
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -----------------------------
        # Collect inputs (NO fraud score)
        # -----------------------------
        amount = float(request.form["amount"])
        card_type = request.form["card_type"]
        bank = request.form["bank"]
        category = request.form["category"]
        state = request.form["state"]

        # -----------------------------
        # Encode categorical features
        # -----------------------------
        df_input = pd.DataFrame(
            [[card_type, bank, category, state]],
            columns=["Card Type", "Bank", "Transaction Category", "State"]
        )

        encoded = encoder.transform(df_input)
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out()
        )

        # -----------------------------
        # IMPORTANT FIX
        # Inject dummy fraud score
        # -----------------------------
        dummy_fraud_score = 50  # neutral value

        full_input = np.concatenate(
            [[amount, dummy_fraud_score], encoded_df.values.flatten()]
        )

        scaled = scaler.transform([full_input])

        # -----------------------------
        # Prediction
        # -----------------------------
        base_prob = model.predict_proba(scaled)[0][1]
        risk_percent = base_prob

        # Amount influence
        if amount > 50000:
            risk_percent += 0.15
        elif amount > 20000:
            risk_percent += 0.10
        elif amount > 10000:
            risk_percent += 0.05

        # Category influence
        if category in risky_categories:
            risk_percent += 0.10

        # Clamp
        risk_percent = max(0, min(risk_percent, 1))
        risk_display = round(risk_percent * 100, 2)

        # -----------------------------
        # Risk classification
        # -----------------------------
        if risk_percent < 0.3:
            risk_level = "Low"
            color = "green"
            prediction_text = "✅ Transaction appears Legitimate"
            advice = "This transaction is safe."
        elif risk_percent < 0.7:
            risk_level = "Medium"
            color = "orange"
            prediction_text = "⚠️ Transaction may be Risky"
            advice = "Verify transaction details carefully."
        else:
            risk_level = "High"
            color = "red"
            prediction_text = "🚨 Potential Fraud Detected"
            advice = "High risk! Contact your bank immediately."

        # -----------------------------
        # Insights
        # -----------------------------
        insights = []
        if amount > 20000:
            insights.append("💰 High transaction amount increases risk.")
        if category in risky_categories:
            insights.append(f"📦 '{category}' category has higher fraud incidence.")
        insights.append(f"📍 Regional monitoring applied for {state}.")

        # -----------------------------
        # Save history
        # -----------------------------
        record = {
            "amount": amount,
            "bank": bank,
            "category": category,
            "risk_percent": risk_display,
            "risk_level": risk_level
        }

        session.setdefault("history", [])
        session["history"].insert(0, record)
        session["history"] = session["history"][:5]

        return render_template(
            "index.html",
            prediction_text=prediction_text,
            risk_percent=risk_display,
            risk_level=risk_level,
            color=color,
            advice=advice,
            insights=insights,
            amount=amount,
            card_type_input=card_type,
            bank_input=bank,
            category_input=category,
            state_input=state,
            card_types=encoder.categories_[0],
            banks=encoder.categories_[1],
            categories=encoder.categories_[2],
            states=encoder.categories_[3],
            history=session["history"]
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}",
            card_types=encoder.categories_[0],
            banks=encoder.categories_[1],
            categories=encoder.categories_[2],
            states=encoder.categories_[3]
        )

if __name__ == "__main__":
    app.run(debug=True)
