import gradio as gr
import requests
import json

API_URL = "http://localhost:8000/predict"

def predict_churn(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_cr_card,
    is_active_member,
    estimated_salary
):
    data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }

    response = requests.post(API_URL, json=data)
    result = response.json()

    label = result["label"]
    prob  = result["probability"]

    color = "🔴" if label == "Churn" else "🟢"
    return f"{color} {label} (probability: {prob:.2%})"


demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Credit Score", value=600),
        gr.Dropdown(
            choices=["France", "Germany", "Spain"],
            label="Geography",
            value="France"
        ),
        gr.Dropdown(
            choices=["Male", "Female"],
            label="Gender",
            value="Male"
        ),
        gr.Slider(minimum=18, maximum=92, label="Age", value=35),
        gr.Slider(minimum=0, maximum=10, label="Tenure", value=5),
        gr.Number(label="Balance", value=0),
        gr.Slider(minimum=1, maximum=4, label="Num of Products", value=1),
        gr.Radio(choices=[0, 1], label="Has Credit Card", value=1),
        gr.Radio(choices=[0, 1], label="Is Active Member", value=1),
        gr.Number(label="Estimated Salary", value=50000),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🏦 Customer Churn Prediction",
    description="Predict if a customer will churn using XGBoost model"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)