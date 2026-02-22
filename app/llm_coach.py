import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # ← 追加

def generate_coach_message(score, weekly_drop, predicted_weight, plateau):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が読み込めていません")

    client = OpenAI(api_key=api_key)

    prompt = f"""
あなたはフィジーク大会専門の減量コーチです。

減量スコア: {score}
週間減少率: {weekly_drop}%
予測体重: {predicted_weight}kg
停滞: {plateau}

100文字程度で実践的アドバイスを。
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたはプロのフィジークコーチです。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content