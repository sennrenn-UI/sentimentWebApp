# sentimentWebApp
#感情解析webアプリのコードの中身
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# モデルとトークナイザーの準備
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model


# モデルとトークナイザーの読み込み
tokenizer, model = load_model()

# StreamlitのUI
st.title("日本語感情分析AI 😊😭😡")
st.subheader("このWebアプリについて💻")
st.text("こんにちは！AIに興味を持っているそこらへんの高校生です👩‍🎓\n"
        "今回は人間とAIの文章の受け取り方の違いを調べるために、日本語テキストの感情をリアルタイムで解析できるWebアプリの開発をしてみました🛠️\n"
        "テキストを入力すると非常に悪い、悪い、普通、良い、非常に良いの５つの項目で文章にどんな感情が含まれているかを教えてくれます！！")
st.caption("文章を入力してね👇")

# ユーザー入力を受け取る
text = st.text_input("感情分析をする文章を入力してください😊")

if st.button("感情を分析する"):
    if text.strip() == "":
        st.warning("文章を入力してね！")
    else:
        # トークン化
        max_seq_length = 512
        token = tokenizer(text, truncation=True, max_length=max_seq_length, padding="max_length", return_tensors="pt")

        # モデルによる予測
        output = model(input_ids=token['input_ids'], attention_mask=token['attention_mask'])
        logits = output.logits

        # 最大のスコアを持つラベルを取得
        max_index = torch.argmax(logits, dim=1).item()

        # ラベル番号に対応する感情をマッピング
        labels = [
            "非常に悪い",  # 0
            "悪い",  # 1
            "普通",  # 2
            "良い",  # 3
            "非常に良い"  # 4
        ]

        # 結果を表示
        sentiment = labels[max_index]
        st.success(f"🎉 感情は：**{sentiment}** だよ！")
