# app.py
import streamlit as st
import numpy as np

# import tensorflow as tf # 如果要加載 Keras 模型
# from tensorflow import keras # 同上

# --- 頁面配置 (建議放最前面) ---
st.set_page_config(
    page_title="ATP 結合位點預測工具",
    page_icon="🧬",  # 可以用 emoji
    layout="wide"
)

# --- 網站標題和描述 ---
st.title("🧬 ATP 結合位點預測器")
st.markdown("""
歡迎使用本工具！這是一個基於機器學習模型的 ATP 結合位點預測器。
請在下方文本框中輸入您的蛋白質序列 (FASTA 格式可接受，但請僅粘貼序列部分，或腳本會嘗試自動處理)。
""")
st.markdown("---")  # 分割線

# --- 用戶輸入蛋白質序列 ---
st.header("1. 輸入蛋白質序列")
sequence_input_raw = st.text_area(
    "在此粘貼蛋白質序列:",
    height=150,
    placeholder="例如：MSEQALKWV...",
    key="protein_sequence_input"
)

# --- 預測按鈕 ---
if st.button("預測 ATP 結合位點", key="predict_button"):
    if not sequence_input_raw.strip():
        st.warning("請先輸入蛋白質序列。")
    else:
        # 預處理輸入的序列
        sequence_to_predict = sequence_input_raw.strip()
        if sequence_to_predict.startswith(">"):  # 簡單處理 FASTA 標頭
            try:
                sequence_to_predict = "".join(sequence_to_predict.splitlines()[1:])
            except:
                st.error("FASTA 格式序列處理失敗，請確保標頭後有序列內容。")
                sequence_to_predict = ""  # 清空以避免後續錯誤

        if sequence_to_predict:
            st.info(
                f"接收到的序列 (長度: {len(sequence_to_predict)}):\n```\n{sequence_to_predict[:100]}...\n```" if len(
                    sequence_to_predict) > 100 else f"接收到的序列 (長度: {len(sequence_to_predict)}):\n```\n{sequence_to_predict}\n```")

            # --- 在這裡將調用模型進行預測 (後續步驟實現) ---
            with st.spinner("正在分析序列並預測結合位點..."):
                # 模擬預測延遲
                import time

                time.sleep(2)  # 替換為真實的模型預測調用

                # 假設預測結果 (後續會替換為真實結果)
                # 預測結果可以是 (位置, 概率) 或直接是標記的序列
                mock_predictions = []
                for i, aa in enumerate(sequence_to_predict):
                    if i % 10 == 0 and i < 100:  # 假設每10個有一個是結合位點 (純模擬)
                        mock_predictions.append({'position': i + 1, 'amino_acid': aa, 'is_binding': True,
                                                 'probability': np.random.uniform(0.6, 0.95)})
                    # else:
                    #     mock_predictions.append({'position': i + 1, 'amino_acid': aa, 'is_binding': False, 'probability': np.random.uniform(0.05, 0.4)})

                st.subheader("2. 預測結果")
                if mock_predictions:
                    st.success("預測完成！")

                    # 以不同方式展示結果 (可以選擇或都用)
                    # 方法一：高亮顯示序列
                    highlighted_sequence_html = ""
                    binding_site_indices = [p['position'] - 1 for p in mock_predictions if p.get('is_binding')]

                    for i, char_aa in enumerate(sequence_to_predict):
                        if i in binding_site_indices:
                            highlighted_sequence_html += f"<span style='background-color: yellow; font-weight: bold; padding: 1px;'>{char_aa}</span>"
                        else:
                            highlighted_sequence_html += char_aa
                    st.markdown("**高亮預測的結合位點:**")
                    st.markdown(
                        f"<div style='font-family: monospace; word-wrap: break-word;'>{highlighted_sequence_html}</div>",
                        unsafe_allow_html=True)
                    st.markdown("---")

                    # 方法二：列表形式展示
                    st.markdown("**詳細預測列表 (部分模擬結果):**")
                    # import pandas as pd
                    # df_predictions = pd.DataFrame([p for p in mock_predictions if p.get('is_binding')])
                    # if not df_predictions.empty:
                    #     st.dataframe(df_predictions[['position', 'amino_acid', 'probability']].round(3))
                    # else:
                    #     st.info("未預測到 ATP 結合位點。")
                    # 簡化版，直接打印
                    predicted_sites_info = []
                    for p in mock_predictions:
                        if p.get('is_binding'):
                            predicted_sites_info.append(
                                f"  - 位置: {p['position']}, 胺基酸: {p['amino_acid']}, 概率 (模擬): {p['probability']:.3f}")
                    if predicted_sites_info:
                        st.text("\n".join(predicted_sites_info))
                    else:
                        st.info("未預測到 ATP 結合位點。")

                else:
                    st.info("根據當前模型，未在此序列中預測到明確的 ATP 結合位點。")
        else:
            st.warning("未能獲取有效序列進行預測。")

# --- 側邊欄 (可選，用於放置說明或其他選項) ---
st.sidebar.header("關於工具")
st.sidebar.info("""
本工具使用機器學習模型預測蛋白質序列中的 ATP 結合位點。
結果僅供研究參考。
""")
st.sidebar.markdown("---")
st.sidebar.subheader("開發者")
st.sidebar.text("陳箴")
# (可以添加模型的簡要說明，例如 "當前使用基於 CNN 的模型")