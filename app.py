# app.py
import streamlit as st
import numpy as np
import tensorflow as tf

# --- 頁面配置 (建議放最前面) ---
st.set_page_config(
    page_title="ATP 結合位點預測工具",
    page_icon="🧬",  # 可以用 emoji
    layout="wide"
)

AMINO_ACIDS_ORDER = sorted(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS_ORDER)}
WINDOW_SIZE = 21
ALPHABET_SIZE = len(AMINO_ACIDS_ORDER)

# --- 您的模型檔案路徑 (相對於 app.py) ---
MODEL_PATH = "best_model_onehot_cnn_ws21_dataAll.keras"
# --- 您之前為這個模型找到的最佳分類閾值 ---
OPTIMAL_THRESHOLD = 0.9157

@st.cache_resource # 對於模型這種大型且不變的對象，使用 cache_resource
def load_prediction_model(model_path):
    """加載訓練好的 Keras 模型。"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"模型從 {model_path} 加載成功。")
        return model
    except Exception as e:
        st.error(f"加載模型失敗: {e}")
        print(f"錯誤: 加載模型 {model_path} 失敗: {e}")
        return None

# 在應用主邏輯的某個地方（例如，在定義 Streamlit 佈局之後，但在按鈕邏輯之前）調用它
# 這樣模型只會在第一次需要時加載一次。
cnn_model = load_prediction_model(MODEL_PATH)


def create_sliding_windows(sequence, window_size):
    """將長序列切分成滑動窗口。"""
    windows = []
    seq_len = len(sequence)
    half_w = window_size // 2
    # 添加填充以處理序列邊緣，使得序列的每個位置都能成為窗口中心
    # 例如，使用一個在訓練時模型未見過的特殊字符或只是重複邊緣字符
    # 為了簡單，這裡我們先不填充，只取能形成完整窗口的部分
    # （這意味著序列兩端的 (window_size-1)/2 個胺基酸不會被預測）
    # 或者，您可以實現與訓練時相同的填充邏輯

    # 不填充的簡化版本 (可能導致預測結果比原序列短)
    # for i in range(seq_len - window_size + 1):
    #     windows.append(sequence[i : i + window_size])

    # 確保每個胺基酸都能作為中心點（如果應用需要對所有位置進行預測）
    # 您可能需要根據訓練時的處理方式來決定這裡的填充策略。
    # 以下是假設我們為每個可能的中心點提取窗口
    valid_center_indices = []
    for i in range(seq_len):
        if i >= half_w and i < seq_len - half_w:
            window = sequence[i - half_w: i + half_w + 1]
            if len(window) == window_size:
                windows.append(window)
                valid_center_indices.append(i)  # 記錄這個窗口對應原序列的中心索引

    # 返回窗口列表和它們在原序列中的中心點索引
    return windows, valid_center_indices


def one_hot_encode_windows(windows_list, window_size, alphabet_map, alphabet_len):
    """將窗口列表進行 One-Hot 編碼。"""
    num_windows = len(windows_list)
    encoded_data = np.zeros((num_windows, window_size, alphabet_len), dtype=np.int8)
    for i, window_seq in enumerate(windows_list):
        for j, amino_acid in enumerate(window_seq):
            if j < window_size and amino_acid in alphabet_map:
                encoded_data[i, j, alphabet_map[amino_acid]] = 1
    return encoded_data

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
    # --- 在 app.py 的 if st.button("預測 ATP 結合位點", ...) 內部 ---
        if not sequence_input_raw.strip():
            st.warning("請先輸入蛋白質序列。")
        else:
            sequence_to_predict = sequence_input_raw.strip().upper()  # 轉換為大寫，去除多餘空格
            # 簡單移除常見的FASTA標頭和換行符
            if sequence_to_predict.startswith(">"):
                try:
                    sequence_to_predict = "".join(sequence_to_predict.splitlines()[1:])
                except:
                    st.error("FASTA 格式序列處理失敗。")
                    sequence_to_predict = ""

            # 移除非法胺基酸字符 (可選，但推薦)
            valid_aas = "".join(AMINO_ACIDS_ORDER)
            cleaned_sequence = "".join([aa for aa in sequence_to_predict if aa in valid_aas])
            if len(cleaned_sequence) != len(sequence_to_predict):
                st.warning(f"輸入序列中包含非標準胺基酸字符，已被移除。處理後序列長度: {len(cleaned_sequence)}")
            sequence_to_predict = cleaned_sequence

            if sequence_to_predict and cnn_model is not None:  # 確保模型已加載
                st.info(
                    f"接收到的有效序列 (長度: {len(sequence_to_predict)}):\n```\n{sequence_to_predict[:100]}...\n```" if len(
                        sequence_to_predict) > 100 else f"接收到的有效序列 (長度: {len(sequence_to_predict)}):\n```\n{sequence_to_predict}\n```")

                with st.spinner("模型正在分析序列並預測結合位點..."):
                    try:
                        # 1. 滑動窗口切分
                        protein_windows, original_center_indices = create_sliding_windows(sequence_to_predict,
                                                                                          WINDOW_SIZE)

                        if not protein_windows:
                            st.info("序列過短，無法形成有效的預測窗口。")
                        else:
                            # 2. One-Hot 編碼
                            encoded_windows = one_hot_encode_windows(protein_windows, WINDOW_SIZE, AA_TO_INT,
                                                                     ALPHABET_SIZE)

                            # 3. 模型預測 (Keras模型通常期望一個批次的輸入)
                            predicted_probabilities = cnn_model.predict(encoded_windows, verbose=0)  # verbose=0 避免打印進度條

                            # 4. 應用閾值得到二分類結果
                            predicted_labels = (predicted_probabilities.flatten() > OPTIMAL_THRESHOLD).astype(int)

                            st.subheader("2. 預測結果")
                            st.success("預測完成！")

                            # 展示結果
                            highlighted_sequence_html = ""
                            # 創建一個與原序列等長的標籤列表，默認為非結合位點
                            final_site_labels = [0] * len(sequence_to_predict)
                            for i, center_idx in enumerate(original_center_indices):
                                if predicted_labels[i] == 1:
                                    final_site_labels[center_idx] = 1  # 標記中心點為結合位點

                            for i, char_aa in enumerate(sequence_to_predict):
                                if final_site_labels[i] == 1:
                                    highlighted_sequence_html += f"<span style='background-color: yellow; font-weight: bold; padding: 1px;'>{char_aa}</span>"
                                else:
                                    highlighted_sequence_html += char_aa
                            st.markdown("**高亮預測的結合位點:**")
                            st.markdown(
                                f"<div style='font-family: monospace; word-wrap: break-word;'>{highlighted_sequence_html}</div>",
                                unsafe_allow_html=True)
                            st.markdown("---")

                            predicted_sites_info = []
                            for i, center_idx in enumerate(original_center_indices):
                                if predicted_labels[i] == 1:
                                    # probability for positive class
                                    prob_for_site = predicted_probabilities[i][0] if predicted_probabilities[
                                                                                         i].ndim > 0 else \
                                    predicted_probabilities[i]
                                    predicted_sites_info.append(
                                        f"  - 位置 (中心): {center_idx + 1}, 胺基酸: {sequence_to_predict[center_idx]}, 預測概率: {prob_for_site:.4f}")

                            if predicted_sites_info:
                                st.markdown("**詳細預測列表 (預測為結合位點的窗口中心):**")
                                st.text("\n".join(predicted_sites_info))
                            else:
                                st.info("根據模型預測，未在此序列中找到 ATP 結合位點。")

                    except Exception as e_predict:
                        st.error(f"預測過程中發生錯誤: {e_predict}")
                        print(f"預測錯誤: {e_predict}")

            elif cnn_model is None:
                st.error("模型未能成功加載，無法進行預測。請檢查服務器日誌。")
            else:  # sequence_to_predict 為空
                st.warning("輸入的序列無效或過短，無法進行預測。")

# --- 側邊欄 (可選，用於放置說明或其他選項) ---
st.sidebar.header("關於工具")
st.sidebar.info("""
本工具使用機器學習模型預測蛋白質序列中的 ATP 結合位點。
結果僅供研究參考。
""")
st.sidebar.markdown("---")
st.sidebar.subheader("開發者")
st.sidebar.text("陳箴")