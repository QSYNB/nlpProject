from pathlib import Path
import hashlib
import html
import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_multitask_bert.pt"

MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "to",
    "too",
    "very",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


st.set_page_config(
    page_title="Multi-task BERT Dashboard",
    page_icon="",
    layout="wide",
)


class MultiTaskBERT(nn.Module):
    def __init__(self, model_name, absa_num_labels=4, emotion_num_labels=6, dropout=0.1):
        super().__init__()
        config = BertConfig.from_pretrained(model_name, local_files_only=True)
        self.encoder = BertModel(config)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.absa_classifier = nn.Linear(hidden_size, absa_num_labels)
        self.emotion_classifier = nn.Linear(hidden_size, emotion_num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, task="absa"):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = self.dropout(outputs.last_hidden_state[:, 0])

        if task == "absa":
            return self.absa_classifier(cls_output)
        if task == "emotion":
            return self.emotion_classifier(cls_output)
        raise ValueError("task must be 'absa' or 'emotion'")


@st.cache_data
def load_data():
    absa_df = pd.read_csv(PROCESSED_DIR / "absa_semeval_2014.csv")
    emotion_df = pd.read_csv(PROCESSED_DIR / "emotion_goemotions_6class.csv")

    with open(PROCESSED_DIR / "label_metadata.json", "r", encoding="utf-8") as f:
        label_metadata = json.load(f)

    absa_label2id = label_metadata["absa_label2id"]
    emotion_label2id = label_metadata["emotion_label2id"]
    id2absa_label = {int(v): k for k, v in absa_label2id.items()}
    id2emotion_label = {int(v): k for k, v in emotion_label2id.items()}

    return absa_df, emotion_df, absa_label2id, emotion_label2id, id2absa_label, id2emotion_label


@st.cache_resource
def load_model():
    model = MultiTaskBERT(
        model_name=str(MODEL_DIR),
        absa_num_labels=len(ABSA_LABEL2ID),
        emotion_num_labels=len(EMOTION_LABEL2ID),
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    return model, tokenizer


def tokenize_with_spans(text):
    return [
        {"token": match.group(0), "start": match.start(), "end": match.end()}
        for match in re.finditer(r"\w+|[^\w\s]", text)
    ]


def get_candidate_words(text):
    candidates = []
    seen = set()
    for token in tokenize_with_spans(text):
        word = token["token"].strip()
        word_norm = word.lower()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9'-]*", word):
            continue
        if word_norm in STOP_WORDS:
            continue
        if len(word_norm) <= 1:
            continue
        if word_norm in seen:
            continue
        seen.add(word_norm)
        candidates.append(word)
    return candidates


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def find_aspect_token_range(text, aspect, tokens):
    aspect_norm = normalize_text(aspect)
    if not aspect_norm:
        return None

    text_lower = text.lower()
    aspect_lower = str(aspect).lower().strip()
    char_start = text_lower.find(aspect_lower)

    if char_start < 0:
        for i in range(len(tokens)):
            for j in range(i, len(tokens)):
                candidate = text[tokens[i]["start"] : tokens[j]["end"]]
                if normalize_text(candidate) == aspect_norm:
                    return i, j
        return None

    char_end = char_start + len(aspect_lower)
    selected = [
        idx
        for idx, token in enumerate(tokens)
        if token["start"] < char_end and token["end"] > char_start
    ]
    if selected:
        return min(selected), max(selected)
    return None


def selected_text_from_tokens(text, tokens, token_range):
    start_idx, end_idx = token_range
    return text[tokens[start_idx]["start"] : tokens[end_idx]["end"]].strip()


def render_highlighted_text(text, tokens, token_range):
    start_idx, end_idx = token_range
    parts = []
    cursor = 0

    for idx, token in enumerate(tokens):
        parts.append(html.escape(text[cursor : token["start"]]))
        token_text = html.escape(text[token["start"] : token["end"]])
        if start_idx <= idx <= end_idx:
            parts.append(f'<mark class="selected-token">{token_text}</mark>')
        else:
            parts.append(f'<span class="plain-token">{token_text}</span>')
        cursor = token["end"]

    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def render_sentence_with_aspects(text, aspects):
    spans = []
    text_lower = text.lower()

    for aspect in aspects:
        aspect = str(aspect).strip()
        if not aspect:
            continue
        pattern = re.escape(aspect.lower())
        for match in re.finditer(pattern, text_lower):
            spans.append((match.start(), match.end()))

    if not spans:
        return html.escape(text)

    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    parts = []
    cursor = 0
    for start, end in merged:
        parts.append(html.escape(text[cursor:start]))
        parts.append(f'<mark class="gold-aspect-token">{html.escape(text[start:end])}</mark>')
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def render_sentence_panel(text, aspects=None):
    highlighted = render_sentence_with_aspects(text, aspects or [])
    st.markdown(
        f"""
        <div class="sentence-panel">
            <div class="sentence-panel-label">Sentence</div>
            <div class="sentence-panel-text">{highlighted}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_aspect_chips(aspects):
    chips = "".join(
        f'<span class="aspect-chip">{html.escape(str(aspect))}</span>'
        for aspect in aspects
    )
    st.markdown(
        f"""
        <div class="aspect-chip-row">
            <span class="aspect-chip-label">Gold aspects</span>
            {chips}
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_absa(text, aspect):
    encoded = tokenizer(
        text=str(text),
        text_pair=str(aspect),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return predict_from_encoded(encoded, task="absa", id2label=ID2ABSA_LABEL)


def predict_emotion(text):
    encoded = tokenizer(
        text=str(text),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return predict_from_encoded(encoded, task="emotion", id2label=ID2EMOTION_LABEL)


def predict_from_encoded(encoded, task, id2label):
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task=task,
        )
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    rows = [
        {"label": id2label[i], "confidence": float(probs[i])}
        for i in sorted(id2label.keys())
    ]
    prob_df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
    top = prob_df.iloc[0]
    return top["label"], float(top["confidence"]), prob_df


def metric_card(title, value, caption=None, status=None):
    status_class = f" metric-card-{status}" if status else ""
    st.markdown(
        f"""
        <div class="metric-card{status_class}">
            <div class="metric-title">{html.escape(title)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            {f'<div class="metric-caption">{html.escape(caption)}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_aspect_results_table(result_df):
    rows = []
    for _, row in result_df.iterrows():
        correct = bool(row["correct"])
        row_class = "row-ok" if correct else "row-bad"
        badge_class = "ok" if correct else "bad"
        badge_text = "Yes" if correct else "No"
        rows.append(
            f'<tr class="{row_class}">'
            f"<td>{html.escape(str(row['aspect']))}</td>"
            f"<td>{html.escape(str(row['gold_label']))}</td>"
            f"<td>{html.escape(str(row['prediction']))}</td>"
            f"<td>{float(row['confidence']):.1%}</td>"
            f'<td><span class="result-badge {badge_class}">{badge_text}</span></td>'
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="result-table-wrap">
            <table class="result-table">
                <thead>
                    <tr>
                        <th>Aspect</th>
                        <th>Gold label</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Correct</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_confidence_bars(prob_df, title):
    rows = []
    for _, row in prob_df.sort_values("confidence", ascending=False).iterrows():
        confidence = float(row["confidence"])
        rows.append(
            '<div class="compact-prob-row">'
            f'<div class="compact-prob-label">{html.escape(str(row["label"]))}</div>'
            '<div class="compact-prob-track">'
            f'<div class="compact-prob-fill" style="width: {confidence * 100:.1f}%"></div>'
            '</div>'
            f'<div class="compact-prob-value">{confidence:.1%}</div>'
            '</div>'
        )

    st.markdown(
        '<div class="compact-prob-card">'
        f'<div class="compact-prob-title">{html.escape(title)}</div>'
        f'{"".join(rows)}'
        '</div>',
        unsafe_allow_html=True,
    )


def confidence_chart(prob_df, title):
    plot_df = prob_df.sort_values("confidence", ascending=True)
    fig = px.bar(
        plot_df,
        x="confidence",
        y="label",
        orientation="h",
        text=plot_df["confidence"].map(lambda x: f"{x:.1%}"),
        range_x=[0, 1],
        color="confidence",
        color_continuous_scale=["#b6c8c3", "#466a6a"],
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=10, r=24, t=52, b=18),
        coloraxis_showscale=False,
        xaxis_title="Confidence",
        yaxis_title="",
    )
    fig.update_traces(textposition="inside", cliponaxis=True)
    st.plotly_chart(fig, width="stretch")


def sample_row(df, split, key):
    subset = df[df["split"] == split].reset_index(drop=True)
    if subset.empty:
        st.error(f"No rows found for split={split!r}.")
        st.stop()

    if key not in st.session_state:
        st.session_state[key] = int(subset.sample(1).index[0])

    if st.button("Random validation sample", width="stretch"):
        st.session_state[key] = int(subset.sample(1).index[0])

    return subset.iloc[st.session_state[key]], subset


def sample_absa_sentence(df, split, key):
    subset = df[df["split"] == split].reset_index(drop=True)
    if subset.empty:
        st.error(f"No rows found for split={split!r}.")
        st.stop()

    sentence_df = subset[["text"]].drop_duplicates().reset_index(drop=True)
    if key not in st.session_state:
        st.session_state[key] = int(sentence_df.sample(1).index[0])

    if st.button("Random validation sentence", width="stretch"):
        st.session_state[key] = int(sentence_df.sample(1).index[0])

    text = str(sentence_df.iloc[st.session_state[key]]["text"])
    sentence_rows = subset[subset["text"] == text].reset_index(drop=True)
    return text, sentence_rows


def mark_page_loading():
    selected_mode = st.session_state.get("mode_selector")
    if selected_mode and selected_mode != st.session_state.get("active_mode"):
        st.session_state.pending_mode = selected_mode
        st.session_state.page_loading = True


def finish_page_loading():
    st.session_state.active_mode = st.session_state.get(
        "pending_mode",
        st.session_state.get("mode_selector", "ABSA validation sample"),
    )
    st.session_state.page_loading = False


def render_loading_panel(mode_name):
    st.markdown(
        '<div class="loading-panel">'
        '<div class="loading-spinner"></div>'
        '<div>'
        f'<div class="loading-title">Loading {html.escape(mode_name)}</div>'
        '<div class="loading-caption">Preparing the page and running model predictions...</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def auto_continue_loading():
    st.button(
        "Continue loading",
        key="continue_mode_switch",
        on_click=finish_page_loading,
    )
    components.html(
        """
        <script>
        setTimeout(() => {
            const buttons = Array.from(window.parent.document.querySelectorAll("button"));
            const target = buttons.find((button) => button.innerText.trim() === "Continue loading");
            if (target) {
                const wrapper = target.closest('[data-testid="stButton"]');
                if (wrapper) wrapper.style.display = "none";
                target.click();
            }
        }, 80);
        </script>
        """,
        height=0,
    )


def set_candidate_aspect(state_key, candidate):
    st.session_state[state_key] = candidate


def selected_aspect_control(text, aspect_options=None, default_aspect=None, allow_gold=True):
    tokens = tokenize_with_spans(text)
    if not tokens:
        st.warning("No selectable tokens found in the text.")
        st.stop()

    options = []
    for aspect in aspect_options or []:
        aspect = str(aspect).strip()
        if aspect and aspect not in options:
            options.append(aspect)

    fallback_aspect = str(default_aspect or tokens[0]["token"]).strip()
    if fallback_aspect and fallback_aspect not in options:
        options.insert(0, fallback_aspect)

    if allow_gold and options:
        source = st.radio(
            "Aspect source",
            ["Gold aspect", "Candidate word"],
            horizontal=True,
        )
    else:
        source = "Candidate word"

    if source == "Gold aspect":
        default_index = options.index(fallback_aspect) if fallback_aspect in options else 0
        selected_aspect = st.selectbox("Aspect", options, index=default_index)
    else:
        candidate_words = get_candidate_words(text)
        if not candidate_words:
            st.warning("No candidate words found after filtering punctuation and stop words.")
            selected_aspect = fallback_aspect
        else:
            text_key = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            state_key = f"candidate_aspect_{text_key}"
            if state_key not in st.session_state or st.session_state[state_key] not in candidate_words:
                st.session_state[state_key] = candidate_words[0]

            st.markdown('<div class="candidate-word-label">Candidate words</div>', unsafe_allow_html=True)
            columns = st.columns(4)
            for idx, candidate in enumerate(candidate_words):
                button_type = "primary" if candidate == st.session_state[state_key] else "secondary"
                with columns[idx % 4]:
                    st.button(
                        candidate,
                        key=f"{state_key}_{idx}",
                        type=button_type,
                        width="stretch",
                        on_click=set_candidate_aspect,
                        args=(state_key, candidate),
                    )

            selected_aspect = st.session_state[state_key]

    selected_aspect = str(selected_aspect).strip()
    token_range = find_aspect_token_range(text, selected_aspect, tokens)
    if token_range is None:
        highlighted = html.escape(text)
    else:
        highlighted = render_highlighted_text(text, tokens, token_range)

    st.markdown(f'<div class="sentence-box">{highlighted}</div>', unsafe_allow_html=True)
    return selected_aspect


ABSA_DF, EMOTION_DF, ABSA_LABEL2ID, EMOTION_LABEL2ID, ID2ABSA_LABEL, ID2EMOTION_LABEL = load_data()


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="column"] {
        min-width: 0;
    }
    .metric-card {
        border: 1px solid #d9e0dc;
        border-radius: 8px;
        box-sizing: border-box;
        width: 100%;
        padding: 14px 16px;
        background: #f8faf9;
        min-height: 118px;
        margin-bottom: 16px;
        overflow: hidden;
    }
    .metric-card-ok {
        border-color: #76b98a;
        background: #f0fbf3;
        box-shadow: inset 4px 0 0 #2f9e55;
    }
    .metric-card-ok .metric-value {
        color: #176c3a;
    }
    .metric-card-bad {
        border-color: #e9b8a9;
        background: #fff7f3;
        box-shadow: inset 4px 0 0 #d4835f;
    }
    .metric-card-bad .metric-value {
        color: #8b4a2f;
    }
    .metric-card-neutral {
        border-color: #cbd6d2;
        background: #f8faf9;
        box-shadow: inset 4px 0 0 #94a7a1;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #52605c;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #19211f;
        line-height: 1.25;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .metric-caption {
        margin-top: 8px;
        color: #687772;
        font-size: 0.86rem;
        line-height: 1.35;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .sentence-box {
        border: 1px solid #d9e0dc;
        border-radius: 8px;
        background: #ffffff;
        padding: 16px;
        font-size: 1.05rem;
        line-height: 1.8;
        margin: 8px 0 12px;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .sentence-panel {
        border: 1px solid #cbd9d4;
        border-radius: 8px;
        background: #f7fbff;
        padding: 18px 20px;
        margin: 10px 0 18px;
        box-shadow: inset 4px 0 0 #4c8dbb;
    }
    .sentence-panel-label {
        color: #52605c;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .sentence-panel-text {
        color: #17202a;
        font-size: 1.35rem;
        line-height: 1.65;
        font-weight: 620;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .gold-aspect-token {
        background: #ffd166;
        color: #232323;
        padding: 2px 6px;
        border-radius: 5px;
        box-shadow: 0 1px 0 rgba(0, 0, 0, 0.08);
    }
    .aspect-chip-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin: -4px 0 18px;
    }
    .aspect-chip-label {
        color: #52605c;
        font-size: 0.9rem;
        font-weight: 700;
        margin-right: 2px;
    }
    .aspect-chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        background: #fff0bd;
        border: 1px solid #f4cc62;
        color: #604600;
        padding: 4px 10px;
        font-size: 0.9rem;
        font-weight: 700;
    }
    .candidate-word-label {
        color: #52605c;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 4px 0 8px;
    }
    .loading-panel {
        display: flex;
        align-items: center;
        gap: 14px;
        border: 1px solid #cbd9d4;
        border-radius: 8px;
        background: #f7fbff;
        padding: 16px 18px;
        margin: 12px 0 18px;
        box-shadow: inset 4px 0 0 #4c8dbb;
    }
    .loading-spinner {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        border: 3px solid #d7e4df;
        border-top-color: #4c8dbb;
        animation: spin 0.85s linear infinite;
        flex: 0 0 auto;
    }
    .loading-title {
        color: #17202a;
        font-size: 1rem;
        font-weight: 750;
    }
    .loading-caption {
        color: #5f6d69;
        font-size: 0.9rem;
        margin-top: 2px;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .compact-prob-card {
        border: 1px solid #d9e0dc;
        border-radius: 8px;
        background: #ffffff;
        padding: 14px 16px;
        margin-bottom: 16px;
    }
    .compact-prob-title {
        color: #52605c;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .compact-prob-row {
        display: grid;
        grid-template-columns: minmax(70px, 110px) 1fr 54px;
        align-items: center;
        gap: 10px;
        margin: 7px 0;
    }
    .compact-prob-label {
        color: #4e5d59;
        font-size: 0.88rem;
        overflow-wrap: anywhere;
    }
    .compact-prob-track {
        height: 10px;
        border-radius: 999px;
        background: #e9efec;
        overflow: hidden;
    }
    .compact-prob-fill {
        height: 100%;
        border-radius: 999px;
        background: #4c8d7a;
    }
    .compact-prob-value {
        color: #52605c;
        font-size: 0.82rem;
        text-align: right;
    }
    .selected-token {
        background: #ffe08a;
        color: #1f2523;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .plain-token {
        padding: 2px 1px;
    }
    .note {
        color: #5f6d69;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .selected-aspect-box {
        border: 1px solid #d9e0dc;
        border-radius: 8px;
        background: #ffffff;
        padding: 12px 14px;
        margin: 6px 0 12px;
    }
    .selected-aspect-label {
        color: #52605c;
        font-size: 0.82rem;
        margin-bottom: 4px;
    }
    .selected-aspect-value {
        color: #19211f;
        font-size: 1.05rem;
        font-weight: 650;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .result-table-wrap {
        margin: 2px 0 18px;
        border: 1px solid #d9e0dc;
        border-radius: 8px;
        overflow: hidden;
        background: #ffffff;
    }
    .result-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 0.95rem;
    }
    .result-table th {
        background: #f6f8f7;
        color: #5f6d69;
        font-weight: 650;
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid #d9e0dc;
    }
    .result-table td {
        color: #202927;
        padding: 11px 12px;
        border-bottom: 1px solid #edf1ef;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .result-table tr:last-child td {
        border-bottom: 0;
    }
    .result-table tr.row-ok td {
        background: #f3fbf5;
    }
    .result-table tr.row-ok td:first-child {
        box-shadow: inset 4px 0 0 #2f9e55;
    }
    .result-table tr.row-bad td {
        background: #fffaf7;
    }
    .result-table tr.row-bad td:first-child {
        box-shadow: inset 4px 0 0 #d89a78;
    }
    .result-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 56px;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .result-badge.ok {
        background: #2f9e55;
        color: #ffffff;
        box-shadow: 0 2px 7px rgba(47, 158, 85, 0.22);
    }
    .result-badge.bad {
        background: #f4d8cc;
        color: #8b4a2f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Multi-task BERT Validation Dashboard")
st.caption(
    "Random validation samples, word-level aspect selection, confidence scores, and correctness checks where gold labels exist."
)

MODES = ["ABSA validation sample", "Emotion validation sample", "Custom sentence analysis"]
if "active_mode" not in st.session_state:
    st.session_state.active_mode = MODES[0]
if "mode_selector" not in st.session_state:
    st.session_state.mode_selector = st.session_state.active_mode

with st.sidebar:
    st.header("Controls")
    mode = st.radio(
        "Mode",
        MODES,
        help="Use validation samples with gold labels, or analyze a sentence you type yourself.",
        key="mode_selector",
        on_change=mark_page_loading,
    )
    st.divider()
    st.write(f"Device: `{DEVICE}`")
    st.write(f"ABSA dev rows: `{len(ABSA_DF[ABSA_DF['split'] == 'dev'])}`")
    st.write(f"Emotion dev rows: `{len(EMOTION_DF[EMOTION_DF['split'] == 'dev'])}`")

if st.session_state.get("page_loading", False):
    render_loading_panel(st.session_state.get("pending_mode", mode))
    auto_continue_loading()
    st.stop()

mode = st.session_state.active_mode
model, tokenizer = load_model()

if mode == "ABSA validation sample":
    text, sentence_rows = sample_absa_sentence(ABSA_DF, split="dev", key="absa_sentence_index")
    default_aspect = str(sentence_rows.iloc[0]["aspect"])

    st.subheader("ABSA validation sample")
    render_sentence_panel(text, aspects=sentence_rows["aspect"].astype(str).tolist())
    render_aspect_chips(sentence_rows["aspect"].astype(str).tolist())

    aspect_predictions = []
    for _, aspect_row in sentence_rows.iterrows():
        aspect = str(aspect_row["aspect"])
        gold_label = str(aspect_row["label_name"])
        pred_label, confidence, _ = predict_absa(text, aspect)
        aspect_predictions.append(
            {
                "aspect": aspect,
                "gold_label": gold_label,
                "prediction": pred_label,
                "confidence": confidence,
                "correct": pred_label == gold_label,
            }
        )

    aspect_result_df = pd.DataFrame(aspect_predictions)
    correct_count = int(aspect_result_df["correct"].sum())
    absa_status = "ok" if correct_count == len(aspect_result_df) else "bad"
    emotion_pred, emotion_conf, emotion_probs = predict_emotion(text)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Gold aspects", str(len(aspect_result_df)), "From SemEval validation labels")
    with col2:
        metric_card("ABSA correct", f"{correct_count}/{len(aspect_result_df)}", status=absa_status)
    with col3:
        metric_card("Emotion prediction", emotion_pred, f"Confidence {emotion_conf:.1%}")

    render_aspect_results_table(aspect_result_df)

    st.markdown(
        '<p class="note">Emotion is predicted for the same sentence, but this SemEval ABSA validation row has no emotion gold label.</p>',
        unsafe_allow_html=True,
    )
    confidence_chart(emotion_probs, "Emotion confidence distribution")

    st.divider()
    st.subheader("Manual aspect check")
    st.caption("Choose a gold aspect, or click a candidate word from the sentence for an extra prediction.")

    selected_aspect = selected_aspect_control(
        text,
        aspect_options=sentence_rows["aspect"].astype(str).tolist(),
        default_aspect=default_aspect,
    )
    selected_pred, selected_conf, selected_probs = predict_absa(text, selected_aspect)

    matching_gold = sentence_rows[
        sentence_rows["aspect"].astype(str).map(normalize_text) == normalize_text(selected_aspect)
    ]
    if matching_gold.empty:
        selected_gold = "N/A"
        selected_correct = "N/A"
        selected_caption = "Selected aspect is not in the gold aspect list"
        selected_status = "neutral"
    else:
        selected_gold = str(matching_gold.iloc[0]["label_name"])
        selected_correct = "Yes" if selected_pred == selected_gold else "No"
        selected_caption = f"Gold: {selected_gold}"
        selected_status = "ok" if selected_correct == "Yes" else "bad"

    st.markdown(
        f"""
        <div class="selected-aspect-box">
            <div class="selected-aspect-label">Selected aspect</div>
            <div class="selected-aspect-value">{html.escape(selected_aspect)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        metric_card("ABSA prediction", selected_pred, f"Confidence {selected_conf:.1%}")
    with col2:
        metric_card("ABSA correct", selected_correct, selected_caption, status=selected_status)

    confidence_chart(selected_probs, "Selected aspect ABSA confidence distribution")

    with st.expander("Raw validation rows for this sentence"):
        st.dataframe(sentence_rows, width="stretch")

elif mode == "Emotion validation sample":
    row, _ = sample_row(EMOTION_DF, split="dev", key="emotion_sample_index")
    text = str(row["text"])
    gold_emotion = str(row["label_name"])

    st.subheader("Emotion validation sample")
    render_sentence_panel(text)

    emotion_pred, emotion_conf, emotion_probs = predict_emotion(text)
    emotion_correct = emotion_pred == gold_emotion

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Emotion prediction", emotion_pred, f"Confidence {emotion_conf:.1%}")
    with col2:
        metric_card("Emotion gold", gold_emotion)
    with col3:
        metric_card("Emotion correct", "Yes" if emotion_correct else "No", status="ok" if emotion_correct else "bad")

    confidence_chart(emotion_probs, "Emotion confidence distribution")

    with st.expander("Raw validation row"):
        st.dataframe(pd.DataFrame([row]), width="stretch")

else:
    st.subheader("Custom sentence analysis")

    if "custom_analysis_done" not in st.session_state:
        st.session_state.custom_analysis_done = False
    if "custom_sentence_text" not in st.session_state:
        st.session_state.custom_sentence_text = ""
    if "custom_sentence_input" not in st.session_state:
        st.session_state.custom_sentence_input = ""

    if not st.session_state.custom_analysis_done:
        st.text_area(
            "Enter a sentence",
            key="custom_sentence_input",
            height=120,
            placeholder="Example: The food was great but the service was slow.",
        )
        if st.button("Analyze", type="primary", width="stretch"):
            sentence = st.session_state.custom_sentence_input.strip()
            if not sentence:
                st.warning("Please enter a sentence before analyzing.")
            else:
                st.session_state.custom_sentence_text = sentence
                st.session_state.custom_analysis_done = True
                st.rerun()
    else:
        text = st.session_state.custom_sentence_text
        if st.button("Next", width="stretch"):
            st.session_state.custom_analysis_done = False
            st.session_state.custom_sentence_text = ""
            st.session_state.custom_sentence_input = ""
            st.rerun()

        render_sentence_panel(text)

        st.markdown("### Emotion analysis")
        emotion_pred, emotion_conf, emotion_probs = predict_emotion(text)
        col1, col2 = st.columns([1, 2])
        with col1:
            metric_card("Emotion prediction", emotion_pred, f"Confidence {emotion_conf:.1%}")
        with col2:
            render_compact_confidence_bars(emotion_probs, "Confidence")

        st.markdown("### ABSA analysis")
        candidate_words = get_candidate_words(text)
        if not candidate_words:
            st.warning("No candidate words found after filtering punctuation and stop words.")
        else:
            selected_aspect = selected_aspect_control(
                text,
                aspect_options=[],
                default_aspect=candidate_words[0],
                allow_gold=False,
            )
            absa_pred, absa_conf, absa_probs = predict_absa(text, selected_aspect)

            st.markdown(
                f"""
                <div class="selected-aspect-box">
                    <div class="selected-aspect-label">Selected aspect</div>
                    <div class="selected-aspect-value">{html.escape(selected_aspect)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                metric_card("ABSA prediction", absa_pred, f"Confidence {absa_conf:.1%}", status="neutral")
            with col2:
                render_compact_confidence_bars(absa_probs, "ABSA confidence")
