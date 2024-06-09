import streamlit as st
import re
from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenizer import Tokenizer as JanomeTokenizer
from janome.tokenfilter import POSKeepFilter, ExtractAttributeFilter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def normalize_text(contents):
    # 氏名と時間を取り除くための正規表現パターン
    pattern = r"\[.*?\] \d{2}:\d{2}:\d{2}\n"
    # パターンに一致する部分を削除
    contents = re.sub(pattern, "", contents)
    contents = contents.strip()
    # 文章を文単位で分割
    text = re.findall("[^。]+。?", contents.replace('\n', ''))
    return text

def start_document_summarize(text, ratio):
    """
    文章を要約する関数
    :param text: 要約する文章
    :param ratio: 要約率（%）
    """
    # Janomeの設定
    tokenizer = JanomeTokenizer()
    char_filters = [
        UnicodeNormalizeCharFilter(),
        RegexReplaceCharFilter(r'[()「」、。]', ' ')
    ]
    token_filters = [
        POSKeepFilter(['名詞', '形容詞', '副詞', '動詞']),
        ExtractAttributeFilter('base_form')
    ]
    analyzer = Analyzer(char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters)
    
    # 文章のトークン化
    corpus = [' '.join(analyzer.analyze(sentence)) + u'。' for sentence in text]
    
    # Sumyの設定
    parser = PlaintextParser.from_string(''.join(corpus), Tokenizer('japanese'))
    summarizer = LexRankSummarizer()
    
    # 要約の実行
    summary = summarizer(document=parser.document, sentences_count=ratio)
    
    # 要約結果の表示
    st.write('文書要約完了:')
    for sentence in summary:
        st.write(sentence)

# Webアプリケーションのインターフェース
st.title("文章要約システム")
st.write("長文の議事録や資料の文章を要約できます。要約率を入力して「要約開始」ボタンを押してください。")
ratio = st.number_input(label="要約率 ex:30(%)", min_value=1, max_value=99, value=30, step=1)

# 入力方法の選択
select_box = st.selectbox("入力方法：", ["直接入力", "テキストファイル(.txt)"])
contents = ""
if select_box == "直接入力":
    contents = st.text_area(label="入力欄", height=500)
elif select_box == "テキストファイル(.txt)":
    uploaded_file = st.file_uploader(label='Upload file:')
    if uploaded_file is not None:
        contents = uploaded_file.getvalue().decode('utf-8')

# 「要約開始」ボタンが押された場合の処理
if st.button("要約開始") and contents:
    text = normalize_text(contents)
    start_document_summarize(text, ratio)
