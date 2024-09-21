from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import streamlit as st
import re

def filter_sentences_by_tfidf(corpus, original_sentences, threshold=0.1):
    # TF-IDFベクトライザーの初期化
    vectorizer = TfidfVectorizer()
    # コーパスからTF-IDF行列を作成
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # 各文の平均TF-IDFスコアを計算
    mean_tfidf_scores = tfidf_matrix.mean(axis=1)
    # 平均スコアが閾値以上の文を選択
    filtered_sentences = []
    filtered_corpus = []
    for i in range(len(corpus)):
        if mean_tfidf_scores[i, 0] >= threshold:
            filtered_sentences.append(original_sentences[i])
            filtered_corpus.append(corpus[i])
    return filtered_sentences, filtered_corpus

def start_document_summarize(contents, ratio):
    # 不要な改行を削除
    contents = contents.replace('\n', ' ').replace('\r', ' ')
    
    # 氏名と時間を取り除く
    pattern = r"\[.*?\] \d{2}:\d{2}:\d{2} "
    contents = re.sub(pattern, "", contents)
    
    # 文の分割
    text = re.findall("[^。]+。?", contents)
    
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
    
    # 形態素解析とコーパスの作成
    corpus = [' '.join(analyzer.analyze(sentence)) for sentence in text]
    
    # TF-IDFによる文のフィルタリング
    filtered_text, filtered_corpus = filter_sentences_by_tfidf(corpus, text, threshold=0.1)
    
    # Sumyの設定
    parser = PlaintextParser.from_string(' '.join(filtered_corpus), Tokenizer('japanese'))
    summarizer = LexRankSummarizer()
    summarizer.stop_words = [' ']
    
    # 要約率をセンテンス数に変換
    lens = len(filtered_corpus)
    a = 100 / lens
    pers = ratio / a
    pers = math.ceil(pers)
    
    # 要約の実行
    summary = summarizer(document=parser.document, sentences_count=int(pers))
    
    # 要約結果の表示
    print(u'文書要約完了')
    for sentence in summary:
        st.write(filtered_text[filtered_corpus.index(sentence.__str__())])

# Webアプリケーションのインターフェース
st.title("文章要約システム")
st.write("長文の議事録や資料の文章を要約できます。要約率を入力して「要約開始」ボタンを押してください。")
st.write("+zoomの文字起こしフォーマット処理を加えています。要約時に発言者と発言内容の整合性おかしくならないように削除します。")
# 要約率の入力
ratio = st.number_input(label="要約率 ex:30(%)", min_value=1, max_value=99, value=30, step=1)

# 入力方法の選択
select_box = st.selectbox("入力方法：", ["直接入力", "テキストファイル(.txt)"])

contents = ""

# 直接入力またはファイルアップロードに応じてコンテンツを取得
if select_box == "直接入力":
    texts = st.text_area(label="入力欄", height=500)
    contents = texts.lower()
elif select_box == "テキストファイル(.txt)":
    uploaded_file = st.file_uploader(label='Upload file:')
    if uploaded_file is not None:
        try:
            texts = uploaded_file.getvalue().decode('utf-8')
            contents = texts.lower()
        except UnicodeDecodeError:
            st.error("ファイルのデコードに失敗しました。utf-8形式のファイルをアップロードしてください。")

# 「要約開始」ボタンが押された場合の処理
if st.button("要約開始") and contents:
    start_document_summarize(contents, ratio)
