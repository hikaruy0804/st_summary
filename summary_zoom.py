import re

# 不要なフレーズを事前に除外するためのリスト
# ここには「あいづち」や意味のないフレーズを追加
exclude_phrases = [
    r"お願いします", r"なんか", r"とか", r"ま", r"かな", r"確かに", r"ですけどね", r"ですかね", r"と思います", r"申請してない",
    r"見やすいかどうか", r"ただ", r"というのも"
]

def remove_exclude_phrases(text):
    """
    不要なフレーズを削除する関数
    :param text: 元のテキスト
    :return: フレーズ削除後のテキスト
    """
    for phrase in exclude_phrases:
        text = re.sub(phrase, '', text)
    return text

def start_document_summarize(contents, ratio):
    """
    文章を要約する関数
    :param contents: 要約する文章
    :param ratio: 要約率（%）
    """
    # 不要な改行を削除
    contents = contents.replace('\n', ' ').replace('\r', ' ')
    
    # 氏名と時間を取り除くための正規表現パターン
    pattern = r"\[.*?\] \d{2}:\d{2}:\d{2} "
    contents = re.sub(pattern, "", contents)
    
    # 不要なフレーズを削除
    contents = remove_exclude_phrases(contents)
    
    # 文章の正規化と文単位での分割
    contents = ''.join(contents)
    text = re.findall("[^。]+。?", contents)
    
    # Janomeの設定
    tokenizer = JanomeTokenizer('japanese')
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
    summarizer.stop_words = [' ']
    
    # 要約率をセンテンス数に変換
    lens = len(corpus)
    a = 100 / lens
    pers = ratio / a
    pers = math.ceil(pers)
    
    # 要約の実行
    summary = summarizer(document=parser.document, sentences_count=int(pers))
    
    # 要約結果の表示
    print(u'文書要約完了')
    for sentence in summary:
        st.write(text[corpus.index(sentence.__str__())])

# Webアプリケーションのインターフェース
st.title("文章要約システム")
st.write("長文の議事録や資料の文章を要約できます。要約率を入力して「要約開始」ボタンを押してください。")
ratio = st.number_input(label="要約率 ex:30(%)", min_value=1, max_value=99, value=30, step=1)
select_box = st.selectbox("入力方法：", ["直接入力", "テキストファイル(.txt)"])

contents = ""

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

if st.button("要約開始") and contents:
    start_document_summarize(contents, ratio)
