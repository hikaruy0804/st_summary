import streamlit as st
import re
import math
from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenizer import Tokenizer as JanomeTokenizer
from janome.tokenfilter import POSKeepFilter, ExtractAttributeFilter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# フィラー、曖昧な表現や不要な言葉リスト
filler_words = ['えーと', 'あの', 'うん', 'そうですね', 'まあ', 'なんか', 'お願いします', 'とは思います', 'ちょっと', 'まぁ', 'いいや', 'それ', 'あれ', 'ですね', 'ですかね', 'とか', 'ですね', 'ですし', 'し。','してますね。','あ、']

def remove_filler_words(text):
    """
    フィラーやあいづち、曖昧な表現を削除する関数
    :param text: 入力文章
    :return: フィラーや曖昧な表現を削除した文章
    """
    for word in filler_words:
        text = text.replace(word, '')
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

    # フィラーや曖昧な表現の削除
    contents = remove_filler_words(contents)

    # 文章を文単位で分割
    text = re.findall("[^。]+。?", contents)

    # Janomeの設定
    tokenizer = JanomeTokenizer('japanese')
    char_filters = [
        UnicodeNormalizeCharFilter(),
        RegexReplaceCharFilter(r'[()「」、。]', ' ')
    ]
    # 名詞や動詞などに限定しつつ、一般的で曖昧な動詞を削除するフィルター
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
    st.write(u'文書要約完了')
    for sentence in summary:
        st.write(text[corpus.index(sentence.__str__())])

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
