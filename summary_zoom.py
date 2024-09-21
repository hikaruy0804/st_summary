from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
