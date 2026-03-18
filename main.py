import math
import re

import pymorphy3

# Создаём анализатор один раз для обеих функций
_morph = pymorphy3.MorphAnalyzer()


def normalize_word(word: str) -> str:
    """
    Приводит слово к нормальной (словарной) форме.
    Знаки препинания, прилегающие к слову, отбрасываются.
    """
    # Ищем последовательность букв (включая ё) и дефисов
    match = re.search(r'[а-яА-ЯёЁa-zA-Z-]+', word)
    if match:
        clean_word = match.group(0)
        parsed = _morph.parse(clean_word)[0]
        return parsed.normal_form
    # Если в строке нет букв, возвращаем её без изменений
    return word


def normalize_text(text: str) -> str:
    return " ".join(normalize_word(word) for word in text.lower().split())


def word_in_sentence(word, sentence) -> bool:
    """переводит все аргументы в нормальную форму и проверяет, что слово есть в предложении"""
    normal_word = normalize_word(word)
    normal_sentence = normalize_text(sentence)
    return normal_word.lower() in normal_sentence.lower().split()


def phrase_amount_in_text(text: str, *words) -> int:
    """
    Возвращает количество вхождений словосочетания (последовательности слов)
    в текст. Слова и текст предварительно нормализуются.
    """
    # Нормализуем искомые слова
    target = [normalize_word(w) for w in words]
    if not target:
        return 0

    # Нормализуем текст и разбиваем на список слов
    text_words = normalize_text(text).split()

    count = 0
    len_target = len(target)
    # Ищем все вхождения подсписка target в text_words
    for i in range(len(text_words) - len_target + 1):
        if text_words[i:i + len_target] == target:
            count += 1
    return count


def find_TF(word: str, text: str) -> float:
    """
    TF = количество вхождений слова / общее количество слов в тексте.
    Все слова приводятся к нормальной форме.
    """
    word_count = phrase_amount_in_text(text, word)
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    tf = word_count / total_words
    return round(tf, 2)


def find_IDF(word, user_sentences) -> float:
    """IDF (Inverse Document Frequency) = log_10 (N / Nw), где N  — общее число текстов,
    a Nw  — число текстов, в которых встречается слово word.
    :arg word может быть в любой форме
    :arg user_sentences может быть в любой форме"""
    N = len(user_sentences)
    Nw = 0
    for sentence in user_sentences:
        if word_in_sentence(word, sentence):
            Nw += 1
    if Nw == 0:
        return 0
    IDF = math.log10(N / Nw)
    return round(IDF, 2)


def find_TF_IDF(word: str, d: str, user_sentences):
    TF = find_TF(word, d)
    IDF = find_IDF(word, user_sentences)
    return round(TF * IDF, 2)


def compute_tf_idf_for_sentences(sentences):
    """
    Вычисляет TF‑IDF для каждого уникального слова в каждом предложении корпуса.

    Аргументы:
        sentences (list of str): список предложений (документов).

    Возвращает:
        dict: словарь вида {слово: {предложение: значение TF‑IDF}}.
              Ключи предложений формируются как "sent_1", "sent_2", ... по порядку.
    """
    # Собираем все уникальные нормализованные слова из корпуса
    all_words = set()
    for sent in sentences:
        words = [normalize_word(w) for w in sent.split()]
        all_words.update(words)

    # Для каждого слова считаем IDF один раз (он одинаков для всех документов)
    idf_cache = {word: find_IDF(word, sentences) for word in all_words}

    result = {}
    for word in all_words:
        idf = idf_cache[word]
        word_dict = {}
        for idx, sent in enumerate(sentences, start=1):
            tf = find_TF(word, sent)  # уже округлён до 2 знаков
            tf_idf = round(tf * idf, 2)  # итоговое значение
            word_dict[f'sent_{idx}'] = tf_idf
        result[word] = word_dict

    return result


def words_frequency(text: str, *words) -> float:
    """ word1, word2, word3..., text могут быть ненормализованный"""
    nw = phrase_amount_in_text(text, *words)  # кол-во слова/словосочетаний в тексте
    n = len(text.split())  # всего слов в тексте
    return round(nw / n, 2)


def find_pmi(text: str, *words) -> float:
    """
    Вычисляет PMI для словосочетания из *words в заданном тексте.
    Все слова приводятся к нормальной форме.
    """
    # Частота совместной встречи (числитель)
    p_xy = words_frequency(text, *words)

    # Частоты отдельных слов (знаменатель)
    denominator = 1.0
    for word in words:
        p_x = words_frequency(text, word)
        denominator *= p_x

    if denominator == 0:
        return 0.0

    pmi = math.log10(p_xy / denominator)
    return round(pmi, 2)


def get_sentences_from_file(filename):
    with open(filename, encoding='utf8') as file:
        sentences = file.readlines()
    return sentences


def find_words_probability(text: str, *words):
    """Вероятность появления слов words в корпусе text"""
    n = len(text.split())
    nw = phrase_amount_in_text(normalize_text(text), *words)
    probability = nw / n
    return probability


def find_words_sequence_probability(text: str, m_word: str, *words):
    """Найдет вероятность того, что слово m_word следует за словами words"""
    whole_phrase = ' '.join(normalize_word(word) for word in words) + ' ' + m_word
    # whole_phrase - фраза, вероятность появления которой мы хотим найти
    n_phrase = phrase_amount_in_text(text, *whole_phrase.split())  # число раз, когда в корпусе встречается whole_phrase
    n_without_m = phrase_amount_in_text(text, *words)
    if n_without_m == 0:
        print(f"Искал фразу: {n_phrase}, но не нашел")
        return 0
    p = n_phrase / n_without_m
    return p


def find_random_phrase_probability(text: str, *words):
    p = 1
    for index in range(len(words)):
        m_word = words[index]
        p_words = words[:index]
        if len(p_words) == 0:
            pw = find_words_probability(text, m_word)
        else:
            pw = find_words_sequence_probability(text, m_word, *p_words)
        p *= pw
    return p


def find_perplexity(text, *words):
    p = find_random_phrase_probability(text, *words)
    m = len(list(words))
    perplexity = (1 / p) ** 1 / m
    return perplexity


if __name__ == '__main__':
    words = "ноль один два три четыре пять шесть семь восемь девять".split()
