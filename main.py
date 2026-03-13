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


def word_in_text_amount(word: str, text: str) -> int:
    """Возвращает количество вхождений нормализованной формы слова в нормализованный текст."""
    normal_word = normalize_word(word)
    # Нормализуем каждое слово текста
    normal_words = [normalize_word(w) for w in text.split()]
    return normal_words.count(normal_word)


def find_TF(word: str, text: str) -> float:
    """
    TF = количество вхождений слова / общее количество слов в тексте.
    Все слова приводятся к нормальной форме.
    """
    word_count = word_in_text_amount(word, text)
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


if __name__ == '__main__':
    sentences = ["Дом — милый дом.",
                 "Здорово оказаться дома.",
                 "Я люблю путешествовать."]
    print(find_TF_IDF('Дом', sentences[1], sentences))
