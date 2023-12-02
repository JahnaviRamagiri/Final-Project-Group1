def remove_repeating_words(words, range_limit=3):
    unique_words = []
    seen_words = set()

    for i, word in enumerate(words):
        if word not in seen_words:
            unique_words.append(word)
            seen_words.add(word)

            # Remove words that are beyond the range limit
            if i >= range_limit:
                seen_words.remove(words[i - range_limit])

    return unique_words

# Example usage:
word_list = ["apple", "banana", "orange", "apple", 'kiwi', "grape", "banana", "kiwi", "apple", "orange"]
result = remove_repeating_words(word_list)
print(word_list)
print(result)