with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all possible characters in data set
chars = sorted(list(set(text)))
vocab_size = len(chars)

