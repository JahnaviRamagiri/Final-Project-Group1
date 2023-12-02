def print_title(title, pattern="*", pattern_length=20, num_blank_lines=1):
    """
    :param title: String to be printed
    :param pattern: Pattern preceeding and Succeding the String in the title
    :param pattern_length: Length of pattern
    :param num_blank_lines: Total blank lines before and after the Title
    """
    print((num_blank_lines // 2 + 1) * "\n", pattern_length * pattern, title, pattern_length * pattern,
          num_blank_lines // 2 * "\n")