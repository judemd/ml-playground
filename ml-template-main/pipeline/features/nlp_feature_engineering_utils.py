import gensim.parsing.preprocessing as gsp


TEXT_CLEANING_FUNCTION_MAPPINGS: dict = {
    "lower_casing_unicode": gsp.lower_to_unicode,  # remove tags
    "strip_tags": gsp.strip_tags,  # remove tags
    "strip_punctuation": gsp.strip_punctuation,  # replace ASCII punctuation characters with spaces
    "strip_multiple_whitespaces": gsp.strip_multiple_whitespaces,  # remove repeating whitespace characters
    "remove_stopwords": gsp.remove_stopwords,
}


NLPFeatures = [
    'item1A_summary',
    'item3_summary',
    'item7_summary',
    'item7A_summary',
]


sw_to_remove = [
    "amount",
    "across",
    "serious",
    "most",
    "back",
    "below",
    "behind",
    "due",
    "afterwards",
    "afterward",
    "less",
    "beforehand",
    "always",
    "more",
    "various",
    "third",
    "full",
    "empty",
    "everywhere",
    "side",
    "beside",
    "besides",
    "front",
    "against",
    "not",
    "cannot",
    "through",
    "thru",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "nowhere",
]
