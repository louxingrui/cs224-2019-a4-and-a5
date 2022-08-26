import copy
import string

def init_para_frompretrained(m, pm, share_para=False):
    print("------Init posteriorEncoder------")
    m.shared.weight = pm.embed_tokens.weight

    for i in range(min(len(m.encoder.block), len(pm.block))):
        # layer[0].SelfAttention
        m.encoder.block[i].layer[0].SelfAttention.q.weight = pm.block[i].layer[0].SelfAttention.q.weight if share_para else copy.copy(pm.block[i].layer[0].SelfAttention.q.weight)
        m.encoder.block[i].layer[0].SelfAttention.k.weight = pm.block[i].layer[0].SelfAttention.k.weight if share_para else copy.copy(pm.block[i].layer[0].SelfAttention.k.weight)
        m.encoder.block[i].layer[0].SelfAttention.v.weight = pm.block[i].layer[0].SelfAttention.v.weight if share_para else copy.copy(pm.block[i].layer[0].SelfAttention.v.weight)
        m.encoder.block[i].layer[0].SelfAttention.o.weight = pm.block[i].layer[0].SelfAttention.o.weight if share_para else copy.copy(pm.block[i].layer[0].SelfAttention.o.weight)
        if i == 0:
            m.encoder.block[i].layer[0].SelfAttention.relative_attention_bias.weight = pm.block[i].layer[0].SelfAttention.relative_attention_bias.weight if share_para else copy.copy(pm.block[i].layer[0].SelfAttention.relative_attention_bias.weight)
        # layer[1].Dense
        m.encoder.block[i].layer[1].DenseReluDense.wi_0.weight = pm.block[i].layer[1].DenseReluDense.wi_0.weight if share_para else copy.copy(pm.block[i].layer[1].DenseReluDense.wi_0.weight)
        m.encoder.block[i].layer[1].DenseReluDense.wi_1.weight = pm.block[i].layer[1].DenseReluDense.wi_1.weight if share_para else copy.copy(pm.block[i].layer[1].DenseReluDense.wi_1.weight)
        m.encoder.block[i].layer[1].DenseReluDense.wo.weight = pm.block[i].layer[1].DenseReluDense.wo.weight if share_para else copy.copy(pm.block[i].layer[1].DenseReluDense.wo.weight)
    print("------Done------")

# Global trandform for removing punctuation from words
remove_punctuation = str.maketrans('', '', string.punctuation)

# MTLD internal implementation
def mtld_calc(word_array, ttr_threshold):
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0

    for token in word_array:
        token = token.translate(remove_punctuation).lower()  # trim punctuation, make lowercase
        token_count += 1
        if token not in types:
            type_count += 1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0

    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1

# MTLD implementation
def mtld(word_array, ttr_threshold=0.72):
    if isinstance(word_array, str):
        raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
    # if len(word_array) < 50:
    #     raise ValueError("Input word list should be at least 50 in length")
    return (mtld_calc(word_array, ttr_threshold) + mtld_calc(word_array[::-1], ttr_threshold)) / 2


# HD-D internals

# x! = x(x-1)(x-2)...(1)
def factorial(x):
    if x <= 1:
        return 1
    else:
        return x * factorial(x - 1)


# n choose r = n(n-1)(n-2)...(n-r+1)/(r!)
def combination(n, r):
    r_fact = factorial(r)
    numerator = 1.0
    num = n - r + 1.0
    while num < n + 1.0:
        numerator *= num
        num += 1.0
    return numerator / r_fact


# hypergeometric probability: the probability that an n-trial hypergeometric experiment results
#  in exactly x successes, when the population consists of N items, k of which are classified as successes.
#  (here, population = N, population_successes = k, sample = n, sample_successes = x)
#  h(x; N, n, k) = [ kCx ] * [ N-kCn-x ] / [ NCn ]
def hypergeometric(population, population_successes, sample, sample_successes):
    return (combination(population_successes, sample_successes) * \
            combination(population - population_successes, sample - sample_successes)) / \
           combination(population, sample)


# HD-D implementation
def hdd(word_array, sample_size=42.0):
    if isinstance(word_array, str):
        raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
    if len(word_array) < 50:
        raise ValueError("Input word list should be at least 50 in length")

    # Create a dictionary of counts for each type
    type_counts = {}
    for token in word_array:
        token = token.translate(remove_punctuation).lower()  # trim punctuation, make lowercase
        if token in type_counts:
            type_counts[token] += 1.0
        else:
            type_counts[token] = 1.0
    # Sum the contribution of each token - "If the sample size is 42, the mean contribution of any given
    #  type is 1/42 multiplied by the percentage of combinations in which the type would be found." (McCarthy & Jarvis 2010)
    hdd_value = 0.0
    for token_type in type_counts.keys():
        contribution = (1.0 - hypergeometric(len(word_array), sample_size, type_counts[token_type], 0.0)) / sample_size
        hdd_value += contribution

    return hdd_value
