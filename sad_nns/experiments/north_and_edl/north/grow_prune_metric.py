from neurops import effective_rank, svd_score, weight_sum


"""
Add you're own metric here! Don't forget to add it to the metrics dict below.
"""

metrics = {
    'effective_rank': effective_rank,
    'svd_score': svd_score,
    'weight_sum': weight_sum,
}
