"""
    Stuff related to comparing units and deciding which are similar
"""

import numpy as np


def get_similar_units(template_similarity, unit_ids, unit_1, merged_units):

    units_worst_to_best = unit_ids[np.argsort(template_similarity[unit_1, :])]
    similar_units = []
    for unit_index in reversed(units_worst_to_best):
        if template_similarity[unit_1, unit_index] > 0.2:
            if (other_unit_id := unit_ids[unit_index]) not in merged_units:
                similar_units.append(other_unit_id)
    return similar_units
