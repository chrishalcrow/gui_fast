"""
    Curate the units.

    This means:
        1) Check for good quality units
        2) Find the units which are "split" => good merging candidates

"""
import numpy as np


def get_good_units(analyzer):
    qms = analyzer.get_extension('quality_metrics').get_data()
    good_qms = qms.query(
        "snr > 1 & firing_rate > 0.05 & rp_contamination < 0.2")
    return good_qms


def get_type_of_firing(spike, rec_samples, threshold=0.3):

    num_of1spikes = np.sum(
        spike < rec_samples['of1'])/rec_samples['of1']*30_000
    num_vrspikes = np.sum((rec_samples['of1']+rec_samples['vr'] > spike) & (
        spike > rec_samples['of1']))/rec_samples['vr']*30_000
    num_of2spikes = np.sum(
        spike > rec_samples['of1']+rec_samples['vr'])/rec_samples['of2']*30_000

    all_frs = [num_of1spikes, num_vrspikes, num_of2spikes]

    which_is_max = np.argmax(all_frs)
    other_ones = set([0, 1, 2]).difference({which_is_max})

    active = [1, 1, 1]
    for other_one in other_ones:
        if all_frs[other_one] < all_frs[which_is_max]*threshold:
            active[other_one] = 0

    return active


def get_outlier_units(all_spikes, samples):

    outliers = []

    for unit_id, spike_train in all_spikes.items():
        type_of_firing = get_type_of_firing(spike_train, samples)
        if type_of_firing != [1, 1, 1]:
            outliers.append(unit_id)

    return outliers
