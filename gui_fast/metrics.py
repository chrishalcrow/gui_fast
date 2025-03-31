import numpy as np
import spikeinterface.full as si


def compute_metrics(data, unit_id_1, unit_id_2):

    single_metrics = get_single_unit_metrics(data, unit_id_1, unit_id_2)
    relative_metrics = get_relative_metrics(data, unit_id_1, unit_id_2)

    return single_metrics | relative_metrics


def get_single_unit_metrics(data, unit_id_1, unit_id_2):

    single_unit_metrics = {}

    quality_metrics = data.quality_metrics
    template_metrics = data.template_metrics

    qm_list = ['presence_ratio', 'snr',
               'isi_violations_ratio', 'rp_contamination',
               'amplitude_cv_median', 'amplitude_cv_range',
               'sync_spike_2', 'sync_spike_4', 'firing_range',
               'sd_ratio']
    tm_list = ['peak_to_valley', 'peak_trough_ratio', 'half_width',
               'repolarization_slope', 'recovery_slope', 'num_positive_peaks',
               'num_negative_peaks', 'velocity_above', 'velocity_below', 'exp_decay',
               'spread']

    for a, unit_id in enumerate([unit_id_1, unit_id_2]):

        for metric in qm_list:
            single_unit_metrics[f'{metric}_{a}'] = quality_metrics[metric][unit_id]

        for metric in tm_list:
            single_unit_metrics[f'{metric}_{a}'] = template_metrics[metric][unit_id]

    return single_unit_metrics


def l2_metric(d1, d2):
    return np.sqrt(np.pow(d1[0] - d2[0], 2) + np.pow(d1[1] - d2[1], 2))


def get_relative_metrics(data, unit_id_1, unit_id_2):

    relative_unit_metrics = {}

    unit_locations = data.unit_locations

    unit_location_1 = unit_locations[unit_id_1]
    unit_location_2 = unit_locations[unit_id_2]

    relative_unit_metrics['separation'] = l2_metric(
        unit_location_1, unit_location_2)

    total_samples = np.sum(list(data.rec_samples.values()))
    combined_spikes = np.sort(np.concatenate(
        [data.spikes[unit_id_1], data.spikes[unit_id_2]]))

    relative_unit_metrics['combined_firing_range'] = compute_firing_range(
        combined_spikes, total_samples)

    relative_unit_metrics['combined_isi_violations'] = compute_contamination(
        combined_spikes, total_samples)

    relative_unit_metrics['template_similarity'] = data.template_similarity[unit_id_1, unit_id_2]

    binned_spikes_1, _ = np.histogram(data.spikes[unit_id_1], bins=6)
    binned_spikes_2, _ = np.histogram(data.spikes[unit_id_2], bins=6)

    for a, binned_spikes in enumerate([binned_spikes_1, binned_spikes_2]):
        for b, bin in enumerate(binned_spikes):
            relative_unit_metrics[f'spikes_{a}_bin_{b}'] = bin

    return relative_unit_metrics


def compute_firing_range(spikes, total_samples, percentiles=(5, 95), bins=60):

    bin_size_s = (total_samples / 30_000)/bins

    spike_counts, _ = np.histogram(spikes, bins=bins)
    firing_rates = spike_counts / bin_size_s

    # finally we compute the percentiles
    firing_range = np.percentile(
        firing_rates, percentiles[1]) - np.percentile(firing_rates, percentiles[0])

    return firing_range


def compute_contamination(spikes, total_samples, isi_threshold_s=0.0015):

    total_duration_s = total_samples/30_000
    isi_threshold_samples = round(isi_threshold_s*30_000)

    isis = np.diff(spikes)
    num_violations = np.sum(isis < isi_threshold_samples)
    num_spikes = len(spikes)

    violation_time = 2 * num_spikes * isi_threshold_s

    total_rate = num_spikes / total_duration_s
    violation_rate = num_violations / violation_time
    isi_violations_ratio = violation_rate / total_rate
    isi_violations_rate = num_violations / total_duration_s

    return isi_violations_rate
