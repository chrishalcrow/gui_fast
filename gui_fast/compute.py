import numpy as np
from numpy.linalg import norm

from sklearn.decomposition import IncrementalPCA, PCA
import spikeinterface.full as si


def get_binned_spikes(spikes_1, spikes_2):

    binned_spikes_1, _ = np.histogram(spikes_1, bins=20)
    binned_spikes_2, _ = np.histogram(spikes_2, bins=20)

    return binned_spikes_1, binned_spikes_2


def get_concat_waveforms(waveforms, unit_id_1, unit_id_2, unit_id_to_channel_indices, n_components=2, whiten=False):

    unit_1_channels = unit_id_to_channel_indices[unit_id_1]
    unit_2_channels = unit_id_to_channel_indices[unit_id_2]

    common_ids = np.intersect1d(unit_1_channels, unit_2_channels)
    if len(common_ids) == 0:
        return None

    waveforms_1 = waveforms.get_waveforms_one_unit(
        unit_id=unit_id_1, force_dense=True)[:, :, common_ids]
    waveforms_2 = waveforms.get_waveforms_one_unit(
        unit_id=unit_id_2, force_dense=True)[:, :, common_ids]

    num_waveforms = min(np.shape(waveforms_1)[0], np.shape(waveforms_2)[0])

    waveforms_1_concat = np.array(
        [np.concatenate(waveforms_1[a, :, :]) for a in range(num_waveforms)])
    waveforms_2_concat = np.array(
        [np.concatenate(waveforms_2[a, :, :]) for a in range(num_waveforms)])

    return waveforms_1_concat, waveforms_2_concat


def get_pcs_from_waveforms(waveforms_1, waveforms_2, n_components=4, whiten=True):

    pca_model = IncrementalPCA(n_components=n_components, whiten=whiten)

    waveforms = np.concatenate([waveforms_1, waveforms_2])

    pca_model.fit(waveforms)

    pcas_1 = pca_model.transform(waveforms_1)
    pcas_2 = pca_model.transform(waveforms_2)

    return pcas_1, pcas_2
