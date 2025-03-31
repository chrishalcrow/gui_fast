"""
    Wrangling the data needed to construct the GUI
"""
import numpy as np
from copy import deepcopy

import spikeinterface.full as si
from spikeinterface.widgets import unit_locations
from curate import get_outlier_units, get_good_units
from compute import get_concat_waveforms, get_pcs_from_waveforms, get_binned_spikes


class DataForGUI:

    def __init__(self, sorting_analyzer):

        self.merged_units = []
        self.sorting_analyzer = sorting_analyzer

        self.unit_ids = deepcopy(sorting_analyzer.unit_ids)

        self.rec_samples = {"of1": 30007677, "vr": 54217811, "of2": 32232891}
        ###############   Get data from sorting analyzer ###############

        print("caching amplitudes...")
        self.amps = sorting_analyzer.get_extension(
            "spike_amplitudes").get_data(outputs="by_unit")[0]
        print("caching spikes...")
        self.spikes = si.spike_vector_to_spike_trains(
            sorting_analyzer.sorting.to_spike_vector(concatenated=False), unit_ids=sorting_analyzer.unit_ids)[0]
        self.template_similarity = sorting_analyzer.get_extension(
            "template_similarity").get_data()

        self.sparsity_mask = sorting_analyzer.sparsity.mask
        self.channel_locations = sorting_analyzer.get_channel_locations()
        self.unit_locations = sorting_analyzer.get_extension(
            "unit_locations").get_data()[:, 0:2]
        self.unit_xmin = min(self.channel_locations[:, 0])
        self.unit_xmax = max(self.channel_locations[:, 0])
        self.unit_ymin = min(self.channel_locations[:, 1])
        self.unit_ymax = max(self.channel_locations[:, 1])

        sparsity_for_pca = si.compute_sparsity(sorting_analyzer, radius_um=50)

        self.unit_id_to_channel_indices = sparsity_for_pca.unit_id_to_channel_indices
        self.waveforms = sorting_analyzer.get_extension("waveforms")

        max_channels = sorting_analyzer.channel_ids_to_indices(
            si.get_template_extremum_channel(sorting_analyzer).values()
        )
        templates_data = sorting_analyzer.get_extension("templates").get_data()
        self.templates = {unit_id_1:
                          templates_data[unit_id_1, :, max_channels[sorting_analyzer.sorting.id_to_index(
                              unit_id_1)]]
                          for unit_id_1 in sorting_analyzer.unit_ids}
        self.all_templates = {unit_id_1:
                              templates_data[unit_id_1, :, self.sparsity_mask[sorting_analyzer.sorting.id_to_index(
                                  unit_id_1)]]
                              for unit_id_1 in sorting_analyzer.unit_ids}

        self.quality_metrics = sorting_analyzer.get_extension(
            "quality_metrics").get_data().astype('float')
        self.template_metrics = sorting_analyzer.get_extension(
            "template_metrics").get_data().astype('float')

        all_correlograms = sorting_analyzer.get_extension(
            "correlograms").get_data()[0]
        self.correlograms = all_correlograms

    def get_unit_data(self, unit_index_1, unit_index_2):

        unit_data = {}

        unit_data['amp_1'] = self.amps[unit_index_1]
        unit_data['amp_2'] = self.amps[unit_index_2]

        unit_data['spike_1'] = self.spikes[unit_index_1]
        unit_data['spike_2'] = self.spikes[unit_index_2]

        unit_data['template_1'] = self.templates[unit_index_1]
        unit_data['template_2'] = self.templates[unit_index_2]

        unit_data['correlogram_11'] = self.correlograms[unit_index_1][unit_index_1]
        unit_data['correlogram_12'] = self.correlograms[unit_index_1][unit_index_2]
        unit_data['correlogram_22'] = self.correlograms[unit_index_2][unit_index_2]
        unit_data['correlogram_21'] = unit_data['correlogram_11'] + unit_data['correlogram_12'] + \
            unit_data['correlogram_22'] + \
            self.correlograms[unit_index_2][unit_index_1]

        waveforms = get_concat_waveforms(
            self.waveforms, unit_index_1, unit_index_2, self.unit_id_to_channel_indices)
        if waveforms is None:
            unit_data['pca_1'] = np.array([[0, 0, 0, 0]])
            unit_data['pca_2'] = np.array([[0, 0, 0, 0]])
        else:
            unit_data['pca_1'], unit_data['pca_2'] = get_pcs_from_waveforms(
                waveforms[0], waveforms[1])

        unit_data['binned_spikes_1'], unit_data['binned_spikes_2'] = get_binned_spikes(
            unit_data['spike_1'], unit_data['spike_2'])

        unit_data['unit_location_1'] = self.unit_locations[unit_index_1]
        unit_data['unit_location_2'] = self.unit_locations[unit_index_2]

        unit_data['all_template_1'] = self.all_templates[unit_index_1]
        unit_data['all_template_2'] = self.all_templates[unit_index_2]

        unit_data['channel_locations'] = self.channel_locations

        return unit_data

    def merge_data(self, unit_id_1, unit_id_2):

        unit_index_1 = unit_id_1
        unit_index_2 = unit_id_2

        new_unit_location = (
            self.unit_locations[unit_index_1] + self.unit_locations[unit_index_2])/2
        new_spikes = np.sort(np.concatenate(
            [self.spikes[unit_id_1], self.spikes[unit_id_2]]))
        new_max_templates = (
            self.templates[unit_index_1] + self.templates[unit_id_2])/2
        new_quality_metrics = (
            self.quality_metrics.iloc[unit_id_1].values + self.quality_metrics.iloc[unit_id_2].values)/2
        new_template_metrics = (
            self.template_metrics.iloc[unit_id_1].values + self.template_metrics.iloc[unit_id_2].values)/2
        new_amps = np.concatenate([self.amps[unit_id_1], self.amps[unit_id_2]])

        combined_sparsity_mask = self.sparsity_mask[unit_id_1] * \
            self.sparsity_mask[unit_id_2]

        combined_and_1_mask = combined_sparsity_mask[self.sparsity_mask[unit_id_1]]
        combined_and_2_mask = combined_sparsity_mask[self.sparsity_mask[unit_id_2]]

        reduced_template_1 = self.all_templates[unit_index_1][combined_and_1_mask, :]
        reduced_template_2 = self.all_templates[unit_index_2][combined_and_2_mask, :]

        new_all_templates = (reduced_template_1 + reduced_template_2)/2

        self.sparsity_mask[unit_index_1] = combined_sparsity_mask
        self.unit_locations[unit_index_1] = new_unit_location
        self.spikes[unit_id_1] = new_spikes
        self.amps[unit_id_1] = new_amps
        self.all_templates[unit_index_1] = new_all_templates
        self.templates[unit_index_1] = new_max_templates
        self.template_metrics.iloc[unit_index_1] = new_template_metrics
        self.quality_metrics.iloc[unit_index_1] = new_quality_metrics

        self.correlograms[unit_index_1, :] = self.correlograms[unit_index_1,
                                                               :] + self.correlograms[unit_index_2, :]
        self.correlograms[:, unit_index_1] = self.correlograms[:,
                                                               unit_index_1] + self.correlograms[:, unit_index_2]

        self.template_similarity[unit_index_1, :] = (self.template_similarity[unit_index_1,
                                                                              :] + self.template_similarity[unit_index_2, :])/2
        self.template_similarity[:, unit_index_1] = (self.template_similarity[:,
                                                                              unit_index_1] + self.template_similarity[:, unit_index_2])/2

        self.merged_units.append(unit_id_2)
