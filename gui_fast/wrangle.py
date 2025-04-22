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

    def __init__(self, sorting_analyzer, have_extension, rec_samples):

        self.merged_units = []
        self.sorting_analyzer = sorting_analyzer

        self.unit_ids = deepcopy(sorting_analyzer.unit_ids)

        self.rec_samples = rec_samples
        ###############   Get data from sorting analyzer ###############

        print("caching amplitudes and locations...")

        random_spike_indices = si.random_spikes_selection(sorting_analyzer.sorting, max_spikes_per_unit=3000)
        spike_vector = sorting_analyzer.sorting.to_spike_vector()
        random_spikes = spike_vector[random_spike_indices]
        self.random_spikes = si.spike_vector_to_spike_trains([random_spikes], unit_ids = sorting_analyzer.unit_ids)[0]

        self.amps = {}
        if have_extension['spike_amplitudes']:
            amps = sorting_analyzer.get_extension("spike_amplitudes").get_data()
            random_amps = amps[random_spike_indices]

            for unit_id in sorting_analyzer.unit_ids:
                self.amps[unit_id] = []
            
            for spike, amp in zip(random_spikes, random_amps):
                unit_id = spike['unit_index']
                self.amps[unit_id].append(amp)
        amps = None


        self.locs_x = {}
        self.locs_y = {}
        if have_extension['spike_locations']:
            locs_y = sorting_analyzer.get_extension("spike_locations").get_data()['y']
            locs_x = sorting_analyzer.get_extension("spike_locations").get_data()['x']
            random_locs_x = locs_x[random_spike_indices]
            random_locs_y = locs_y[random_spike_indices]
            
            for unit_id in sorting_analyzer.unit_ids:
                self.locs_x[unit_id] = []
                self.locs_y[unit_id] = []

            for spike, loc_x, loc_y in zip(random_spikes, random_locs_x, random_locs_y):
                unit_id = spike['unit_index']
                self.locs_x[unit_id].append(loc_x)
                self.locs_y[unit_id].append(loc_y)
        #locs = None

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

        unit_data['locs_x_1'] = self.locs_x[unit_index_1]
        unit_data['locs_x_2'] = self.locs_x[unit_index_2]
        unit_data['locs_y_1'] = self.locs_y[unit_index_1]
        unit_data['locs_y_2'] = self.locs_y[unit_index_2]

        unit_data['spike_1'] = self.spikes[unit_index_1]
        unit_data['spike_2'] = self.spikes[unit_index_2]

        unit_data['rand_spike_1'] = self.random_spikes[unit_index_1]
        unit_data['rand_spike_2'] = self.random_spikes[unit_index_2]

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
        
        new_max_templates = (
            self.templates[unit_index_1] + self.templates[unit_id_2])/2
        new_quality_metrics = (
            self.quality_metrics.iloc[unit_id_1].values + self.quality_metrics.iloc[unit_id_2].values)/2
        new_template_metrics = (
            self.template_metrics.iloc[unit_id_1].values + self.template_metrics.iloc[unit_id_2].values)/2
        
        new_spikes = np.sort(np.concatenate(
            [self.spikes[unit_id_1], self.spikes[unit_id_2]]))

        new_random_spikes = np.sort(np.concatenate(
            [self.random_spikes[unit_id_1], self.random_spikes[unit_id_2]]))
        new_locs_x = np.concatenate(
            [self.locs_x[unit_id_1], self.locs_x[unit_id_2]])
        new_locs_y = np.concatenate(
            [self.locs_y[unit_id_1], self.locs_y[unit_id_2]])
        new_amps = np.concatenate([self.amps[unit_id_1], self.amps[unit_id_2]])

        new_rand_spike_indices = np.random.choice(range(0,len(new_random_spikes)), size=min(3000,len(new_random_spikes)), replace=False)

        new_random_spikes=new_random_spikes[new_rand_spike_indices]
        new_locs_x=new_locs_x[new_rand_spike_indices]
        new_locs_y=new_locs_y[new_rand_spike_indices]
        new_amps=new_amps[new_rand_spike_indices]
       

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
        self.random_spikes[unit_id_1] = new_random_spikes
        self.amps[unit_id_1] = new_amps
        self.locs_x[unit_id_1] = new_locs_x
        self.locs_y[unit_id_1] = new_locs_y
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
