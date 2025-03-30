"""
    Wrangling the data needed to construct the GUI
"""
import spikeinterface.full as si
from curate import get_outlier_units, get_good_units
from compute import get_concat_waveforms, get_pcs_from_waveforms


class DataForGUI:

    def __init__(self, sorting_analyzer):

        self.unit_ids = sorting_analyzer.unit_ids

        self.rec_samples = {"of1": 30007677, "vr": 54217811, "of2": 32232891}
        ###############   Get data from sorting analyzer ###############

        print("caching amplitudes...")
        self.amps = sorting_analyzer.get_extension(
            "spike_amplitudes").get_data(outputs="by_unit")[0]
        print("caching spikes...")
        self.spikes = si.spike_vector_to_spike_trains(
            sorting_analyzer.sorting.to_spike_vector(concatenated=False), unit_ids=sorting_analyzer.unit_ids)[0]

        good_units = list(get_good_units(sorting_analyzer).index)
        outlier_units = get_outlier_units(self.spikes, self.rec_samples)
        good_and_outlier_units = set(outlier_units).intersection(good_units)
        self.outlier_ids = list(good_and_outlier_units)

        self.outlier_ids = get_outlier_units(self.spikes, self.rec_samples)
        self.id_1_tracker = 1
        self.unit_id_1 = self.outlier_ids[self.id_1_tracker]

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


        self.unit_id_to_channel_indices = sorting_analyzer.sparsity.unit_id_to_channel_indices

        #waveforms = sorting_analyzer.get_extension("waveforms")
        #self.waveforms = {unit_id: waveforms.get_waveforms_one_unit(unit_id = unit_id, force_dense=True) for unit_id in sorting_analyzer.unit_ids}

        
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

        unit_data['correlogram_11'] = self.correlograms[unit_index_1][unit_index_2]
        unit_data['correlogram_12'] = self.correlograms[unit_index_1][unit_index_2]
        unit_data['correlogram_21'] = self.correlograms[unit_index_2][unit_index_1]
        unit_data['correlogram_22'] = self.correlograms[unit_index_2][unit_index_2]

        #waveforms = get_concat_waveforms(self.waveforms, self.unit_id_to_channel_indices, unit_index_1, unit_index_2)
        #unit_data['pca_1'], unit_data['pca_2'] = get_pcs_from_waveforms(waveforms[0], waveforms[1])

        unit_data['unit_location_1'] = self.unit_locations[unit_index_1]
        unit_data['unit_location_2'] = self.unit_locations[unit_index_2]

        unit_data['all_template_1'] = self.all_templates[unit_index_1]
        unit_data['all_template_2'] = self.all_templates[unit_index_2]

        unit_data['channel_locations'] = self.channel_locations

        return unit_data





