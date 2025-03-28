import pandas as pd
import sys
import PyQt6.QtWidgets as QtWidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.functions import mkPen
import spikeinterface.full as si
import numpy as np

pg.setConfigOption('background', 'w')

save_folder = ""

unit_1_color = (78, 121, 167)
unit_2_color = (242, 142, 43)


def get_similar_units(template_similarity, unit_ids, unit_1):
    units_worst_to_best = unit_ids[np.argsort(template_similarity[unit_1, :])]
    similar_units = []
    for unit_index in reversed(units_worst_to_best):
        if template_similarity[unit_1, unit_index] > 0.2:
            similar_units.append(unit_ids[unit_index])
    return similar_units


def get_good_units(analyzer):
    qms = analyzer.get_extension('quality_metrics').get_data()
    good_qms = qms.query(
        "snr > 1 & firing_rate > 0.05 & rp_contamination < 0.2")
    return good_qms


def get_type_of_firing(spike, rec_samples, threshold=0.1):

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


def main():

    #    sa_path = "/home/nolanlab/Work/Harry_Project/derivatives/M25/D25/kilosort4_sa"
    #    sa_path = "/home/nolanlab/Work/Projects/MotionCorrect/correct_ks_sa"
    sa_path = "/Users/christopherhalcrow/Work/Harry_Project/derivatives/kilosort4_sa"
    print("loading sorting analyzer...")
    sorting_analyzer = si.load_sorting_analyzer(sa_path, load_extensions=False)

    for extension in ['spike_amplitudes', 'correlograms', 'unit_locations', 'templates']:
        sorting_analyzer.load_extension(extension)

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow(sorting_analyzer)

    window.resize(1600, 800)
    window.show()

    sys.exit(app.exec())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sorting_analyzer):

        self.unit_ids = sorting_analyzer.unit_ids

        self.rec_samples = {"of1": 30007677, "vr": 54217811, "of2": 32232891}
        ###############   Get data from sorting analyzer ###############

        print("storing amplitudes")
        self.amps = sorting_analyzer.get_extension(
            "spike_amplitudes").get_data(outputs="by_unit")[0]
        print("storing spikes")
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
        # num_bins = np.shape(all_correlograms)[2]
        # self.correlograms = all_correlograms[:,:,num_bins//2-1:]
        self.correlograms = all_correlograms

        ############### Intialise widgets and do some layout ###############

        possible_units = get_similar_units(
            self.template_similarity, self.unit_ids, self.unit_id_1)
        self.possible_units = possible_units
        self.id_2_tracker = 1
        if len(possible_units) == 1:
            self.unit_id_2 = possible_units[0]
        else:
            self.unit_id_2 = possible_units[self.id_2_tracker]

        super().__init__()

        self.setWindowTitle("QuickCurate")

        layout = QtWidgets.QGridLayout()
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        for a in [0, 1, 2, 3, 4]:
            layout.setColumnStretch(a, 1)

        self.text_widget = pg.PlotWidget(self)
        self.amps_1_widget = pg.PlotWidget(self)
        self.amps_2_widget = pg.PlotWidget(self)

        self.template_widget = pg.PlotWidget(self)

        self.correlogram_11_widget = pg.PlotWidget(self)
        self.correlogram_12_widget = pg.PlotWidget(self)
        self.correlogram_21_widget = pg.PlotWidget(self)
        self.correlogram_22_widget = pg.PlotWidget(self)
        self.unit_locations_widget = pg.PlotWidget(self)
        self.all_templates_widget = pg.PlotWidget(self)

        layout.addWidget(self.text_widget, 0, 3, 1, 2)
        layout.addWidget(self.unit_locations_widget, 0, 0)
        layout.addWidget(self.template_widget, 0, 2)
        layout.addWidget(self.all_templates_widget, 1, 2, 2, 1)
        layout.addWidget(self.correlogram_11_widget, 1, 3)
        layout.addWidget(self.correlogram_12_widget, 1, 4)
        layout.addWidget(self.correlogram_21_widget, 2, 3)
        layout.addWidget(self.correlogram_22_widget, 2, 4)
        layout.addWidget(self.amps_1_widget, 1, 0, 1, 2)
        layout.addWidget(self.amps_2_widget, 2, 0, 1, 2)

        self.initialise_plot()

        ############### Go go go! ###############

        self.initialise_choice_df()
        self.setCentralWidget(widget)

    def get_unit_data(self):

        unit_data = {}

        unit_data['amp_1'] = self.amps[self.unit_id_1]
        unit_data['amp_2'] = self.amps[self.unit_id_2]

        unit_data['spike_1'] = self.spikes[self.unit_id_1]
        unit_data['spike_2'] = self.spikes[self.unit_id_2]

        unit_data['template_1'] = self.templates[self.unit_id_1]
        unit_data['template_2'] = self.templates[self.unit_id_2]

        unit_data['correlogram_11'] = self.correlograms[self.unit_id_1][self.unit_id_1]
        unit_data['correlogram_12'] = self.correlograms[self.unit_id_1][self.unit_id_2]
        unit_data['correlogram_21'] = self.correlograms[self.unit_id_2][self.unit_id_1]
        unit_data['correlogram_22'] = self.correlograms[self.unit_id_2][self.unit_id_2]

        unit_data['unit_location_1'] = self.unit_locations[self.unit_id_1]
        unit_data['unit_location_2'] = self.unit_locations[self.unit_id_2]

        unit_data['all_template_1'] = self.all_templates[self.unit_id_1]
        unit_data['all_template_2'] = self.all_templates[self.unit_id_2]

        return unit_data

    def keyPressEvent(self, event):  # Checks if a specific key was pressed
        if event.text() == "n":
            self.last_keystroke = "n"
            self.save_choice()
            self.id_1_tracker = int(self.id_1_tracker) + 1
            self.unit_id_1 = self.outlier_ids[self.id_1_tracker]
            possible_units = get_similar_units(
                self.template_similarity, self.unit_ids, self.unit_id_1)
            self.possible_units = possible_units
            self.id_2_tracker = 1
            if self.id_2_tracker >= len(self.possible_units):
                print(f"No candidates for {self.unit_id_1}")
            else:
                self.unit_id_2 = self.possible_units[self.id_2_tracker]
                self.unit_ids_updated()
        if event.text() == "u":
            self.id_1_tracker = int(self.id_1_tracker) - 1
            self.unit_id_1 = self.outlier_ids[self.id_1_tracker]
            possible_units = get_similar_units(
                self.template_similarity, self.unit_ids, self.unit_id_1)
            self.possible_units = possible_units
            self.id_2_tracker = 1
            self.unit_id_2 = possible_units[self.id_2_tracker]
            self.unit_ids_updated()
        if event.text() == "m":
            self.last_keystroke = "m"
            self.save_choice()
            self.id_2_tracker += 1
            if self.id_2_tracker >= len(self.possible_units):
                print("no more matching units...")
            else:
                self.unit_id_2 = self.possible_units[self.id_2_tracker]
                self.unit_ids_updated()

    def unit_ids_updated(self):

        print("Unit 1:", self.unit_id_1, "Unit 2:", self.unit_id_2)
        self.compute_comparitive()

        # self.unit_id_1 += 1
        unit_data = self.get_unit_data()

        self.update_plot(unit_data)

    def compute_comparitive(self):
        self.relative_unit_id = self.unit_id_1 - self.unit_id_2

    def initialise_plot(self):

        self.compute_comparitive()

        self.unit_locations_widget.setXRange(self.unit_xmin, self.unit_xmax)
        self.unit_locations_widget.setYRange(self.unit_ymin, self.unit_ymax)

        self.unit_locations_plot_1 = self.unit_locations_widget.plot(
            self.channel_locations, pen=None, symbol="s", symbolSize=6)
        self.unit_locations_plot_2 = self.unit_locations_widget.plot(
            self.unit_locations, pen=None, symbol="o", symbolSize=6, symbolBrush=(50, 200, 200, 200))
        self.unit_locations_plot_3 = self.unit_locations_widget.plot(
            symbol="x", symbolSize=20, symbolBrush=unit_1_color)
        self.unit_locations_plot_4 = self.unit_locations_widget.plot(
            symbol="x", symbolSize=20, symbolBrush=unit_2_color)

        self.templates_1_plot = self.template_widget.plot(
            pen=pg.mkPen(unit_1_color, width=3))
        self.templates_2_plot = self.template_widget.plot(
            pen=pg.mkPen(unit_2_color, width=3))

        self.amps_plot_1 = self.amps_1_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_1_color, symbolSize=4)
        self.amps_plot_2 = self.amps_2_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_2_color, symbolSize=4)

        self.correlogram_11_plot = self.correlogram_11_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
        self.correlogram_12_plot = self.correlogram_12_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
        self.correlogram_21_plot = self.correlogram_21_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
        self.correlogram_22_plot = self.correlogram_22_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))

        self.all_templates_1_plot = self.all_templates_widget.plot(
            pen=pg.mkPen(unit_1_color, width=2))
        self.all_templates_2_plot = self.all_templates_widget.plot(
            pen=pg.mkPen(unit_2_color, width=2))

        unit_data = self.get_unit_data()

        self.update_plot(unit_data)

    def update_plot(self, unit_data):
        channel_locations = self.channel_locations

        self.text_widget

        the_text = f"Unit\n{self.unit_id_1}\n{self.unit_id_2}\n"
        the_text += f"Diff: {self.relative_unit_id}"

        text = pg.TextItem(the_text, color=(0, 0, 0))
        self.text_widget.clear()
        self.text_widget.addItem(text)
        text.setPos(0, 0)

        self.amps_plot_1.setData(unit_data['spike_1'], unit_data['amp_1'])
        self.amps_plot_2.setData(unit_data['spike_2'], unit_data['amp_2'])

        self.templates_1_plot.setData(unit_data['template_1'])
        self.templates_2_plot.setData(unit_data['template_2'])

        self.correlogram_11_plot.setData(
            np.arange(0, len(unit_data['correlogram_11']), 1), unit_data['correlogram_11'])
        self.correlogram_12_plot.setData(
            np.arange(0, len(unit_data['correlogram_12']), 1), unit_data['correlogram_12'])
        self.correlogram_21_plot.setData(
            np.arange(0, len(unit_data['correlogram_21']), 1), unit_data['correlogram_21'])
        self.correlogram_22_plot.setData(
            np.arange(0, len(unit_data['correlogram_22']), 1), unit_data['correlogram_22'])

        self.unit_locations_plot_3.setData([unit_data['unit_location_1'][0]], [
                                           unit_data['unit_location_1'][1]])
        self.unit_locations_plot_4.setData([unit_data['unit_location_2'][0]], [
                                           unit_data['unit_location_2'][1]])

        self.all_templates_widget.clear()
        self.update_template_plot(
            channel_locations, unit_data['all_template_1'], unit_data['all_template_2'])

    def update_template_plot(self, channel_locations, all_template_1, all_template_2):

        template_channels_locs_1 = channel_locations[self.sparsity_mask[self.unit_id_1]]
        template_channels_locs_2 = channel_locations[self.sparsity_mask[self.unit_id_2]]

        for template_index, template_channel_loc in enumerate(template_channels_locs_1):
            curve = pg.PlotCurveItem(4*template_channel_loc[0] + np.arange(
                90), template_channel_loc[1]/1 + all_template_1[template_index, :], pen=pg.mkPen(unit_2_color, width=2))
            self.all_templates_widget.addItem(curve)

        for template_index, template_channel_loc in enumerate(template_channels_locs_2):
            curve = pg.PlotCurveItem(4*template_channel_loc[0] + np.arange(
                90), template_channel_loc[1]/1 + all_template_2[template_index, :], pen=pg.mkPen(unit_1_color, width=2))
            self.all_templates_widget.addItem(curve)

    def initialise_choice_df(self):
        decision_data = pd.DataFrame()
        decision_data.to_csv(save_folder + "decision_data.csv")

    def save_choice(self):
        with open(save_folder + "decision_data.csv", 'a') as decision_file:
            decision_file.write(
                f"\n{self.unit_id_1},{self.unit_id_2},{self.last_keystroke},{self.relative_unit_id}")


if __name__ == '__main__':
    main()
