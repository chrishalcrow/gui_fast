"""
    Controls the visualisation. All the GUI stuff!
"""

import pandas as pd
import sys
import PyQt6.QtWidgets as QtWidgets
import numpy as np
import pyqtgraph as pg
import spikeinterface.full as si

from similarity import get_similar_units
from wrangle import DataForGUI
from curate import get_good_units, get_outlier_units
from metrics import compute_metrics

pg.setConfigOption('background', 'w')

save_folder = ""

unit_1_color = (78, 121, 167)
unit_2_color = (242, 142, 43)


def main():

    #    sa_path = "/home/nolanlab/Work/Harry_Project/derivatives/M25/D25/kilosort4_sa"
    #    sa_path = "/home/nolanlab/Work/Projects/MotionCorrect/correct_ks_sa"
    sa_path = "/Users/christopherhalcrow/Work/Harry_Project/derivatives/kilosort4_sa"
    print("loading sorting analyzer...", end=None)
    sorting_analyzer = si.load_sorting_analyzer(sa_path, load_extensions=False)
    for extension in ['spike_amplitudes', 'correlograms', 'unit_locations', 'templates', 'waveforms', 'template_similarity', 'quality_metrics', 'template_metrics']:
        sorting_analyzer.load_extension(extension)
    print(" done!")

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow(sorting_analyzer)

    window.resize(1600, 800)
    window.show()

    sys.exit(app.exec())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sorting_analyzer):

        self.data = DataForGUI(sorting_analyzer)
        ############### Intialise widgets and do some layout ###############
        self.decision_counter = 0
        good_units = list(get_good_units(sorting_analyzer).index)
        outlier_units = get_outlier_units(
            self.data.spikes, self.data.rec_samples)
        good_and_outlier_units = set(outlier_units).intersection(good_units)
        self.outlier_ids = np.sort(np.array(list(good_and_outlier_units)))

        self.metrics = {}
        self.unit_id_1 = self.outlier_ids[0]

        self.id_1_tracker = 0

        possible_units = get_similar_units(
            self.data.template_similarity, self.data.unit_ids, self.unit_id_1, self.data.merged_units)
        self.possible_units = possible_units
        self.unit_id_2 = self.possible_units[1]
        self.id_2_tracker = 1

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
        self.pca_11_widget = pg.PlotWidget(self)
        self.pca_12_widget = pg.PlotWidget(self)
        self.pca_21_widget = pg.PlotWidget(self)
        self.pca_22_widget = pg.PlotWidget(self)
        self.binned_spikes_widget = pg.PlotWidget(self)

        layout.addWidget(self.template_widget, 0, 2)
        layout.addWidget(self.all_templates_widget, 1, 2, 3, 1)

        layout.addWidget(self.pca_11_widget, 0, 3)
        layout.addWidget(self.pca_12_widget, 0, 4)
        layout.addWidget(self.pca_21_widget, 1, 3)
        layout.addWidget(self.pca_22_widget, 1, 4)

        layout.addWidget(self.amps_1_widget, 2, 0, 1, 2)
        layout.addWidget(self.amps_2_widget, 3, 0, 1, 2)

        layout.addWidget(self.text_widget, 0, 0, 2, 1)
        layout.addWidget(self.unit_locations_widget, 0, 1)
        layout.addWidget(self.binned_spikes_widget, 1, 1)

        layout.addWidget(self.correlogram_11_widget, 2, 3)
        layout.addWidget(self.correlogram_12_widget, 2, 4)
        layout.addWidget(self.correlogram_21_widget, 3, 3)
        layout.addWidget(self.correlogram_22_widget, 3, 4)

        self.initialise_plot()

        ############### Go go go! ###############

        self.initialise_choice_df()
        self.setCentralWidget(widget)

    # Key presses!!
    def keyPressEvent(self, event):  # Checks if a specific key was pressed

        keystroke = event.text()

        if keystroke == "m":
            self.data.merge_data(self.unit_id_1, self.unit_id_2)
            self.id_2_tracker = 1
            self.possible_units = get_similar_units(
                self.data.template_similarity, self.data.unit_ids, self.unit_id_1, self.data.merged_units)
        elif keystroke == "n":
            self.id_1_tracker += 1
            self.id_2_tracker = 1
        elif keystroke == "b":
            self.id_1_tracker -= 1
            if self.id_1_tracker == -1:
                print("You're at the start!!")
                self.id_1_tracker = 0
            self.id_2_tracker = 1
        elif keystroke == "s":
            self.id_2_tracker += 1
        elif keystroke == "a":
            self.id_2_tracker -= 1
            if self.id_2_tracker == 0:
                print("You're at the start!")
                self.id_2_tracker = 1

        if keystroke in ["n", "b"]:
            if keystroke == "n":
                direction = 1
            elif keystroke == "b":
                direction = -1
            self.unit_id_1 = self.outlier_ids[self.id_1_tracker]
            while self.unit_id_1 in self.data.merged_units:
                self.id_1_tracker += direction
                self.unit_id_1 = self.outlier_ids[self.id_1_tracker]

            self.possible_units = get_similar_units(
                self.data.template_similarity, self.data.unit_ids, self.unit_id_1, self.data.merged_units)

        if self.id_2_tracker >= len(self.possible_units):
            print(f"No more match candidates for {self.unit_id_1}")
        else:
            self.unit_id_2 = self.possible_units[self.id_2_tracker]
            self.save_choice(keystroke)
            self.unit_ids_updated()

    def unit_ids_updated(self):

        self.compute_comparitive()

        unit_data = self.data.get_unit_data(self.unit_id_1, self.unit_id_2)
        self.metrics = compute_metrics(
            self.data, self.unit_id_1, self.unit_id_2)

        self.update_plot(unit_data, self.metrics)

    def compute_comparitive(self):
        self.relative_unit_id = self.unit_id_1 - self.unit_id_2

    def initialise_plot(self):

        self.compute_comparitive()

        self.unit_locations_widget.setXRange(
            self.data.unit_xmin, self.data.unit_xmax)
        self.unit_locations_widget.setYRange(
            self.data.unit_ymin, self.data.unit_ymax)

        self.unit_locations_plot_1 = self.unit_locations_widget.plot(
            self.data.channel_locations, pen=None, symbol="s", symbolSize=6)
        self.unit_locations_plot_2 = self.unit_locations_widget.plot(
            self.data.unit_locations, pen=None, symbol="o", symbolSize=6, symbolBrush=(50, 200, 200, 200))
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

        self.pca_11_plot_1 = self.pca_11_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_1_color, symbolSize=4)
        self.pca_11_plot_2 = self.pca_11_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_2_color, symbolSize=4)
        self.pca_12_plot_1 = self.pca_12_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_1_color, symbolSize=4)
        self.pca_12_plot_2 = self.pca_12_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_2_color, symbolSize=4)
        self.pca_21_plot_1 = self.pca_21_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_1_color, symbolSize=4)
        self.pca_21_plot_2 = self.pca_21_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_2_color, symbolSize=4)
        self.pca_22_plot_1 = self.pca_22_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_1_color, symbolSize=4)
        self.pca_22_plot_2 = self.pca_22_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_2_color, symbolSize=4)

        self.correlogram_11_plot = self.correlogram_11_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_1_color)
        self.correlogram_12_plot = self.correlogram_12_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
        self.correlogram_21_plot = self.correlogram_21_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
        self.correlogram_22_plot = self.correlogram_22_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_2_color)

        self.binned_spikes_plot_2 = self.binned_spikes_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_2_color)
        self.binned_spikes_plot_1 = self.binned_spikes_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_1_color)

        self.all_templates_1_plot = self.all_templates_widget.plot(
            pen=pg.mkPen(unit_1_color, width=2))
        self.all_templates_2_plot = self.all_templates_widget.plot(
            pen=pg.mkPen(unit_2_color, width=2))

        unit_data = self.data.get_unit_data(self.unit_id_1, self.unit_id_2)
        self.metrics = compute_metrics(
            self.data, self.unit_id_1, self.unit_id_2)

        self.update_plot(unit_data, self.metrics)

    def strike_merged(self, unit_id):
        if unit_id in self.data.merged_units:
            return f"<s>{unit_id}</s>"
        else:
            return f"{unit_id}"

    def update_plot(self, unit_data, metrics_data):

        if self.id_1_tracker == 0:
            previous_id_1 = None
        else:
            previous_id_1 = self.outlier_ids[self.id_1_tracker-1]

        if self.id_1_tracker == len(self.outlier_ids):
            next_id_1 = None
        else:
            next_id_1 = self.outlier_ids[self.id_1_tracker+1]

        previous_id_2 = self.possible_units[self.id_2_tracker-1]
        if self.id_2_tracker == len(self.possible_units):
            next_id_2 = None
        else:
            next_id_2 = self.possible_units[self.id_2_tracker+1]

        the_text = f"""
            <p style='color: rgb({unit_1_color[0]}, {unit_1_color[1]}, {unit_1_color[2]})'>
                (b)ack   {self.strike_merged(previous_id_1)} -- <strong>{self.unit_id_1}</strong> -- {self.strike_merged(next_id_1)}   (n)ext
            </p>
            <p style='color: rgb({unit_2_color[0]}, {unit_2_color[1]}, {unit_2_color[2]})'>
                (a)nti-skip   {previous_id_2} -- <strong>{self.unit_id_2}</strong> -- {next_id_2}   (s)kip
            </p>
            """

        for metric, data in metrics_data.items():
            the_text += f"{metric}: {data}.<br />"

        text = pg.TextItem(html=the_text)
        self.text_widget.clear()
        self.text_widget.addItem(text)
        text.setPos(0, 0)
        self.text_widget.setXRange(0, 15)
        self.text_widget.setYRange(-1, 0)

        self.amps_plot_1.setData(unit_data['spike_1'], unit_data['amp_1'])
        self.amps_plot_2.setData(unit_data['spike_2'], unit_data['amp_2'])

        self.templates_1_plot.setData(unit_data['template_1'])
        self.templates_2_plot.setData(unit_data['template_2'])

        self.pca_11_plot_1.setData(
            unit_data['pca_1'][:, 0], unit_data['pca_1'][:, 1])
        self.pca_11_plot_2.setData(
            unit_data['pca_2'][:, 0], unit_data['pca_2'][:, 1])
        self.pca_12_plot_1.setData(
            unit_data['pca_1'][:, 0], unit_data['pca_1'][:, 2])
        self.pca_12_plot_2.setData(
            unit_data['pca_2'][:, 0], unit_data['pca_2'][:, 2])
        self.pca_21_plot_1.setData(
            unit_data['pca_1'][:, 1], unit_data['pca_1'][:, 2])
        self.pca_21_plot_2.setData(
            unit_data['pca_2'][:, 1], unit_data['pca_2'][:, 2])
        self.pca_22_plot_1.setData(
            unit_data['pca_1'][:, 2], unit_data['pca_1'][:, 3])
        self.pca_22_plot_2.setData(
            unit_data['pca_2'][:, 2], unit_data['pca_2'][:, 3])

        self.binned_spikes_plot_2.setData(
            unit_data['binned_spikes_1']+unit_data['binned_spikes_2'])
        self.binned_spikes_plot_1.setData(
            unit_data['binned_spikes_1'])

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
            unit_data['channel_locations'], unit_data['all_template_1'], unit_data['all_template_2'])

    def update_template_plot(self, channel_locations, all_template_1, all_template_2):

        template_channels_locs_1 = channel_locations[self.data.sparsity_mask[self.unit_id_1]]
        template_channels_locs_2 = channel_locations[self.data.sparsity_mask[self.unit_id_2]]

        for template_index, template_channel_loc in enumerate(template_channels_locs_1):
            curve = pg.PlotCurveItem(4*template_channel_loc[0] + np.arange(
                90), template_channel_loc[1]/1 + all_template_1[template_index, :], pen=pg.mkPen(unit_2_color, width=2))
            self.all_templates_widget.addItem(curve)

        for template_index, template_channel_loc in enumerate(template_channels_locs_2):
            curve = pg.PlotCurveItem(4*template_channel_loc[0] + np.arange(
                90), template_channel_loc[1]/1 + all_template_2[template_index, :], pen=pg.mkPen(unit_1_color, width=2))
            self.all_templates_widget.addItem(curve)

    def initialise_choice_df(self):
        string_to_write = "index,keystroke,unit_id_1,unit_id_2"
        for key in self.metrics.keys():
            string_to_write += f",{key}"
        string_to_write += "\n"

        with open(save_folder + "decision_data.csv", 'w') as decision_file:
            decision_file.write(string_to_write)

    def save_choice(self, keystroke):

        string_to_write = f"{self.decision_counter},{keystroke},{self.unit_id_1},{self.unit_id_2}"
        for values in self.metrics.values():
            string_to_write += f",{values}"
        string_to_write += "\n"

        with open(save_folder + "decision_data.csv", 'a') as decision_file:
            decision_file.write(string_to_write)

        self.decision_counter += 1


if __name__ == '__main__':
    main()
