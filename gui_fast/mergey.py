"""
    Controls the visualisation. All the GUI stuff!
"""

import pandas as pd
import sys
import PyQt6.QtWidgets as QtWidgets
import numpy as np
import pyqtgraph as pg
import spikeinterface.full as si
import numpy as np

from curate import get_good_units, get_outlier_units
from similarity import get_similar_units 
from wrangle import DataForGUI

pg.setConfigOption('background', 'w')

save_folder = ""

unit_1_color = (78, 121, 167)
unit_2_color = (242, 142, 43)

def main():

    #    sa_path = "/home/nolanlab/Work/Harry_Project/derivatives/M25/D25/kilosort4_sa"
    sa_path = "/home/nolanlab/Work/Projects/MotionCorrect/correct_ks_sa"
#    sa_path = "/Users/christopherhalcrow/Work/Harry_Project/derivatives/kilosort4_sa"
    print("loading sorting analyzer...")
    sorting_analyzer = si.load_sorting_analyzer(sa_path, load_extensions=False)
    for extension in ['spike_amplitudes', 'correlograms', 'unit_locations', 'templates', 'waveforms', 'template_similarity']:
        sorting_analyzer.load_extension(extension)

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow(sorting_analyzer)

    window.resize(1600, 800)
    window.show()

    sys.exit(app.exec())

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sorting_analyzer):

        self.data = DataForGUI(sorting_analyzer)
        ############### Intialise widgets and do some layout ###############
        self.unit_id_1 = 111
        self.unit_id_2 = 108

        print(self.data)
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

        layout.addWidget(self.template_widget, 0, 0)
        layout.addWidget(self.all_templates_widget, 1, 0, 3, 1)

        layout.addWidget(self.pca_11_widget, 0,1)
        layout.addWidget(self.pca_12_widget, 0,2)
        layout.addWidget(self.pca_21_widget, 1,1)
        layout.addWidget(self.pca_22_widget, 1,2)

        layout.addWidget(self.amps_1_widget, 2, 1, 1, 2)
        layout.addWidget(self.amps_2_widget, 3, 1, 1, 2)

        layout.addWidget(self.text_widget, 0, 3, 1, 2)
        layout.addWidget(self.unit_locations_widget, 1, 3)

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

        unit_data = self.data.get_unit_data()

        self.update_plot(unit_data)

    def compute_comparitive(self):
        self.relative_unit_id = self.unit_id_1 - self.unit_id_2

    def initialise_plot(self):

        self.compute_comparitive()

        self.unit_locations_widget.setXRange(self.data.unit_xmin, self.data.unit_xmax)
        self.unit_locations_widget.setYRange(self.data.unit_ymin, self.data.unit_ymax)

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

        unit_data = self.data.get_unit_data(self.unit_id_1, self.unit_id_2)

        self.update_plot(unit_data)

    def update_plot(self, unit_data):
        
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

        #self.pca_11_plot_1.setData(unit_data['pca_1'][0], unit_data['pca_1'][1]) 
        #self.pca_11_plot_1.setData(unit_data['pca_2'][0], unit_data['pca_2'][1]) 

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
        decision_data = pd.DataFrame()
        decision_data.to_csv(save_folder + "decision_data.csv")

    def save_choice(self):
        with open(save_folder + "decision_data.csv", 'a') as decision_file:
            decision_file.write(
                f"\n{self.unit_id_1},{self.unit_id_2},{self.last_keystroke},{self.relative_unit_id}")


if __name__ == '__main__':
    main()
