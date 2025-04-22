"""
    Controls the visualisation. All the GUI stuff!
"""

import pandas as pd
import sys
from pathlib import Path
import PyQt6.QtWidgets as QtWidgets
import numpy as np
import pyqtgraph as pg
import spikeinterface.full as si


from similarity import get_similar_units
from wrangle import DataForGUI
from curate import get_good_units, get_outlier_units
from metrics import compute_metrics, qm_metrics_list, tm_metrics_list

pg.setConfigOption('background', 'w')



unit_1_color = (78, 121, 167)
unit_2_color = (242, 142, 43)



mouse_days = {
    20: [20],
    21: [21],
    22: [36],
    25: [19],
    26: [14],
    27: [23],
    28: [22],
    29: [20],
}




def load_sa_and_extensions(analyzer_path):
    """Loads the sorting analyzer and it's extensions"""

    print("\nLoading sorting analyzer...")
    have_extension = {}
    sorting_analyzer = si.load_sorting_analyzer(analyzer_path, load_extensions=False)
    missing_an_extension = False
    for extension in ['correlograms', 'unit_locations', 'templates', 'spike_amplitudes', 'spike_locations', 'quality_metrics', 'template_metrics']:
        have_extension[extension] = True
        try:
            sorting_analyzer.load_extension(extension)
        except:
            if missing_an_extension is False:
                print("")
            missing_an_extension = True
            have_extension[extension] = False
            print(f"    - No {extension} found. Will not display certain plots.")
    if missing_an_extension:
        print("")

    return sorting_analyzer, have_extension


def main():

    mouse = 22
    day = 36

    analyzer_path = Path(f"/home/nolanlab/Work/Harry_Project/derivatives/M{mouse}/D{day}/full/kilosort4_sa/")
    save_folder = analyzer_path / "merge_info/"
    save_folder.mkdir(exist_ok=True)

    sorting_analyzer, have_extension = load_sa_and_extensions(analyzer_path)

    #sorting_analyzer.compute({"template_metrics": {"include_multi_channel_metrics": True}})
    app = QtWidgets.QApplication(sys.argv)

    rec_samples = pd.read_csv("/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/labels/all_rec_samples_of_vr_of.csv")
    me_rec = rec_samples.query(f"mouse == {mouse} & day == {day}")[['of1', 'vr', 'of2']]
    rec_samples = dict(zip(list(me_rec.keys()), list(me_rec.values[0])))

    window = MainWindow(sorting_analyzer, have_extension, rec_samples, save_folder)

    window.resize(1600, 800)
    window.show()

    sys.exit(app.exec())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sorting_analyzer, have_extension, rec_samples, save_folder):

        self.have_extension = have_extension
        self.data = DataForGUI(sorting_analyzer, have_extension, rec_samples)
        self.save_folder = save_folder
        
        self.decision_counter = 0

        good_units = list(get_good_units(sorting_analyzer).index)
        outlier_units = get_outlier_units(
            self.data.spikes, self.data.rec_samples)
        good_and_outlier_units = set(outlier_units).intersection(good_units)
        self.outlier_ids = np.sort(np.array(list(good_and_outlier_units)))
        #self.outlier_ids = outlier_units

        print(f"{len(good_units)} good units.")
        print(f"{len(self.outlier_ids)} outlier units.")

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
        self.locations_widget = pg.PlotWidget(self)
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

        layout.addWidget(self.text_widget, 0, 0)
        layout.addWidget(self.locations_widget, 1, 0)
        layout.addWidget(self.unit_locations_widget, 0, 1)
        layout.addWidget(self.binned_spikes_widget, 1, 1)

        layout.addWidget(self.correlogram_11_widget, 2, 3)
        layout.addWidget(self.correlogram_12_widget, 2, 4)
        layout.addWidget(self.correlogram_21_widget, 3, 3)
        layout.addWidget(self.correlogram_22_widget, 3, 4)

        print("Starting plot...")

        self.initialise_plot()

        ############### Go go go! ###############

        self.initialise_choice_df()
        self.setCentralWidget(widget)

    def initialise_plot(self):

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
        
        self.unit_locations_widget.setLabels(
            title="Unit locations on probe", bottom="x position (microns)", left="y position (microns)")

        self.templates_1_plot = self.template_widget.plot(
            pen=pg.mkPen(unit_1_color, width=3))
        self.templates_2_plot = self.template_widget.plot(
            pen=pg.mkPen(unit_2_color, width=3))
        self.template_widget.setLabels(
            title="Templates on max channel", bottom="time (ms)", left="Signal (mV)")

        self.amps_plot_1 = self.amps_1_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_1_color, symbolSize=4)
        self.amps_plot_2 = self.amps_2_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_2_color, symbolSize=4)
        self.amps_1_widget.setLabels(
            title=f"Amplitudes of unit {self.unit_id_1}", bottom="time (s)", left="Max amp per spike (mV)")
        self.amps_2_widget.setLabels(
            title=f"Amplitudes of unit {self.unit_id_2}", bottom="time (s)", left="Max amp per spike (mV)")
        
        self.locs_raster_plot_1 = self.locations_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_1_color, symbolSize=4)
        self.locs_raster_plot_2 = self.locations_widget.plot(
            pen=None, symbolPen=None, symbol="o", symbolBrush=unit_2_color, symbolSize=4)
        self.locations_widget.setLabels(
            title="Location of spikes", bottom="x-location (um)", left="y-location (um)")

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
        self.pca_11_widget.setLabels(title=f"PCA 0 vs 1")
        self.pca_12_widget.setLabels(title=f"PCA 1 vs 2")
        self.pca_21_widget.setLabels(title=f"PCA 0 vs 2")
        self.pca_22_widget.setLabels(title=f"PCA 0 vs 3")


        self.correlogram_11_plot = self.correlogram_11_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_1_color)
        self.correlogram_12_plot = self.correlogram_12_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
        self.correlogram_21_plot = self.correlogram_21_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
        self.correlogram_22_plot = self.correlogram_22_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_2_color)
        self.correlogram_11_widget.setLabels(title=f"Auto correlogram for unit {self.unit_id_1}")
        self.correlogram_12_widget.setLabels(title=f"Cross-correlogram")
        self.correlogram_21_widget.setLabels(title=f"Auto correlogram, if units were merged")
        self.correlogram_22_widget.setLabels(title=f"Auto correlogram for unit {self.unit_id_2}")

        self.binned_spikes_plot_2 = self.binned_spikes_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_2_color)
        self.binned_spikes_plot_1 = self.binned_spikes_widget.plot(
            stepMode="left", fillLevel=0, fillOutline=True, brush=unit_1_color)
        self.binned_spikes_widget.setLabels(title=f"Binned spikes for both units, added together")


        self.all_templates_1_plot = self.all_templates_widget.plot(
            pen=pg.mkPen(unit_1_color, width=2))
        self.all_templates_2_plot = self.all_templates_widget.plot(
            pen=pg.mkPen(unit_2_color, width=2))
        self.all_templates_widget.setLabels(title=f"All templates")


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
                (<strong>j</strong>)olt back   {self.strike_merged(previous_id_1)} -- <strong>{self.unit_id_1}</strong> -- {self.strike_merged(next_id_1)}   (<strong>k</strong>)ontinue
            </p>
            <p style='color: rgb({unit_2_color[0]}, {unit_2_color[1]}, {unit_2_color[2]})'>
                (<strong>a</strong>)nti-skip   {previous_id_2} -- <strong>{self.unit_id_2}</strong> -- {next_id_2}   (<strong>s</strong>)kip
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

        self.amps_plot_1.setData(unit_data['rand_spike_1'], unit_data['amp_1'])
        self.amps_plot_2.setData(unit_data['rand_spike_2'], unit_data['amp_2'])

        if self.have_extension["spike_locations"]:
            self.locs_raster_plot_1.setData(
                unit_data['locs_x_1'], unit_data['locs_y_1'])
            self.locs_raster_plot_2.setData(
                unit_data['locs_x_2'], unit_data['locs_y_2'])

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
        
        self.amps_1_widget.setLabels(
            title=f"Amplitudes of unit {self.unit_id_1}", bottom="time (s)", left="Max amp per spike (mV)")
        self.amps_2_widget.setLabels(
            title=f"Amplitudes of unit {self.unit_id_2}", bottom="time (s)", left="Max amp per spike (mV)")
        
        self.correlogram_11_widget.setLabels(title=f"Auto correlogram for unit {self.unit_id_1}")
        self.correlogram_22_widget.setLabels(title=f"Auto correlogram for unit {self.unit_id_2}")

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

        self.all_templates_widget.enableAutoRange()
        # UPDATE VIEW!!!

    # Key presses!!
    def keyPressEvent(self, event):  # Checks if a specific key was pressed

        keystroke = event.text()

        if keystroke == "m":
            self.data.merge_data(self.unit_id_1, self.unit_id_2)
            self.id_2_tracker = 1
            self.possible_units = get_similar_units(
                self.data.template_similarity, self.data.unit_ids, self.unit_id_1, self.data.merged_units)
        elif keystroke == "k":
            self.id_1_tracker += 1
            self.id_2_tracker = 1
        elif keystroke == "j":
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

        if keystroke in ["j", "k"]:
            if keystroke == "k":
                direction = 1
            elif keystroke == "j":
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

        unit_data = self.data.get_unit_data(self.unit_id_1, self.unit_id_2)
        self.metrics = compute_metrics(
            self.data, self.unit_id_1, self.unit_id_2)

        self.update_plot(unit_data, self.metrics)

    def initialise_choice_df(self):
        string_to_write = "index,keystroke,unit_id_1,unit_id_2"
        for key in self.metrics.keys():
            string_to_write += f",{key}"
        string_to_write += "\n"

        with open(self.save_folder / "decision_data.csv", 'w') as decision_file:
            decision_file.write(string_to_write)

    def save_choice(self, keystroke):

        string_to_write = f"{self.decision_counter},{keystroke},{self.unit_id_1},{self.unit_id_2}"
        for values in self.metrics.values():
            string_to_write += f",{values}"
        string_to_write += "\n"

        with open(self.save_folder / "decision_data.csv", 'a') as decision_file:
            decision_file.write(string_to_write)

        self.decision_counter += 1


if __name__ == '__main__':
    main()
