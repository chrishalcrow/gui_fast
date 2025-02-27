#!/usr/bin/python

import sys
import PyQt6.QtWidgets as QtWidgets
import numpy as np
import pyqtgraph as pg
import spikeinterface.full as si
from time import perf_counter

def main():

    sa_path = "/Users/chris/Desktop/kilosort4_3.zarr"
    print("loading sorting analyzer...")
    sorting_analyzer = si.load_sorting_analyzer(sa_path, load_extensions=False)
    sorting_analyzer.load_extension("spike_amplitudes")
    sorting_analyzer.load_extension("correlograms")
    sorting_analyzer.load_extension("unit_locations")
    sorting_analyzer.load_extension("templates")

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow(sorting_analyzer)
    window.resize(1200,600)
    window.show()

    sys.exit(app.exec())

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sorting_analyzer):

        self.analyzer = sorting_analyzer
        self.unit_id = 0

        print("storing amplitudes")
        amps = sorting_analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")[0]
        print("storing spikes")
        spikes = si.spike_vector_to_spike_trains(
            sorting_analyzer.sorting.to_spike_vector(concatenated=False), unit_ids = sorting_analyzer.unit_ids)[0]
        templates_data = sorting_analyzer.get_extension("templates").get_data()
        sparsity_mask = sorting_analyzer.sparsity.mask
        channel_locations = sorting_analyzer.get_channel_locations()
        unit_locations = sorting_analyzer.get_extension("unit_locations").get_data()[:,0:2]
        xmin = min(channel_locations[:,0])
        xmax = max(channel_locations[:,0])
        ymin = min(channel_locations[:,1])
        ymax = max(channel_locations[:,1])

        max_channels = sorting_analyzer.channel_ids_to_indices(
            si.get_template_extremum_channel(sorting_analyzer).values()
        )

        self.templates = {unit_id: 
            templates_data[unit_id,:,max_channels[sorting_analyzer.sorting.id_to_index(unit_id)]]
            for unit_id in sorting_analyzer.unit_ids
        }

        self.all_templates = {unit_id: 
            templates_data[unit_id,:,sparsity_mask[sorting_analyzer.sorting.id_to_index(unit_id)]]
            for unit_id in sorting_analyzer.unit_ids
        }

        all_correlograms = sorting_analyzer.get_extension("correlograms").get_data()[0]
        print(np.shape(all_correlograms))
        num_bins = np.shape(all_correlograms)[2]
        self.correlograms = all_correlograms[:,:,num_bins//2-1:]

        self.sparsity_mask = sparsity_mask
        self.spikes = spikes
        self.amps = amps
        self.channel_locations = channel_locations
        self.unit_locations = unit_locations
        self.unit_xmax = xmax
        self.unit_xmin = xmin
        self.unit_ymax = ymax
        self.unit_ymin = ymin
        
        super().__init__()

        self.setWindowTitle("QuickCurate")

        layout = QtWidgets.QGridLayout()

        widget = QtWidgets.QWidget()
        widget.resize(1200, 480)
        widget.setLayout(layout)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 3)
        layout.setRowStretch(2, 3)

        self.unit_id = 0

        btn = QtWidgets.QPushButton('next one?')
        self.pw = pg.PlotWidget(self)
        self.template_widget = pg.PlotWidget(self)
        self.correlogram_widget = pg.PlotWidget(self)
        self.unit_locations_widget = pg.PlotWidget(self)
        self.all_templates_widget = pg.PlotWidget(self)

        # Add widgets to the layout in their proper positions
        layout.addWidget(btn, 0, 0, 1, 3)  # button goes in upper-left
        layout.addWidget(self.pw, 2, 0,1,3)  # plot goes on right side, spanning 3 rows  
        layout.addWidget(self.template_widget, 1, 2)
        layout.addWidget(self.correlogram_widget, 1, 1)  # plot goes on right side, spanning 3 rows  
        layout.addWidget(self.unit_locations_widget, 1, 0) 
        layout.addWidget(self.all_templates_widget, 1, 3,2,1) 

        self.initialise_plot()
        
        btn.clicked.connect(self.unit_id_updated)
        
        self.setCentralWidget(widget)


    def keyPressEvent(self, event): # Checks if a specific key was pressed
        if event.text() == "n":
            self.unit_id = int(self.unit_id) + 1
            self.unit_id_updated()
        if event.text() == "u":
            self.unit_id = int(self.unit_id) - 1
            self.unit_id_updated()

    def unit_id_updated(self):
        
        t1 = perf_counter()
        amp = self.amps[self.unit_id]
        spike = self.spikes[self.unit_id]
        template = self.templates[self.unit_id]
        correlogram = self.correlograms[self.unit_id][self.unit_id]
        unit_location = self.unit_locations[self.unit_id]
        all_template = self.all_templates[self.unit_id]
        t2 = perf_counter()
        print("time to get data: ", t2-t1)
        print(f"Now on unit_id {self.unit_id}")

        self.update_plot(spike, amp, template, correlogram, unit_location, all_template)

    def initialise_plot(self):

        amp = self.amps[self.unit_id]
        spike = self.spikes[self.unit_id]
        template = self.templates[self.unit_id]
        correlogram = self.correlograms[self.unit_id][self.unit_id]
        print(correlogram)
        channel_locations = self.channel_locations
        unit_location = self.unit_locations[self.unit_id]
        all_template = self.all_templates[self.unit_id]
        
        self.templates_plot = self.template_widget.plot(template)

        self.unit_locations_plot_1 = self.unit_locations_widget.plot(channel_locations, pen=None, symbol="s", symbolSize=6)
        self.unit_locations_plot_2 = self.unit_locations_widget.plot(self.unit_locations, pen=None, symbol="o", symbolSize=6, symbolBrush=(50,200,200,200))
        self.unit_locations_plot_3 = self.unit_locations_widget.plot([unit_location[0]], [unit_location[1]], pen=None, symbol="x", symbolSize=20, symbolBrush=(0,0,0))

        self.unit_locations_widget.setXRange(self.unit_xmin, self.unit_xmax)
        self.unit_locations_widget.setYRange(self.unit_ymin, self.unit_ymax)

        self.amps_plot = self.pw.plot(spike, amp, pen=None, symbol="x", symbolBrush="r")
        y=np.arange(0,len(correlogram),1)
        self.correlogram_plot = self.correlogram_widget.plot(y, correlogram, stepMode="left", fillLevel=0, fillOutline=True, brush=(0,0,255,150))

        print(np.shape(all_template))

        template_channels_locs = channel_locations[self.sparsity_mask[self.unit_id]]
        for template_index, template_channel_loc in enumerate(template_channels_locs):            
            curve = pg.PlotCurveItem(4*template_channel_loc[0] + np.arange(90), template_channel_loc[1]/10 + all_template[template_index,:])
            self.all_templates_widget.addItem(curve)


    def update_plot(self, spike, amp, template, correlogram, unit_location, all_template):
        print(correlogram)
        t3 = perf_counter()
        channel_locations = self.channel_locations
        self.amps_plot.setData(spike, amp)
        self.templates_plot.setData(template)
        y=np.arange(0,len(correlogram),1)
        self.correlogram_plot.setData(y, correlogram)
        self.unit_locations_plot_3.setData([unit_location[0]],[unit_location[1]], pen=None, symbol="x", symbolSize=20, symbolBrush=(0,0,0))

        self.all_templates_widget.clear()
        template_channels_locs = channel_locations[self.sparsity_mask[self.unit_id]]
        for template_index, template_channel_loc in enumerate(template_channels_locs):            
            curve = pg.PlotCurveItem(4*template_channel_loc[0] + np.arange(90), template_channel_loc[1]/10 + all_template[template_index,:])
            self.all_templates_widget.addItem(curve)
        t4 = perf_counter()

        print("time to plot: ", t4-t3)
        

if __name__ == '__main__':
    main()