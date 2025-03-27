import sys
import PyQt6.QtWidgets as QtWidgets
import numpy as np
import pyqtgraph as pg
import spikeinterface.full as si

def main():

    sa_path = "/home/nolanlab/Work/Harry_Project/derivatives/M25/D25/kilosort4_sa"
    print("loading sorting analyzer...")
    sorting_analyzer = si.load_sorting_analyzer(sa_path, load_extensions=False)
    for extension in ['spike_amplitudes', 'correlograms', 'unit_locations', 'templates']:
        sorting_analyzer.load_extension(extension)
    
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(sorting_analyzer)
    window.resize(1600,800)
    window.show()

    sys.exit(app.exec())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sorting_analyzer):

        self.analyzer = sorting_analyzer
        self.unit_id = 0

        ###############   Get data from sorting analyzer ###############

        print("storing amplitudes")
        self.amps = sorting_analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")[0]
        print("storing spikes")
        self.spikes = si.spike_vector_to_spike_trains(
            sorting_analyzer.sorting.to_spike_vector(concatenated=False), unit_ids = sorting_analyzer.unit_ids)[0]
        
        self.sparsity_mask = sorting_analyzer.sparsity.mask
        self.channel_locations = sorting_analyzer.get_channel_locations()
        self.unit_locations = sorting_analyzer.get_extension("unit_locations").get_data()[:,0:2]
        self.unit_xmin = min(self.channel_locations[:,0])
        self.unit_xmax = max(self.channel_locations[:,0])
        self.unit_ymin = min(self.channel_locations[:,1])
        self.unit_ymax = max(self.channel_locations[:,1])

        max_channels = sorting_analyzer.channel_ids_to_indices(
            si.get_template_extremum_channel(sorting_analyzer).values()
        )
        templates_data = sorting_analyzer.get_extension("templates").get_data()
        self.templates = {unit_id: 
            templates_data[unit_id,:,max_channels[sorting_analyzer.sorting.id_to_index(unit_id)]]
            for unit_id in sorting_analyzer.unit_ids }
        self.all_templates = {unit_id: 
            templates_data[unit_id,:,self.sparsity_mask[sorting_analyzer.sorting.id_to_index(unit_id)]]
            for unit_id in sorting_analyzer.unit_ids }

        all_correlograms = sorting_analyzer.get_extension("correlograms").get_data()[0]
        num_bins = np.shape(all_correlograms)[2]
        self.correlograms = all_correlograms[:,:,num_bins//2-1:]

        ############### Intialise widgets and do some layout ###############
        
        super().__init__()

        self.setWindowTitle("QuickCurate")

        layout = QtWidgets.QGridLayout()
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 2)

        self.amps_widget = pg.PlotWidget(self)
        self.template_widget = pg.PlotWidget(self)
        self.correlogram_widget = pg.PlotWidget(self)
        self.unit_locations_widget = pg.PlotWidget(self)
        self.all_templates_widget = pg.PlotWidget(self)

        layout.addWidget(self.amps_widget, 1, 0, 1, 3)
        layout.addWidget(self.template_widget, 0, 2)
        layout.addWidget(self.correlogram_widget, 0, 1)
        layout.addWidget(self.unit_locations_widget, 0, 0) 
        layout.addWidget(self.all_templates_widget, 0, 3, 2, 1) 

        self.initialise_plot()
        
        ############### Go go go! ###############

        self.setCentralWidget(widget)

    def get_unit_data(self):

        unit_data = {}

        unit_data['amp'] = self.amps[self.unit_id]
        unit_data['spike'] = self.spikes[self.unit_id]
        unit_data['template'] = self.templates[self.unit_id]
        unit_data['correlogram'] = self.correlograms[self.unit_id][self.unit_id]
        unit_data['unit_location'] = self.unit_locations[self.unit_id]
        unit_data['all_template'] = self.all_templates[self.unit_id]

        return unit_data


    def keyPressEvent(self, event): # Checks if a specific key was pressed
        if event.text() == "n":
            self.unit_id = int(self.unit_id) + 1
            self.unit_id_updated()
        if event.text() == "u":
            self.unit_id = int(self.unit_id) - 1
            self.unit_id_updated()


    def unit_id_updated(self):
        
        self.unit_id += 1
        unit_data = self.get_unit_data()

        self.update_plot(unit_data)


    def initialise_plot(self):

        self.unit_locations_widget.setXRange(self.unit_xmin, self.unit_xmax)
        self.unit_locations_widget.setYRange(self.unit_ymin, self.unit_ymax)
        self.unit_locations_plot_1 = self.unit_locations_widget.plot(self.channel_locations, pen=None, symbol="s", symbolSize=6)
        self.unit_locations_plot_2 = self.unit_locations_widget.plot(self.unit_locations, pen=None, symbol="o", symbolSize=6, symbolBrush=(50,200,200,200))
        self.unit_locations_plot_3 = self.unit_locations_widget.plot(symbol="x", symbolSize=20, symbolBrush=(0,0,0))

        self.templates_plot = self.template_widget.plot()
        self.amps_plot = self.amps_widget.plot(pen=None, symbol="x", symbolBrush="r")
        self.correlogram_plot = self.correlogram_widget.plot(stepMode="left", fillLevel=0, fillOutline=True, brush=(0,0,255,150))
        self.all_templates_plot = self.all_templates_widget.plot()

        unit_data = self.get_unit_data()
        
        self.update_plot(unit_data)


    def update_plot(self, unit_data):
        channel_locations = self.channel_locations
        self.amps_plot.setData(unit_data['spike'], unit_data['amp'])
        self.templates_plot.setData(unit_data['template'])
        self.correlogram_plot.setData(np.arange(0,len(unit_data['correlogram']),1), unit_data['correlogram'])
        self.unit_locations_plot_3.setData([unit_data['unit_location'][0]],[unit_data['unit_location'][1]], pen=None, symbol="x", symbolSize=20, symbolBrush=(0,0,0))
        self.update_template_plot(channel_locations, unit_data['all_template'])
        
            
    def update_template_plot(self, channel_locations, all_template):
        self.all_templates_widget.clear()
        template_channels_locs = channel_locations[self.sparsity_mask[self.unit_id]]
        for template_index, template_channel_loc in enumerate(template_channels_locs):            
            curve = pg.PlotCurveItem(4*template_channel_loc[0] + np.arange(90), template_channel_loc[1]/1 + all_template[template_index,:])
            self.all_templates_widget.addItem(curve)


if __name__ == '__main__':
    main()