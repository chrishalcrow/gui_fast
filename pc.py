import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from sklearn.decomposition import IncrementalPCA, PCA
import spikeinterface.full as si


def get_closest_channels(sorting_analyzer, channel_index):
    channel_locations = sorting_analyzer.get_channel_locations()
    extremum_channel_location = channel_locations[channel_index]
    relative_channel_locations = channel_locations - extremum_channel_location 
    channel_distances = [norm(relative_channel_location) for relative_channel_location in relative_channel_locations]
    sorted_args = np.argsort(channel_distances)

    return sorted_args[:4]

sa_path = "/home/nolanlab/Work/Projects/MotionCorrect/correct_ks_sa"
sorting_analyzer = si.load_sorting_analyzer(sa_path, load_extensions=False)

waveforms = sorting_analyzer.get_extension("waveforms")


unit_id_1 = 108
unit_id_2 = 102

waveforms_1 = waveforms.get_waveforms_one_unit(unit_id=unit_id_1, force_dense=True)
waveforms_2 = waveforms.get_waveforms_one_unit(unit_id=unit_id_2, force_dense=True)

extremum_channels = si.get_template_extremum_channel(sorting_analyzer)
extremum_channel = sorting_analyzer.recording.ids_to_indices([extremum_channels[unit_id_1]])

pca_channels = get_closest_channels(sorting_analyzer, extremum_channel)

pcas = []

for pca_channel in pca_channels:

    pca_model = PCA(n_components=2, whiten=True)

    waveforms_one_channel_1 = waveforms_1[:,:,pca_channel]
    waveforms_one_channel_2 = waveforms_2[:,:,pca_channel]

    waveforms_one_channel = np.concatenate([waveforms_one_channel_1, waveforms_one_channel_2])

    pca_model.fit(waveforms_one_channel)

    pcas_1 = pca_model.transform(waveforms_one_channel_1)
    pcas_2 = pca_model.transform(waveforms_one_channel_2)

    pcas.append( [pcas_1, pcas_2] )

fig, axes = plt.subplots(2,2)
loop_axes = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

for pca, ax in zip(pcas, loop_axes):
    
    ax.scatter(pca[0][:,0], pca[0][:,1])
    ax.scatter(pca[1][:,0], pca[1][:,1])

fig.show()

#pca_model.partial_fit(waveform)


