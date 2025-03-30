import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from sklearn.decomposition import IncrementalPCA, PCA
import spikeinterface.full as si


def get_closest_channels(sorting_analyzer, channel_index):
    extremum_channel_location = channel_locations[channel_index]
    relative_channel_locations = channel_locations - extremum_channel_location 
    channel_distances = [norm(relative_channel_location) for relative_channel_location in relative_channel_locations]
    sorted_args = np.argsort(channel_distances)

    return sorted_args[:4]

def get_concat_waveforms(waveforms, unit_id_1, unit_id_2, unit_id_to_channel_indices, n_components=2, whiten=True):

    print("getting concat waveforms")

    unit_1_channels = unit_id_to_channel_indices[unit_id_1]
    unit_2_channels = unit_id_to_channel_indices[unit_id_2]

    common_ids = np.intersect1d(unit_1_channels, unit_2_channels)

    waveforms_1 = waveforms.get_waveforms_one_unit(unit_id = unit_id_1, force_dense=True)[:,:,common_ids]
    waveforms_2 = waveforms.get_waveforms_one_unit(unit_id = unit_id_2, force_dense=True)[:,:,common_ids]

    num_waveforms = min(np.shape(waveforms_1)[0], np.shape(waveforms_2)[0])

    waveforms_1_concat = np.array([np.concatenate(waveforms_1[a,:,:]) for a in range(num_waveforms)])
    waveforms_2_concat = np.array([np.concatenate(waveforms_2[a,:,:]) for a in range(num_waveforms)])

    return waveforms_1_concat, waveforms_2_concat



def get_pcs_from_waveforms(waveforms_1, waveforms_2, n_components=4, whiten=True):

    print("computing pcas...")

    pca_model = IncrementalPCA(n_components=n_components, whiten=whiten)

    waveforms = np.concatenate([waveforms_1, waveforms_2])

    pca_model.fit(waveforms)

    pcas_1 = pca_model.transform(waveforms_1)
    pcas_2 = pca_model.transform(waveforms_2)

    return pcas_1, pcas_2



    

def get_pcs_from_analyzer(sorting_analyzer, unit_id_1, unit_id_2, n_components=2, whiten=True):

    waveforms = sorting_analyzer.get_extension("waveforms")

    waveforms_1 = waveforms.get_waveforms_one_unit(unit_id=unit_id_1, force_dense=True)
    waveforms_2 = waveforms.get_waveforms_one_unit(unit_id=unit_id_2, force_dense=True)

    extremum_channels = si.get_template_extremum_channel(sorting_analyzer)
    extremum_channel = sorting_analyzer.recording.ids_to_indices([extremum_channels[unit_id_1]])

    channel_locations = sorting_analyzer.get_channel_locations()
    pca_channels = get_closest_channels(channel_locations, extremum_channel)

    pcas = []

    for pca_channel in pca_channels:

        pca_model = PCA(n_components=n_components, whiten=whiten)

        waveforms_one_channel_1 = waveforms_1[:,:,pca_channel]
        waveforms_one_channel_2 = waveforms_2[:,:,pca_channel]

        waveforms_one_channel = np.concatenate([waveforms_one_channel_1, waveforms_one_channel_2])

        pca_model.fit(waveforms_one_channel)

        pcas_1 = pca_model.transform(waveforms_one_channel_1)
        pcas_2 = pca_model.transform(waveforms_one_channel_2)

        pcas.append( [pcas_1, pcas_2] )
    
    return np.array(pcas)

if __name__ == '__main__':

    sa_path = "/home/nolanlab/Work/Projects/MotionCorrect/correct_ks_sa"
    sorting_analyzer = si.load_sorting_analyzer(sa_path, load_extensions=False)

    waveforms = get_concat_waveforms(sorting_analyzer, 111, 113)

    print(type(waveforms[0]))

    pcas = get_pcs_from_waveforms(waveforms[0], waveforms[1])

    fig, axes = plt.subplots(2,2)
    loop_axes = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    pcs_to_plot = [ [0,1], [0,2], [1,2], [2,3] ]
    
    for ax, pc in zip(loop_axes, pcs_to_plot):

        ax.scatter(pcas[0][:,pc[0]], pcas[0][:,pc[1]])
        ax.scatter(pcas[1][:,pc[0]], pcas[1][:,pc[1]])

        ax.set_title(f"PC{pc[0]} vs PC{pc[1]}")

    
    fig.show()

    #pca_model.partial_fit(waveform)


