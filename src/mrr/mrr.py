import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy import signal

my_dpi = 96

def import_mrr(file_path, var_name, vel_min, vel_max, x_min, x_max, z_min, z_max,
               dish_height, beam_angle, dist_first_gate, time_offset, visu, save_movie):

    if save_movie:
        from matplotlib.animation import FFMpegWriter
        plt.rcParams['animation.ffmpeg_path']='/Users/dujardin/Downloads/ffmpeg-4.1/ffmpeg'
    # MRR parameters
    FS = 500e3  # Sampling rate [Hz]
    LMBDA = 1.238e-2  # Wavelenght [m]

    # Informations about the file content
    dataset = Dataset(file_path)
    n_time = dataset.dimensions['time'].size
    n_range = dataset.dimensions['range'].size
    n_spectral_line = dataset.dimensions['spectrum_n_samples'].size
#    print(dataset.instrument_name)
#    print(dataset.title)
#    print('Number of acquisitions: %d' % n_time)
#    print('Number of range gates: %d' % n_range)
#    print('Number of spectral lines: %d' % n_spectral_line)
#    print('Available data: ' + dataset.field_names)
#    print('Select a variable among: ' + str([i for i in dataset.variables.keys()]))

    # Load the data from the file
    var = dataset.variables[var_name]
    data = np.squeeze(np.asarray(dataset.variables[var_name][:], dtype=np.float32))

    # Retrieve the time of the begining and the end of the acquisition
    time_start = dataset.variables['time_coverage_start'][:]
    time_start = time_start.data[time_start.mask == False]
    tmp = ''
    for i in range(len(time_start)-1):
        tmp = tmp + str(time_start[i])[2:-1]
    time_start = dt.datetime.strptime(tmp, '%Y-%m-%dT%H:%M:%S')

    time_stop = dataset.variables['time_coverage_end'][:]
    time_stop = time_stop.data[time_stop.mask == False]
    tmp = ''
    for i in range(len(time_stop)-1):
        tmp = tmp + str(time_stop[i])[2:-1]
    tmp = '20' + tmp[:-2]  # Mistake in the way Metek wrote the end time
    time_stop = dt.datetime.strptime(tmp, '%Y-%m-%dT%H:%M:%S')

    time_delta = ((time_stop-time_start)/(n_time-1))

    # Offset the time to be correct with UTC
    time_start += dt.timedelta(seconds=time_offset)
    time_stop += dt.timedelta(seconds=time_offset)
    timestamp = [time_start + i*time_delta for i in range(n_time)]

    # Calculation of the distances x, heights z and velocities
    dist = np.asarray(dataset.variables['range'][:], dtype=np.float32)
    dist = dist - dist[0] + dist_first_gate  # MRR software confuses m.a.s.l and m.a.g.l
    z = dist*np.sin(beam_angle*np.pi/180) + dish_height
    x = dist*np.cos(beam_angle*np.pi/180)
    vel_max_MRR = LMBDA * FS / (4*n_range)
    vel = np.linspace(0, vel_max_MRR, n_spectral_line)
    vel = vel*np.cos(beam_angle*np.pi/180)

    # Filter the data accordng to the user parameters
    vel_min_ind = int(np.argwhere(vel >= vel_min)[0])
    vel_max_ind = np.argwhere(vel > vel_max)
    if vel_max_ind.size > 0:
        vel_max_ind = int(vel_max_ind[0])
    else:
        vel_max_ind = vel.shape[0]
    x_min_ind = int(np.argwhere(x >= x_min)[0])
    x_max_ind = np.argwhere(x > x_max)
    if x_max_ind.size > 0:
        x_max_ind = int(x_max_ind[0])
    else:
        x_max_ind = x.shape[0]
    z_min_ind = int(np.argwhere(z >= z_min)[0])
    z_max_ind = np.argwhere(z > z_max)
    if z_max_ind.size > 0:
        z_max_ind = int(z_max_ind[0])
    else:
        z_max_ind = z.shape[0]

    x_min_ind = max(x_min_ind, z_min_ind)
    z_min_ind = x_min_ind
    x_max_ind = min(x_max_ind, z_max_ind)
    z_max_ind = x_max_ind
    x = x[x_min_ind:x_max_ind]
    z = z[z_min_ind:z_max_ind]
    vel = vel[vel_min_ind:vel_max_ind]

    # Time series of velocity based on the highest reflectivity at each time
    if var_name == 'spectrum_raw':
        data_crop = data[:, z_min_ind:z_max_ind, vel_min_ind:vel_max_ind]
        vel_max_reflect = np.zeros((n_time,))
#        max_reflect = np.zeros((n_time,))
        sum_reflect = np.zeros((n_time,))
        x_vel_max = np.zeros((n_time,))
        z_vel_max = np.zeros((n_time,))
        for i_time in range(n_time):
            ind_max_perdist = np.argmax(data_crop[i_time, :, :], axis=1)
            tmp = data_crop[i_time, :, ind_max_perdist]
            tmp = np.diagonal(tmp)
            ind_max = np.argmax(tmp)
            x_vel_max[i_time] = x[ind_max]
            z_vel_max[i_time] = z[ind_max]
            vel_max_reflect[i_time] = vel[ind_max_perdist[ind_max]]
#            max_reflect[i_time] = tmp[ind_max]
            sum_reflect[i_time] = np.sum(data_crop[i_time, :, :])#*(data_crop[i_time, :, :]>10))
        vel_max_reflect[0] = vel_max_reflect[1]
        del data_crop
    elif var_name == 'VEL':
        data_crop = data[:, z_min_ind:z_max_ind]
        vel_max_reflect = np.zeros((n_time,))
        for i_time in range(n_time):
            ind = (data_crop[i_time, :] >= vel_min) & (data_crop[i_time, :] <= vel_max)
            vel_max_reflect[i_time] = np.mean(data_crop[i_time, ind])
        sum_reflect = None
        x_vel_max = None
    else:
        print('Cannot extract velocities: use spectrum_raw or VEL for var_name')
        vel_max_reflect = None
        sum_reflect = None
        x_vel_max = None

    # Plot the data
    if visu:
        if var.ndim == 3:
            if var_name == 'spectrum_raw':
                data_transposed = np.moveaxis(data, 1, 2)
                data_transposed = data_transposed[:, vel_min_ind:vel_max_ind, x_min_ind:x_max_ind]
                data = data[:, z_min_ind:z_max_ind, vel_min_ind:vel_max_ind]

                # Remove the average change of velocity with height
                data_transposed_smooth = signal.convolve(data_transposed, np.ones((1, 4, 1))/4, 'same')  # Smoothing on distance
                ind_max_reflect = np.argmax(data_transposed_smooth, axis=1)  # Instant max vel for each height
                ind_vel_shift = np.asarray(np.round(np.mean(ind_max_reflect, axis=0)), dtype=np.int32)
                ind_vel_shift = ind_vel_shift - int(round(0.5*vel.shape[0]))
                data_transposed_detrend = np.zeros(data_transposed.shape, dtype=np.float32)
                for i in range(ind_vel_shift.shape[0]):
                    data_transposed_detrend[:, :, i] = np.append(data_transposed[:, ind_vel_shift[i]:, i], data_transposed[:, :ind_vel_shift[i], i], axis=1)

                # Remove the average change of velocity with distance
                data_smooth = signal.convolve(data, np.ones((1, 1, 2))/2, 'same')  # Smoothing on height
                data_detrend = np.zeros(data.shape, dtype=np.float32)
                if time_delta.seconds>0:
                    preceding_period = int((x.max()/vel_min)/time_delta.seconds)
                else:
                    preceding_period = int((x.max()/vel_min)/(time_delta.microseconds/1e6))
                for i_time in range(preceding_period, n_time):
                    i_start = int(max(0, i_time - preceding_period))
                    data_previous = data[i_start:i_time, :, :]
                    data_smooth_previous = data_smooth[i_start:i_time, :, :]
                    ind_max_reflect = np.argmax(data_smooth_previous, axis=2)  # Instant max vel for each height
                    for i_z in range(z.shape[0]):
                        dist_from_vel = vel[ind_max_reflect[:, i_z]]*(np.arange(data_previous.shape[0], 0, -1)-1)
                        dist_real = x[i_z]
                        ind_time_shift = np.argmin(np.abs(dist_from_vel - dist_real))
                        data_detrend[i_time, i_z, :] = data_previous[ind_time_shift, i_z, :]

                data_min, data_max = np.min(data), np.max(data)
                fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(1024/my_dpi, 1024/my_dpi), dpi=my_dpi, sharey=False, sharex=False)
                if save_movie:
                    moviewriter = FFMpegWriter(fps=15, codec=None, bitrate=None)
                    moviewriter.setup(fig, file_path.name[:-3] + '.mp4')
                for i in range(n_time):
                    ax1.cla()
                    ax2.cla()
                    ax3.cla()
                    ax4.cla()
                    ax5.cla()
                    ax6.cla()

                    tmp = ax1.pcolormesh(x, vel, data_transposed[i, :, :], clim=(data_min, data_max))
                    tmp = ax1.plot(x_vel_max[i], vel_max_reflect[i], marker='X', color='r')
                    tmp = ax3.pcolormesh(vel, z, data[i, :, :], clim=(data_min, data_max))
                    tmp = ax3.plot(vel_max_reflect[i], z_vel_max[i], marker='X', color='r')
                    tmp = ax2.pcolormesh(x, vel, data_transposed_detrend[i, :, :], clim=(data_min, data_max))
                    tmp = ax4.pcolormesh(vel, z, data_detrend[i, :, :], clim=(data_min, data_max))
                    tmp = ax5.plot(vel_max_reflect)
                    tmp = ax5.plot(i, vel_max_reflect[i], marker='X', color='r')
                    tmp = ax6.plot(z_vel_max)
                    tmp = ax6.plot(i, z_vel_max[i], marker='X', color='r')

                    if i == 0:
                        plt.set_cmap('viridis')
                        tmp = np.linspace(data_min, data_max, 256)
                        im = ax2.pcolormesh(np.repeat(tmp[:, None], 1, axis=1), rasterized=True)
                        fig.colorbar(im, ax=[ax1, ax2, ax3, ax4], shrink=1, label='Reflectivity [dB]')

                    fig.suptitle(str(timestamp[i]) + ' UTC')
                    ax1.set_title('Original data')
                    ax2.set_title('Detrended data')
                    ax1.set_xlabel('Horizontal distance [m]')
                    ax2.set_xlabel('Horizontal distance [m]')
                    ax1.set_ylabel('Horizontal velocity [m/s]')
                    ax3.set_xlabel('Horizontal velocity [m/s]')
                    ax4.set_xlabel('Horizontal velocity [m/s]')
                    ax3.set_ylabel('Height [m.a.g.l]')
                    ax5.set_xlabel('Data point index')
                    ax6.set_xlabel('Data point index')
                    ax5.set_ylabel('Horizontal velocity [m/s]')
                    ax6.set_ylabel('Height [m.a.g.l]')

                    plt.pause(0.01)
                    if save_movie:
                        moviewriter.grab_frame()
                if save_movie:
                    moviewriter.finish()
            else:
                data_min, data_max = np.nanmin(data), np.nanmax(data)
                for i in range(n_time):
                    plt.clf()
                    plt.imshow(data[i, :, :], clim=(data_min, data_max))
                    plt.pause(0.01)

        else:
            data_min, data_max = np.min(data), np.max(data)
            fig = plt.figure()
            for i in range(n_time):
                plt.clf()
                plt.plot(data[i, :])
                plt.ylim(data_min, data_max)
                plt.pause(0.01)


    return vel_max_reflect, x_vel_max, sum_reflect, timestamp
