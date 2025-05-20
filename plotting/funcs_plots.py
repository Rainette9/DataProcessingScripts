

def plot_SFC_slowdata_and_fluxes(slowdata, fluxes_SFC, fluxes_16m, fluxes_26m, sensor, start, end, resample_time='10min'):
    
    fig, ax= plt.subplots(9,1, figsize=(13,18), sharex=True)

    ax[0].plot(slowdata['TA'][start:end].resample(resample_time).mean(), label='TA', color='deepskyblue')
    ax[0].set_ylabel('Temperature [oC]')
    # ax[0].set_ylim(-45, 5)
    ax[0].plot(slowdata['SFTempK'][start:end].resample(resample_time).mean()-273.15, label='TS', color='gold', alpha=0.8)
    ax[0].legend(frameon=False)


    ax[1].plot(convert_RH_liquid_to_ice(slowdata['RH'], slowdata['TA'])[start:end].resample(resample_time).mean(), label='RH', color='deepskyblue')
    ax[1].set_ylabel('RH wrt ice [%]')
    ax[1].legend(frameon=False)
    ax[1].set_ylim(0, 100)

    ax[2].scatter(slowdata.loc[start:end].resample(resample_time).mean().index, slowdata['WD1'][start:end].resample(resample_time).mean(), label='WD1', s=5, color='deepskyblue')
    ax[2].scatter(slowdata.loc[start:end].resample(resample_time).mean().index, slowdata['WD2'][start:end].resample(resample_time).mean(), label='WD2', s=5, color='darkblue')
    ax[2].scatter(fluxes_16m.loc[start:end].resample(resample_time).mean().index, fluxes_16m['wind_dir'].resample(resample_time).mean()[start:end], label='WD_16m', s=5, color='limegreen')
    ax[2].scatter(fluxes_26m.loc[start:end].resample(resample_time).mean().index, fluxes_26m['wind_dir'].resample(resample_time).mean()[start:end], label='WD_26m', s=5, color='gold')

    ax[2].set_ylabel('Wind Direction')
    ax[2].legend(frameon=False)
    ax[2].set_ylim(0, 360)

    ax[3].plot(slowdata['WS1_Avg'][start:end].resample(resample_time).mean(), label='WS1_Avg', color='deepskyblue')
    ax[3].plot(slowdata['WS2_Avg'][start:end].resample(resample_time).mean(), label='WS2_Avg', color='darkblue')
    ax[3].plot(fluxes_16m['wind_speed'].resample(resample_time).mean()[start:end], label='WS_16m', color='limegreen')
    ax[3].plot(fluxes_26m['wind_speed'].resample(resample_time).mean()[start:end], label='WS_26m', color='gold')
    ax[3].set_ylabel('Wind Speed[ms-1]')
    ax[3].legend(frameon=False)
    # ax[3].set_ylim(-1, 30)

    ax[4].plot(-(slowdata['SWdown1']-slowdata['SWup1'])[start:end].resample(resample_time).mean(), label='SW_net1', color='gold')
    ax[4].plot(-(slowdata['LWdown1']-slowdata['LWup1'])[start:end].resample(resample_time).mean(), label='LW_net1', color='limegreen')
    ax[4].plot(-(slowdata['SWdown2']-slowdata['SWup2'])[start:end].resample(resample_time).mean(), label='SW_net2', color='gold', linestyle='dashed', alpha=0.8)
    ax[4].plot(-(slowdata['LWdown2']-slowdata['LWup2'])[start:end].resample(resample_time).mean(), label='LW_net2', color='limegreen', linestyle='dashed', alpha=0.8)
    ax[4].set_ylabel('Net Radiation [Wm-2]')
    ax[4].legend(frameon=False)

    # ax[4].set_ylim(-400, 400)

    # ax[5].plot(-(slowdata['LWdown2']-slowdata['LWup2'])[start:end].resample(resample_time).mean(), label='LW_net2', color='limegreen')
    # ax[5].plot(-(slowdata['SWdown2']-slowdata['SWup2'])[start:end].resample(resample_time).mean(), label='SW_net2', color='gold')
    # ax[5].set_ylabel('Net Radiation [Wm-2]')
    # ax[5].legend(frameon=False)

    ax[5].plot(slowdata['HS_Cor'][start:end].resample(resample_time).mean(), label='HS_Cor', color='deepskyblue')
    ax[5].set_ylabel('HS_Cor [m]')
    ax[5].legend(frameon=False)

    ax[6].plot(slowdata['PF_FC4'][start:end].resample(resample_time).mean(), label='PF_FC4', color='deepskyblue')
    ax[6].set_ylabel('Flowcapt [g/m2/s]')

    ax[7].plot(fluxes_SFC['H'].resample('9min').mean()[start:end], label='H SFC', color='deepskyblue')
    ax[7].plot(fluxes_16m['H'].resample(resample_time).mean()[start:end], label='H 16m', color='limegreen')
    ax[7].plot(fluxes_26m['H'].resample(resample_time).mean()[start:end], label='H 26m', color='gold')
    ax[7].set_ylabel('H [Wm-2]')
    ax[7].set_ylim(-180, 80)

    ax[7].legend(frameon=False)


    ax[8].plot(fluxes_SFC['LE'].resample('9min').mean()[start:end], label='LE SFC', color='deepskyblue')
    ax[8].set_ylabel('LE [Wm-2]')
    ax[8].legend(frameon=False)

    fig.suptitle(f'{sensor} slowdata {start} - {end}', y=0.92, fontsize=16)
    # plt.tight_layout()
    plt.savefig(f'./plots/{sensor}_{start}_slowdata_and_fluxes.png', bbox_inches='tight')
    return fig,ax