"""This script defines some info about the sensors used in the EC system."""
import pandas as pd



def get_sensor_info(sensor, year=None):
    plim = pd.DataFrame({
        'abs.u': [40],
        'abs.v': [40],
        'abs.w': [10],
        'Ts.low': [-40],
        'Ts.up': [12],
        # 'pres.low': [5],
        # 'pres.up': [100],
        'h2o.low': [0],
        'h2o.up': [680]
    })

    if sensor == 'SFC' and year == 2024:
        calibration_coefficients = {
            'A': 4.82004E3,
            'B': 3.79290E6,
            'C': -1.15477E8,
            'H2O_Zero': 0.7087,
            'H20_Span': 0.9885
        }
        heights = {
            'WIND2': 1.45,
            'WIND1': 3.45,
            'sonic': 1.9,
            'SD': 1.7,
            'TH': 2,
            'RAD1': 192.5,
            'RAD2': 42.5,
            'FC': 0,
            'SPC': 0.2
        }
        print("Using 2024 calibration coefficients")
    elif sensor == 'SFC' and year == 2025:
        calibration_coefficients = {
            'A': 4.82004E3,
            'B': 3.79290E6,
            'C': -1.15477E8,
            'H2O_Zero': 0.7087,
            'H20_Span': 0.9885
        }
        heights = {
            'WIND2': 2,
            'WIND1': 3,
            'sonic': 1.9,
            'SD': 1.7,
            'TH': 2,
            'RAD': 2,
            'FC': 0,
            'SPC': 0.2
        }
        print("Using 2025 calibration coefficients")
    elif sensor == 'SFC' and year == 2026:
        # Heights from Table 2.2, Jan 13, 2026 (season 2025-2026)
        # LI-7500 75H-1094 (#2) used from Jan 30, 2026; use year=2025 coefficients for data before Jan 30
        calibration_coefficients = {
            'A': 4.95711E3,
            'B': 3.56955E6,
            'C': -8.18829E7,
            'H2O_Zero': 0.8212,
            'H20_Span': 0.9975
        }
        heights = {
            'SPC': 0.45,          # SN500SS
            'sonic': 1.87,        # CSAT3
            'LiCOR': 1.87,        # LI-7500
            'TH': 1.55,           # temperature/humidity
            'WIND2': 1.74,        # Young wind 2 (lower)
            'WIND1': 3.24,        # Young wind 1 (upper)
            'FC': 0.26,           # FlowCapt (lower bar)
            'RAD1': 1.74,         # Radiation SN-500SS-1
            'RAD2': 0.24,         # Radiation SN-500SS-2
            'RAD': 2.26,         # Radiation CS320
            'SD': 1.48,           # SR-50A (snow depth)
            'Tsurf': 1.62         # SI-111 (surface temperature)
        }
        print("Using 2026 calibration coefficients")
    elif sensor == 'BOTTOM' and year == 2025:
        # Heights from Table 2.1, Feb 11, 2025 (season 2024-2025)
        calibration_coefficients = None
        heights = {
            'WIND1': 5.6,
            'TH': 5.2,
            'RAD': 5.0,
            'sonic': 5.0,
            'FC': 4.5,            # Flowcapt (upper bar)
            'SPC': 4.55
        }
    elif sensor == 'BOTTOM' and year == 2026:
        # Heights from Table 2.1, Jan 13, 2026 (season 2025-2026)
        calibration_coefficients = None
        heights = {
            'WIND1': 6.06,
            'TH': 5.58,
            'RAD': 5.17,
            'sonic': 5.17,
            'FC': 3.85,           # Flowcapt (lower bar)
            'SPC': 4.78
        }
    elif sensor == 'BOTTOM':
        calibration_coefficients = None
        heights = {
            'TH': 5,
            'WIND1': 5,
            'sonic': 5,
            'Tsurf': 1.7,
            'SD': 2
        }
    elif sensor == 'LOWER' and year == 2025:
        # Heights from Table 2.1, Feb 11, 2025 (season 2024-2025)
        # LI-7500 75H-1095 (#3)
        calibration_coefficients = {
            'A': 4.76480E3,
            'B': 3.84869E6,
            'C': -1.27838E8,
            'H2O_Zero': 0.7311,
            'H20_Span': 0.9883
        }
        heights = {
            'WIND2': 18.15,       # upper wind
            'RAD': 17.3,
            'TH2': 17.4,          # upper TH
            'sonic': 16.5,        # Sonic & LI-7500
            'FC1': 9.55,          # Flowcapt (lower bar)
            'TH1': 9.55,          # lower TH
            'WIND1': 9.9          # lower wind
        }
    elif sensor == 'LOWER' and year == 2026:
        # Heights from Table 2.1, Jan 13, 2026 (season 2025-2026)
        # LI-7500 75H-1095 (#3)
        calibration_coefficients = {
            'A': 4.76480E3,
            'B': 3.84869E6,
            'C': -1.27838E8,
            'H2O_Zero': 0.7311,
            'H20_Span': 0.9883
        }
        heights = {
            'WIND2': 18.43,       # upper wind
            'RAD': 17.62,
            'TH2': 17.52,         # upper TH
            'sonic': 16.86,       # Sonic & LI-7500
            'FC1': 14.90,         # Flowcapt (lower bar)
            'TH1': 10.82,         # lower TH
            'WIND1': 10.25        # lower wind
        }
    elif sensor == 'LOWER':
        calibration_coefficients = {
            'A': 5.49957E3,
            'B': 4.00024E6,
            'C': -1.11280E8,
            'H2O_Zero': 0.8164,
            'H20_Span': 1.0103
        }
        heights = {
            'TH1': 10,
            'TH2': 14,
            'WIND1': 10,
            'WIND2': 14,
            'sonic': 14,
            'Tsurf': 1.7,
            'RAD': 14,
            'FC1': 5,
            'FC2': 14
        }
    elif sensor == 'UPPER' and year == 2025:
        # Heights from Table 2.1, Feb 11, 2025 (season 2024-2025)
        # LI-7500 75H-1092 (#1), recalibrated 2024-09-19
        calibration_coefficients = {
            'A': 5.50241E3,
            'B': 4.00266E6,
            'C': -1.44890E8,
            'H2O_Zero': 0.8079,
            'H20_Span': 1.0000
        }
        heights = {
            'WIND': 26.3,
            'TH': 25.8,
            'RAD': 25.6,
            'sonic': 25.0         # Sonic & LI-7500
        }
    elif sensor == 'UPPER' and year == 2026:
        # Heights from Table 2.1, Jan 13, 2026 (season 2025-2026)
        # LI-7500 75H-1092 (#1), recalibrated 2024-09-19
        calibration_coefficients = {
            'A': 5.50241E3,
            'B': 4.00266E6,
            'C': -1.44890E8,
            'H2O_Zero': 0.8079,
            'H20_Span': 1.0000
        }
        heights = {
            'WIND': 26.31,
            'TH': 26.13,
            'RAD': 25.77,
            'sonic': 25.37        # Sonic & LI-7500
        }
    elif sensor == 'UPPER':
        calibration_coefficients = {
            'A': 4.76480E3,
            'B': 3.84869E6,
            'C': -1.27838E8,
            'H2O_Zero': 0.7311,
            'H20_Span': 0.9883
        }
        heights = {
            'TH': 26,
            'WIND': 26,
            'sonic': 26,
            'Tsurf': 1.7,
            'RAD': 26,
            'FC1': 5,
            'FC2': 14
        }
    else:
        calibration_coefficients = None
        heights = None

    print(calibration_coefficients)
    return plim, calibration_coefficients, heights
