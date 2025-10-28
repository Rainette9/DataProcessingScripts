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
            'WIND2': 1.45, # m
            'WIND1': 3.45, # m
            'sonic': 1.9, # m
            'SD': 1.7, # m
            'TH': 2, # m
            'RAD1': 192.5, # m
            'RAD2': 42.5, # m
            'FC': 0, # m 
            'SPC': 0.2 # m
        
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
            'WIND2': 2, # m
            'WIND1': 3, # m
            'sonic': 1.9, # m
            'SD': 1.7, # m
            'TH': 2, # m
            'RAD': 2, # m
            'FC': 0, # m 
            'SPC': 0.2 # m
        
        }
        print("Using 2025 calibration coefficients")
    elif sensor == 'BOTTOM':
        calibration_coefficients = None
        heights = {
            'TH': 5, # m
            'WIND1': 5, # m
            'sonic': 5, # m
            'Tsurf': 1.7, # m
            'SD': 2 # m
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
            'TH1': 10, # m
            'TH2': 14, # m
            'WIND1': 10, # m
            'WIND2': 14, # m
            'sonic': 14, # m
            'Tsurf': 1.7, # m
            'RAD': 14, # m
            'FC1': 5, # m
            'FC2': 14 # m
        }
    elif sensor == 'UPPER':
        calibration_coefficients = {
            'A': 4.76480E3,
            'B': 3.84869E6,
            'C': 3.84869E6,
            'H2O_Zero': 0.7311,
            'H20_Span': 0.9883
        }
        heights = {
            'TH': 26, # m
            'WIND': 26, # m
            'sonic': 26, # m
            'Tsurf': 1.7, # m
            'RAD': 26, # m
            'FC1': 5, # m
            'FC2': 14 # m
        }
    else: 
        calibration_coefficients = None
        heights = None

    print(calibration_coefficients)
    return plim, calibration_coefficients, heights


"""
 Heights:
Upper--------------
DL:25.30
Wind: 26.30
Rad:25.60
TH:25.82
CSAT 25.00
 Lower----------"
DL:13.60
Wind:18.15
Rd: 17.30
Th: 17.40
CSAT: 16.50
Wind: 9.90
Fc2: 9.55
Th:9.55
 Bottom----------"
Wind: 5.60
Rad: 5.00
Th: 5.20
Csat3: 5.00
SPC:4.55
Fc:3.50




NEW

4:15
all you have to add 0.40
4:15
Margin of error +-10cm"""