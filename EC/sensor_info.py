"""This script defines some constants needed to process data of EC and MO"""
import pandas as pd

sensor = 'SFC'

if sensor == 'SFC':
    calibration_coefficients = {
    'A': 5.49957E3,
    'B': 4.00024E6,
    'C': -1.11280E8,
    'H2O_Zero': 0.8164,
    'H20_Span': 1.0103
}
    height_TA = 4 #m
    height_WIND1 = 4 #m
    height_WIND2 = 4 #m
    height_RH = 4 #m
    
elif sensor == 'B':
    calibration_coefficients = {
    'A': 5.49957E3,
    'B': 4.00024E6,
    'C': -1.11280E8,
    'H2O_Zero': 0.8164,
    'H20_Span': 1.0103
}
elif sensor == 'L':
    calibration_coefficients = {
    'A': 5.49957E3,
    'B': 4.00024E6,
    'C': -1.11280E8,
    'H2O_Zero': 0.8164,
    'H20_Span': 1.0103
}
elif sensor == 'U':
    calibration_coefficients = {
    'A': 5.49957E3,
    'B': 4.00024E6,
    'C': -1.11280E8,
    'H2O_Zero': 0.8164,
    'H20_Span': 1.0103
}

