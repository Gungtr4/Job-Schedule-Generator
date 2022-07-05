import pandas
from pandas import Timestamp
import PySimpleGUI as sg

today = Timestamp (sg.popup_get_date())
print(today)