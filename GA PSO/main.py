import PySimpleGUI as sg
import os
import pilihan as pl
import validation as val
import numpy as np
import Algorithm as al

def main():
    layout = [
        [sg.Text("Chosse data file :")],
        [sg.Button("Parameter Test", key="test"),sg.Button("Manual Input", key="input"),sg.Button("Validation", key="validation")],
        [sg.Exit()]
    ]
    window = sg.Window("Main Window", layout)
    while True :
        event, value = window.read()
        if event in (sg.WIN_CLOSED,'Exit'):
            break
        if event == "test":
            window.close()
            pl.test(test(),pop())
        elif event == "input":
            window.close()
            pl.manual(input(),pop())
        elif event == "validation":
            window.close()
            val.open(file())
            val.valprint()
    window.close()
def file():
    working_directory = os.getcwd()
    layout = [
            [sg.Text("Chosse data file :")],
            [sg.InputText(key = "-FILE_PATH-"),
            sg.FileBrowse(initial_folder=working_directory, file_types =[("XLSX files","*.xlsx")])],
            [sg.Button("submit"),sg.Exit()]
    ]
    window = sg.Window("File Loader", layout)

    while True :
        event, value =window.read()
        if event in (sg.WIN_CLOSED,'Exit'):
            break
        if event == "submit":
            window.close()
            data_address = value["-FILE_PATH-"]
            return data_address
    window.close()
def test():
    al.open(file())
    layout =[
            [sg.Text("Min value :"),sg.InputText(key="-MIN-")],
            [sg.Text("Max value :"), sg.InputText(key="-MAX-")],
            [sg.Text("Interval  :"), sg.InputText(key="-INTERVAL-")],
            [sg.Button("submit"), sg.Exit()]
    ]
    window = sg.Window("Test", layout)
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == "submit":
            window.close()
            min = value["-MIN-"]
            max = value["-MAX-"]
            int = value["-INTERVAL-"]
            parameter = np.arange(float(min), float(max), float(int)).tolist()
            parameter = np.round(parameter, 2).tolist()
            return parameter
    window.close()
def input():
    al.open(file())
    layout =[
            [sg.Text("C1 :"),sg.InputText(key="-C1-")],
            [sg.Text("C2 :"), sg.InputText(key="-C2-")],
            [sg.Text("Inertia Weight  :"), sg.InputText(key="-W-")],
            [sg.Text("Crossover rate :"), sg.InputText(key="-CR-")],
            [sg.Text("Mutation rate  :"), sg.InputText(key="-MR-")],
            [sg.Button("submit"), sg.Exit()]
    ]
    window = sg.Window("Parameter Input", layout)
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == "submit":
            window.close()
            C1 = value["-C1-"]
            C2 = value["-C2-"]
            W = value["-W-"]
            CR = value["-CR-"]
            MR = value["-MR-"]
            parameter = [C1,C2,W,CR,MR]
            return parameter
    window.close()
def pop():
    layout =[
            [sg.Text("Population Size :"),sg.InputText(key="-POP-")],
            [sg.Text("Generation :"), sg.InputText(key="-GEN-")],
            [sg.Button("submit"), sg.Exit()]
    ]
    window = sg.Window("Initialize Size", layout)
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == "submit":
            window.close()
            POP = value["-POP-"]
            GEN = value["-GEN-"]
            parameter = [POP,GEN]
            return parameter

    window.close()
if __name__ == "__main__":
    main()