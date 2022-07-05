import Algorithm as al
import pandas as pd
import PySimpleGUI as sg
def test(parameter,par2):
    df = pd.DataFrame(columns = ['parameter C1','parameter C2','parameter W','parameter CR','parameter MR','fitness' ,'sequence'])
    #print(parameter)
    for c1 in parameter:
        for c2 in parameter:
            for w in parameter:
                for cr in parameter:
                    for mr in parameter:
                        #rint(c1)
                        al.Particle.CROSSOVER_PROB = float(cr)
                        al.Particle.MUTATION_PROB = float(mr)
                        al.Particle.W = float(w)
                        al.Particle.C1 = float(c1)
                        al.Particle.C2 = float(c2)
                        al.algorithm(int(par2[0]), int(par2[1]))
                        gen = al.algorithm.gen
                        seq = al.algorithm.best_x
                        fitt = al.algorithm.min
                        layout = [
                                     [sg.Text("Generation number: :"), sg.Text(gen)],
                                     [sg.Text("Best Value :"), sg.Text(seq)],
                                     [sg.Text("Best Sequence  :"), sg.Text(fitt)]
                                 ]
                        window = sg.Window("Result", layout)
                        '''window["-GEN-"].update(gen)
                        window["-FIT-"].update(fitt)
                        window["-SEQ-"].update(seq)'''
                        while True:
                            event, values = window.Read(timeout=1000 * 5)  # in milliseconds
                            if event in ('__TIMEOUT__',):
                                # print('timed execution inside event loop')
                                # sg.popup_auto_close('Timeout')
                                break

                            if event in (sg.WIN_CLOSED, 'Exit'):
                                break
                        window.close()
                        df = df.append({'parameter C1' : c1,'parameter C2' : c2,'parameter W' : w,'parameter CR' : cr,'parameter MR' : mr,'fitness' :  fitt,'sequence' : seq},ignore_index= True)
    minimun = df['fitness'].max()
    print(minimun)
    position = df['fitness'].idxmax()
    layout_fin = [
                 [sg.Text("C1   :"), sg.Text(df['parameter C1'][position])],
                 [sg.Text("C2   :"), sg.Text(df['parameter C2'][position])],
                 [sg.Text("W    :"), sg.Text(df['parameter W'][position])],
                 [sg.Text("CR   :"), sg.Text(df['parameter CR'][position])],
                 [sg.Text("MR   :"), sg.Text(df['parameter MR'][position])],
                 [sg.Text("Best Value :"), sg.Text(df['fitness'][position])],
                 [sg.Text("Best Sequence  :"), sg.Text(df['sequence'][position])]
             ], [
                 [sg.Button("Show Gantt Chart"),sg.Exit()]
             ]
    window = sg.Window("Result", layout_fin)
    while True:
        event, values = window.Read(timeout=1000 * 3)  # in milliseconds
        if event in ('__TIMEOUT__',):
            # print('timed execution inside event loop')
            sg.popup_auto_close('Timeout')
            break
        if event == "Show Gantt Chart":
            window.close()
            al.gant_chart(df['sequence'][position])
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
    window.close()
    #print(minimun)
    #print(position)
    #print(df['sequence'][position])
    #print(gen, fitt, seq)
    al.gant_chart(df['sequence'][position])
    text = sg.popup_get_text('Insert File Name :','')
    dict=[text,".csv"]
    title = "".join(dict)
    df.to_csv(title)

def manual(par,par2):
    al.Particle.CROSSOVER_PROB = float(par[3])
    al.Particle.MUTATION_PROB = float(par[4])
    al.Particle.W = float(par[2])
    al.Particle.C1 = float(par[0])
    al.Particle.C2 = float(par[1])
    al.algorithm(int(par2[0]), int(par2[1]))
    gen = al.algorithm.gen
    seq = al.algorithm.best_x
    fitt = al.algorithm.min
    layout = [
                 [sg.Text("Generation number: :"), sg.Text(gen)],
                 [sg.Text("Best Value :"), sg.Text(fitt)],
                 [sg.Text("Best Sequence  :"), sg.Text(seq)]
             ], [
                 [sg.Button("Show Gantt Chart"),sg.Exit()]
             ]
    window = sg.Window("Result", layout)
    '''window["-GEN-"].update(gen)
    window["-FIT-"].update(fitt)
    window["-SEQ-"].update(seq)'''
    while True:
        event, values = window.Read()  # in milliseconds
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == "Show Gantt Chart":
            window.close()
            al.gant_chart(seq)
    window.close()




