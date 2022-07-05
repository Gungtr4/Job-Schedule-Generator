import pandas as pd
import random
import main
import PySimpleGUI as sg
def open(data_address):
    pt_tmp = pd.read_excel(data_address, sheet_name="Processing Time", index_col=[0])
    ms_tmp = pd.read_excel(data_address, sheet_name="Machines Sequence", index_col=[0])
    dd_tmp = pd.read_excel(data_address, sheet_name="Due Date", index_col=[0])

    open.num_mc = 11 #number of machine
    open.num_job = pt_tmp.shape[0]  # number of jobs
    open.num_gene = pt_tmp.shape[0] * pt_tmp.shape[1]
    #print(num_job)

    open.pt=[list(map(int, pt_tmp.iloc[i])) for i in range(open.num_job)]
    open.ms=[list(map(int,ms_tmp.iloc[i])) for i in range(open.num_job)]
    open.dd=[list(map(int,dd_tmp.iloc[i])) for i in range(open.num_job)]

def calculate_fit(total_chromosome):
        #print(total_chromosome)
        chrom_fitness, chrom_fit = [], []
        j_keys = [j for j in range(open.num_job)]
        key_count = {key: 0 for key in j_keys}
        j_count = {key: 0 for key in j_keys}
        m_keys = [j + 1 for j in range(open.num_mc)]
        m_count = {key: 0 for key in m_keys}
        #print(key_count)
        for i in total_chromosome:
            #print(key_count)
            gen_t = int(open.pt[i][key_count[i]])
            gen_m = int(open.ms[i][key_count[i]])
            j_count[i] = j_count[i] + gen_t
            m_count[gen_m] = m_count[gen_m] + gen_t

            if m_count[gen_m] < j_count[i]:
                    m_count[gen_m] = j_count[i]
            elif m_count[gen_m] > j_count[i]:
                    j_count[i] = m_count[gen_m]

            key_count[i] = key_count[i] + 1

        makespan = max(j_count.values())
        chrom_fitness.append(1 / makespan)
        #chrom_fit.append(makespan)
        #print(chrom_fit)
        return sum(chrom_fitness)
def color(row):
    color.c_dict = {}
    r = lambda: random.randint(0, 255)
    for j in range(open.num_job):
        colors = '#%02X%02X%02X' % (r(), r(), r())
        color.c_dict['Job %s'%(j+1)] = colors
    #print(color.c_dict)
    return color.c_dict[row['Resource']]

def gant_matplotlib(Gbest):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pandas import Timestamp
    import datetime
    j_keys = [j for j in range(open.num_job)]
    key_count = {key: 0 for key in j_keys}
    j_count = {key: 0 for key in j_keys}
    m_keys = [j + 1 for j in range(open.num_mc)]
    m_count = {key: 0 for key in m_keys}
    j_record = {}
    txt = sg.popup_get_date()
    today = datetime.datetime(txt[2],txt[0],txt[1],8)
    #print(today)
    for i in Gbest:
        gen_t = int(open.pt[i][key_count[i]])
        gen_m = int(open.ms[i][key_count[i]])
        j_count[i] = j_count[i] + gen_t
        m_count[gen_m] = m_count[gen_m] + gen_t

        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]

        bil1 = j_count[i] - open.pt[i][key_count[i]]
        bil1d = int(bil1 / 8)
        bil1h = bil1% 8
        bil2 = j_count[i]
        bil2d = int(bil2 / 8)
        bil2h = bil2 % 8

        start_time = Timestamp(today + datetime.timedelta(days=bil1d)+ datetime.timedelta(hours=bil1h))  # convert seconds to hours, minutes and seconds
        end_time = Timestamp(today + datetime.timedelta(days=bil2d)+ datetime.timedelta(hours=bil2h))
        j_record[(i, gen_m)] = [start_time, end_time]

        key_count[i] = key_count[i] + 1

    df = pd.DataFrame()
    for j in j_keys:
        for i in range(3):
            m = int(open.ms[j][i])
            df = df.append(dict(Machine='Machine %s' % (m),Start=j_record[(j, m)][0],
                           Finish=j_record[(j, m)][1], Resource='Job %s' % (j + 1),Task='Task [%s|%s]' %(j,i),Completion=1),ignore_index=True)


    ##### DATA PREP #####
    # project start date
    proj_start = df.Start.min()

    df['color'] = df.apply(color,axis=1)
    # number of days from project start to task start
    df['start_num'] = (df.Start - proj_start).dt.days

    # number of days from project start to end of tasks
    df['end_num'] = (df.Finish - proj_start).dt.days

    # days between start and end of each task
    df['days_start_to_end'] = df.end_num - df.start_num

    # days between start and current progression of each task
    df['current_num'] = (df.days_start_to_end * df.Completion)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    ##### PLOT #####
    fig, (ax, ax1) = plt.subplots(2, figsize=(16, 6), gridspec_kw={'height_ratios': [6, 1]}, facecolor='#36454F')
    ax.set_facecolor('#36454F')
    ax1.set_facecolor('#36454F')
    # bars
    '''ax.barh(df.Task, df.current_num,height= 1, left=df.start_num, color=df.color)
    ax.barh(df.Task, df.days_start_to_end,height= 1, left=df.start_num, color=df.color, alpha=0.5)'''
    ax.barh(df.Task, df.current_num, height=1, left=df.start_num, color=df.color)
    ax.barh(df.Task, df.days_start_to_end, height=1, left=df.start_num, color=df.color, alpha=0.5)

    for idx, row in df.iterrows():
        #ax.text(row.end_num + 0.1, idx, f"{int(row.Completion * 100)}%", va='center', alpha=0.8, color='w')
        ax.text(row.start_num - 0.1, idx, row.Task, va='center', ha='right', alpha=0.8, color='w')

    # grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='k', linestyle='dashed', alpha=0.4, which='both')

    # ticks
    xticks = np.arange(0, df.end_num.max() + 1, 3)
    xticks_labels = pd.date_range(proj_start, end=df.Finish.max()).strftime("%m/%d")
    xticks_minor = np.arange(0, df.end_num.max() + 1, 1)
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(xticks_labels[::3], color='w')
    ax.set_yticks([])

    plt.setp([ax.get_xticklines()], color='w')

    # align x axis
    ax.set_xlim(0, df.end_num.max())

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('w')

    plt.suptitle('PROJECT XYZ', color='w')

    ##### LEGENDS #####
    legend_elements = [Patch(facecolor=color.c_dict[i], label=i) for i in color.c_dict]

    legend = ax1.legend(handles=legend_elements, loc='upper center', ncol=5, frameon=False)
    plt.setp(legend.get_texts(), color='w')

    # clean second axis
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.show()
def gant_matplotlib2(Gbest):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pandas import Timestamp
    import datetime
    j_keys = [j for j in range(open.num_job)]
    key_count = {key: 0 for key in j_keys}
    j_count = {key: 0 for key in j_keys}
    m_keys = [j + 1 for j in range(open.num_mc)]
    m_count = {key: 0 for key in m_keys}
    j_record = {}
    txt = sg.popup_get_date()
    today = datetime.datetime(txt[2],txt[0],txt[1],8)
    #print(today)
    for i in Gbest:
        gen_t = int(open.pt[i][key_count[i]])
        gen_m = int(open.ms[i][key_count[i]])
        j_count[i] = j_count[i] + gen_t
        m_count[gen_m] = m_count[gen_m] + gen_t

        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]

        bil1 = j_count[i] - open.pt[i][key_count[i]]
        bil1d = int(bil1 / 8)
        bil1h = bil1% 8
        bil2 = j_count[i]
        bil2d = int(bil2 / 8)
        bil2h = bil2 % 8

        start_time = Timestamp(today + datetime.timedelta(days=bil1d)+ datetime.timedelta(hours=bil1h))  # convert seconds to hours, minutes and seconds
        end_time = Timestamp(today + datetime.timedelta(days=bil2d)+ datetime.timedelta(hours=bil2h))
        j_record[(i, gen_m)] = [start_time, end_time]

        key_count[i] = key_count[i] + 1

    df = pd.DataFrame()
    for j in j_keys:
        for i in range(3):
            m = int(open.ms[j][i])
            df = df.append(dict(Machine='Machine %s' % (m),Start=j_record[(j, m)][0],
                           Finish=j_record[(j, m)][1], Resource='Job %s' % (j + 1),Task='Task [%s|%s]' %(j,i),Completion=1),ignore_index=True)


    ##### DATA PREP #####
    # project start date
    proj_start = df.Start.min()

    df['color'] = df.apply(color,axis=1)
    # number of days from project start to task start
    df['start_num'] = (df.Start - proj_start).dt.days

    # number of days from project start to end of tasks
    df['end_num'] = (df.Finish - proj_start).dt.days

    # days between start and end of each task
    df['days_start_to_end'] = df.end_num - df.start_num

    # days between start and current progression of each task
    df['current_num'] = (df.days_start_to_end * df.Completion)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    ##### PLOT #####
    fig, ax = plt.subplots(1, figsize=(16, 6))
    ax.set_facecolor('#36454F')
    # bars
    ax.barh(df.Task, df.days_start_to_end, left=df.start_num, color=df.color)

    for idx, row in df.iterrows():
        #ax.text(row.end_num + 0.1, idx, f"{int(row.Completion * 100)}%", va='center', alpha=0.8, color='w')
        ax.text(row.start_num - 0.1, idx, row.Task, va='center', ha='right', alpha=0.8, color='w')

    # grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='k', linestyle='dashed', alpha=0.4, which='both')

    ##### TICKS #####
    xticks = np.arange(0, df.end_num.max() + 1, 3)
    xticks_labels = pd.date_range(proj_start, end=df.End.max()).strftime("%m/%d")
    xticks_minor = np.arange(0, df.end_num.max() + 1, 1)
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(xticks_labels[::3])

    # align x axis
    ax.set_xlim(0, df.end_num.max())

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('w')

    plt.suptitle('PROJECT XYZ', color='w')

    ##### LEGENDS #####
    legend_elements = [Patch(facecolor=color.c_dict[i], label=i) for i in color.c_dict]

    legend = ax1.legend(handles=legend_elements, loc='upper center', ncol=5, frameon=False)
    plt.setp(legend.get_texts(), color='w')

    # clean second axis
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.show()
def gant_chart(G_sequence_best):
    '''--------plot gantt chart-------'''
    import pandas as pd
    # import plotly.plotly as py
    from pandas import Timestamp
    import plotly.express as px
    import plotly.figure_factory as ff
    import datetime

    j_keys = [j for j in range(open.num_job)]
    key_count = {key: 0 for key in j_keys}
    j_count = {key: 0 for key in j_keys}
    m_keys = [j + 1 for j in range(open.num_mc)]
    m_count = {key: 0 for key in m_keys}
    j_record = {}
    txt = sg.popup_get_date()
    today = datetime.datetime(txt[2], txt[0], txt[1], 8)
    # print(today)
    for i in G_sequence_best:
        gen_t = int(open.pt[i][key_count[i]])
        gen_m = int(open.ms[i][key_count[i]])
        j_count[i] = j_count[i] + gen_t
        m_count[gen_m] = m_count[gen_m] + gen_t

        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]

        bil1 = j_count[i] - open.pt[i][key_count[i]]
        bil1d = int(bil1 / 8)
        bil1h = bil1 % 8
        bil2 = j_count[i]
        bil2d = int(bil2 / 8)
        bil2h = bil2 % 8

        start_time = Timestamp(today + datetime.timedelta(days=bil1d) + datetime.timedelta(hours=bil1h))
        end_time = Timestamp(today + datetime.timedelta(days=bil2d) + datetime.timedelta(hours=bil2h))
        j_record[(i, gen_m)] = [start_time, end_time]

        key_count[i] = key_count[i] + 1

    df = pd.DataFrame()
    for j in j_keys:
        for i in range(3):
            m = int(open.ms[j][i])
            df = df.append(dict(Machine='Machine %s' % (m), Start=j_record[(j, m)][0],
                                Finish=j_record[(j, m)][1], Resource='Job %s' % (j + 1), Task='Task [ Job %s|  Machine %s]' % (j, i)), ignore_index=True)
    df.to_csv("tabel1.csv")

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #print(df)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Machine", width=2340, height=1080, color="Resource", text="Resource", title="Job shop Schedule",hover_data=['Machine', 'Start', 'Finish', 'Resource', 'Task'])
    #fig = ff.create_gantt(df,colors="color", index_col='Resource', show_colorbar=True, group_tasks=True,task_names="Task", showgrid_x=True,title='Job shop Schedule', hover_data=['Machine', 'Start', 'Finish', 'Resource', 'Task'])
    #fig.show(fig, filename='GA_job_shop_scheduling', world_readable=True)
    return fig.show()

def valprint():
    nilai = val()
    #print(value)
    obj_val = calculate_fit(nilai)
    layout =[
            [sg.Text("Sequence :"),sg.Text(nilai,key="-VAL-")],
            [sg.Text("Objective Value :"), sg.Text(obj_val)],
            [sg.Button("Show Gantt Chart"), sg.Exit()]
    ]
    window = sg.Window("Validation", layout)
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == "Show Gantt Chart":
            window.close()
            #nil = value["-VAL-"]
            print(nilai)
            gant_chart(nilai)
    window.close()
#open(main.file())
#print())
#)
def val():
    #al.open(file())
    layout =[
            [sg.Text("Sequence :"),sg.InputText(key="-VAL-")],
            [sg.Text("Example : [20,12,10,...]")],
            [sg.Button("submit"), sg.Exit()]
    ]
    window = sg.Window("Validation", layout)
    while True:
        event, value = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == "submit":
            window.close()
            val = value["-VAL-"]
            val_res = val.replace('[', '')
            val_res = val_res.replace(']', '')
            li = list(map(int, val_res.split(",")))
            print(li)
            return li
    window.close()