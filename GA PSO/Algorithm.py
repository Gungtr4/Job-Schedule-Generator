from random import randint
from random import shuffle
from random import random
import random
import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Particle:
    Population_size = 0
    g_best = [4, 20, 21, 6, 2, 19, 18, 25, 20, 8, 24, 18, 1, 23, 12, 1, 22, 26, 26, 12, 3, 8, 4, 9, 0, 16, 14, 2, 22, 16, 9, 21, 1, 14, 21, 24, 3, 0, 14, 15, 10, 9, 7, 20, 11, 13, 17, 5, 7, 5, 6, 25, 10, 15, 23, 12, 19, 4, 18, 0, 6, 13, 11, 23, 7, 15, 8, 3, 22, 5, 25, 13, 17, 17, 16, 2, 11, 10, 26, 24, 19]
    shuffle(g_best)
    g_best = list(g_best)
    #par = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    CROSSOVER_PROB = 0
    MUTATION_PROB = 0
    W = 0
    C1 = 0
    C2 = 0

    def __init__(self):
        def starting_pos():
            """ Helper function for initializing the position of
            each particle
            """
            '''population_list = []
            for i in range(Population_size):
                nxm_random_num = list(np.random.permutation(num_gene))  # generate a random permutation of 0 to num_job*num_mc-1
                population_list.append(nxm_random_num)  # add to the population_list
                for j in range(num_gene):
                    population_list[i][j] = population_list[i][j] % num_job  # convert to job number format, every job appears m times
            return population_list'''
            start = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,11,11,11,12,12,12,13,13,13,14,14,14,15,15,15,16,16,16,17,17,17,18,18,18,19,19,19,20,20,20,21,21,21,22,22,22,23,23,23,24,24,24,25,25,25,26,26,26])
            #for i in range(num_mc)
            shuffle(start)
            par = list(start)
            #print(par)
            #start = [0] + start + [num_job]
            return par

        def starting_vel():
            """ Helper function for initializing the velocity of
            each particle
            """
            '''vel = []
            for i in range(0, num_job):
                vel.append(randint(int(Particle.VMIN), int(Particle.VMAX)))
            vel = [0] + vel + [0]
            return vel'''
            vel = []
            for i in range(open.num_gene):
                vel= np.random.randint(1,10, size=open.num_gene)     #vel = list(np.random.permutation(num_job))
            #print(vel)
            # start = [0] + start + [num_job]
            return vel

        self.v = starting_vel()
        self.x = starting_pos()
        self.p_best = self.x[:]
        self.fit = 0

    def update_velocity(self):
        """Updates the velocity of each dimension in the particle"""
        for i in range(len(self.v)-1):
            #print(self.p_best[i])
            vel = Particle.W * self.v[i] + Particle.C1 * random.random() * (self.p_best[i] - self.x[i]) + Particle.C2 * random.random() * (Particle.g_best[i] - self.x[i])
            #print(vel)
            self.v[i] = vel

    def update_position(self):
        """Updates the position of each dimension in the particle"""
        '''for i in range(1, len(self.x) - 1):        #len(self.x)
            new_pos = self.x[i] + self.v[i] #int(floor((self.x[i] + self.v[i])))
            if new_pos > Particle.XMAX:
                pass
            elif new_pos < Particle.XMIN:
                pass
            else:
                self.x[i] = new_pos'''
        val = []
        for i in range(0, len(self.x)):  # len(self.x)
            temp = [0,0]
            new_pos = float(self.x[i] + self.v[i])  # int(floor((self.x[i] + self.v[i])))
            #print(new_pos)
            temp= [new_pos,self.x[i]]
            val.append(temp)
        val.sort()
        #print(val)
        for j in range(len(val)):
            new_par = val[j][1]
        return  new_par     #self.x[i] = new_pos

    def mutate(self):
        """Changes some parts of x based on mutation probability"""
        #num_mutation_jobs = round(num_gene * 0.2)
        for m in range(0,len(self.x)):  # don't mutate start of goal
            dont_mutate = random.random()
            if Particle.MUTATION_PROB > dont_mutate:
                '''m_chg = list(np.random.choice(num_gene, num_mutation_jobs, replace=False))  # chooses the position to mutation
                t_value_last = self.x[m][m_chg[0]]  # save the value which is on the first mutation position
                for i in range(num_mutation_jobs - 1):
                    self.x[m][m_chg[i]] = self.x[m][m_chg[i + 1]]  # displacement
                self.x[m][m_chg[num_mutation_jobs - 1]] = t_value_last  # move the value of the first mutation position to the last mutation position
                '''
                self.x[m] = randint(0,26)

    def crossover(self, other_particle):
        """Takes two particles and exchanges part of the solution at
        a specific point
        """
        #print(self.x)
        crossover_position = randint(1, len(self.x))
        new1_first_half = self.x[:crossover_position]
        new1_second_half = other_particle.x[crossover_position:]
        #print(new1_first_half)
        #print(new1_second_half)
        temp = new1_first_half + new1_second_half
        new = Particle()
        new.x = temp   #remove_duplicates(temp)
        new.v = self.v
        new.fit = self.fit
        new.p_best = self.p_best
        return new

    def calculate_fit(self,total_chromosome):
        #print(total_chromosome)
        chrom_fitness, chrom_fit = [], []
        '''for m in range(Population_size * 2):
            temp = []'''
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
    #def calculate_tar(self,total_chromosome):

    def remove_duplicates(self):
        """Takes a path and returns it with duplicate nodes removed."""
        job = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        job_count = []
        job_pos = []
        larger = []
        less = []
        for i in range(0, open.num_job):
            job_count.append(self.x.count(i))
        for j in job:
            job_pos.append(list_duplicates_of(self.x, j))

        for i in range(0, open.num_job):
            if job_count[i] >= 3:
                larger.append(i)
            else:
                less.append(i)
        for i in range(0, open.num_job):
            x = 0
            while job_count[i] > 3:
                if not less:
                    break
                else:
                    index = random.choice(less)
                    self.x[job_pos[i][x]] = index
                    job_count[index] = job_count[index] + 1
                    job_count[i] = job_count[i] - 1
                    x += 1
                    if job_count[index] == 3:
                        less.remove(index)
        '''job = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        ser = [i for i in job + self.x if i not in job or i not in self.x]
        if ser:
            res = ['x' if (ele in self.x[:idx]) else ele for idx, ele in enumerate(self.x)]
            for i in range(0, len(res)):
                if (res[i] == 'x'):
                    j = randint(0, len(ser) - 1)
                    res[i] = ser[j]
                    ser.pop(j)
            self.x = res
        else:
            self.x = self.x'''

def instantantiate(pop):
    """Takes the number of particles and returns a swarm of particles"""
    #print('Initializing swarm...')
    swarm = []
    for i in range(0, pop):
        swarm.append(Particle())
    #print('Swarm initialized.')
    return swarm

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def selection(swarm):
    """Performs a selection of particles for the next generation.
    Particles with higher fitness hava a higher probability of being
    selected
    """
    # The following algorithm of selection found in the following page
    # http://arxiv.org/pdf/1308.4675.pdf
    # each step is marked below

    # Probability of each chromosome to be selected
    fit_r = []
    for particle in swarm:
        #print(particle.fit)
        fit_r.append(particle.fit)#(1 / particle.fit)

    # Probability over total probability
    fit_r_sum = sum(fit_r)
    selection_probability = []
    for relative_fit in fit_r:
        selection_probability.append(relative_fit / fit_r_sum)

    # Cumulative probability
    cumulative_probability = []
    the_sum = 0
    for a in selection_probability:
        the_sum += a
        cumulative_probability.append(the_sum)

    # For the new generation, we compare a random number between 0 and 1
    # and we select the particle that has the next greater cumulative
    # probability
    probability = random.random()
    for i in range(0, len(cumulative_probability)):
        if probability <= cumulative_probability[i]:
            new_kid = swarm[i]
            break
    # Make new copy
    a_new_kid = Particle()
    a_new_kid.v = new_kid.v[:]
    a_new_kid.x = new_kid.x[:]
    a_new_kid.p_best = new_kid.p_best[:]
    #print(a_new_kid.x)
    return a_new_kid


def find_best_fit(swarm):
    """Returns the particle with the best fit in the swarm
    in order to perform elitism.
    """
    fitt = []
    for particle in swarm:
        #cprint(particle.fit)
        fitt.append(particle.fit)
    minimum = min(fitt)
    index_of_min = fitt.index(minimum)
    return swarm[index_of_min]

def algorithm(pop, generations):
    """Runs the main pso - ga algorithm as described on the paper"""
    print("Parameter: ", Particle.C1, Particle.C2, Particle.W, Particle.CROSSOVER_PROB, Particle.MUTATION_PROB)
    print('number of Machine %s and Job %s:' %(open.num_mc, open.num_job))
    print("number of Population %s and Generation %s: " %(pop, generations))
    print("-----------------------------------")
    swarm = instantantiate(pop)
    best_history = []  # keeps track of best history to terminate program

    print("Searching...")
    for i in range(generations):
        algorithm.gen = i
        # ---------------Step 3-------------------
        # print "--------Start----------"
        for particle in swarm:
            #print(particle.x)
            particle.fit = particle.calculate_fit(particle.x)
            #particle.tard = particle.calculate_tard(particle.x)
            #print(particle.fit)
        # print"------------------------"

        # ----------------Step 4------------------
        new_gen = []
        the_best = find_best_fit(swarm)
        elite = Particle()
        elite.v = the_best.v[:]
        elite.x = the_best.x[:]
        elite.p_best = the_best.p_best[:]
        new_gen.append(elite)
        for j in range(len(swarm)):
            # Decide for crossover
            dont_crossover = random.random()
            if dont_crossover < particle.CROSSOVER_PROB:
                parent1 = selection(swarm)
                parent2 = selection(swarm)
                a_new_kid = parent1.crossover(parent2)
            else:
                a_new_kid = selection(swarm)
            a_new_kid.mutate()
            a_new_kid.remove_duplicates()
            new_gen.append(a_new_kid)
        swarm = new_gen
        #print("------After mutation-------")
        for particle in swarm:
            #print(particle.x)
            particle.fit = particle.calculate_fit(particle.x)
            # print particle.fit
        # print "---------------------------"

        # ----------------Step 5------------------
        # Find p_best of each particle
        for particle in swarm:
            if particle.fit > particle.calculate_fit(particle.p_best):
                particle.p_best = particle.x[:]
        ''' # Find g_best
        fitt = []
        for particle in swarm:
            fitt.append(particle.calculate_fit(particle.p_best))
        minimum = min(fitt)
        if minimum < particle.calculate_fit(Particle.g_best):
            position = fitt.index(minimum)
            Particle.g_best = swarm[position].x[:]'''
        # Find g_best
        fitt = []
        position = 0
        for particle in swarm:
            fitt.append([particle.calculate_fit(particle.p_best),particle.p_best])
        #print(fitt)
        temp_min = max(fitt)
        algorithm.minimum = temp_min[0]
        if algorithm.minimum > particle.calculate_fit(Particle.g_best):
            position = fitt.index(temp_min)
            Particle.g_best = temp_min[1]
        best_history.append(temp_min)
        # ---------Uncoment the following six lines for faster results----------
        if abs(best_history[i][0] - best_history[i - 1][0]) < 0.00001 and i > 0:
            same += 1
        else:
            same = 0
        if same >= 20:
            j = 1000
            break
        # print "-----After step 5-------"
        for particle in swarm:
            particle.fit = particle.calculate_fit(particle.x)
            #print(particle.fit)
        #print("------------------------")
        # ----------------Step 6------------------
        #print("--------Update position-------")
        #print("position: ", position)
        for i in range(len(swarm)):
            if i != position:
                swarm[i].update_velocity()
            else:
                pass
        for i in range(len(swarm)):
            if i != position:
                swarm[i].update_position()
            else:
                pass
        for particle in swarm:
            particle.fit = particle.calculate_fit(particle.x)
            #print(particle.fit)
        sg.one_line_progress_meter('Searching', same+1, 20, 'Searching For Solution')
    #print(best_history)
    sg.popup_animated(None)
    gui_chart(Particle.C1, Particle.C2, Particle.W, Particle.CROSSOVER_PROB, Particle.MUTATION_PROB, best_history)
    sol = max(best_history)
    algorithm.min = sol[0]
    algorithm.best_x = sol[1]

        #print("------------------------------")
        # raw_input()
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
#dur=[sum(i) for i in pt]

#print(dd)

def chart(best_history):
    plt.cla()
    #print(gen,best_history[gen][0])
    plt.plot([i for i in range(len(best_history))],[best_history[i][0] for i in range(len(best_history))],'b')
    plt.title('Objective Value Chart')
    plt.xlabel('Genertation')
    plt.ylabel('Objective Value')
    #plt.grid(True)
    return plt.gcf()
def draw(canvas,figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure,canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top',fill='both',expand=1)
    return figure_canvas_agg
def gui_chart(C1,C2,W,CR,MR,best_history):
    layout = [
        [sg.Text('line Plot')],
        [sg.Canvas(size=(1000,1000),key="-CANVAS-")],
        [sg.Exit()],
        [sg.Text('Window Auto Close After 10 Seconds')]
    ]
    dict = ["Parameter C1 :",str(C1),",C2 :",str(C2),",W :",str(W),",CR :",str(CR),",MR :",str(MR)]
    title = "".join(dict)
    window = sg.Window(title, layout, finalize=True, element_justification='center')
    #event, value = window.read()
    draw(window["-CANVAS-"].TKCanvas,chart(best_history))
    while True :
        event, values = window.Read(timeout=1000 * 5)  # in milliseconds
        if event in ('__TIMEOUT__',):
            #print('timed execution inside event loop')
            #sg.popup_auto_close('Timeout')
            break

        if event in (sg.WIN_CLOSED,'Exit'):
            break
    window.close()
def color(row):
    r = lambda: random.randint(0, 255)
    colors = ['#%02X%02X%02X' % (r(), r(), r())]
    c_dict = {j+1 : colors}
    return c_dict[row['Resource']]

def gant_matplotlib(Gbest):
    from matplotlib.patches import Patch
    from pandas import Timestamp

    j_keys = [j for j in range(open.num_job)]
    key_count = {key: 0 for key in j_keys}
    j_count = {key: 0 for key in j_keys}
    m_keys = [j + 1 for j in range(open.num_mc)]
    m_count = {key: 0 for key in m_keys}
    j_record = {}
    today = Timestamp (sg.popup_get_date())
    for i in G_sequence_best:
        gen_t = int(open.pt[i][key_count[i]])
        gen_m = int(open.ms[i][key_count[i]])
        j_count[i] = j_count[i] + gen_t
        m_count[gen_m] = m_count[gen_m] + gen_t

        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]

        start_time = str(today + datetime.timedelta(
            seconds=(j_count[i] - open.pt[i][key_count[i]]) * 3600))  # convert seconds to hours, minutes and seconds
        end_time = str(today + datetime.timedelta(seconds=j_count[i] * 3600))
        j_record[(i, gen_m)] = [start_time, end_time]

        key_count[i] = key_count[i] + 1

    df = []
    for j in j_keys:
        for i in range(3):
            m = int(open.ms[j][i])
            df.append(dict(Task='Machine %s' % (m),Start='%s' % (str(j_record[(j, m)][0])),
                           Finish='%s' % (str(j_record[(j, m)][1])), Resource='Job %s' % (j + 1)))

    df['color'] = df.apply(color, axis=1)
    ##### PLOT #####
    fig, (ax, ax1) = plt.subplots(2, figsize=(16, 6), gridspec_kw={'height_ratios': [6, 1]}, facecolor='#36454F')
    ax.set_facecolor('#36454F')
    ax1.set_facecolor('#36454F')
    # bars
    ax.barh(df.Task, df.current_num, left=df.start_num, color=df.color)
    ax.barh(df.Task, df.days_start_to_end, left=df.start_num, color=df.color, alpha=0.5)

    for idx, row in df.iterrows():
        ax.text(row.end_num + 0.1, idx, f"{int(row.Completion * 100)}%", va='center', alpha=0.8, color='w')
        ax.text(row.start_num - 0.1, idx, row.Task, va='center', ha='right', alpha=0.8, color='w')

    # grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='k', linestyle='dashed', alpha=0.4, which='both')

    # ticks
    xticks = np.arange(0, df.end_num.max() + 1, 3)
    xticks_labels = pd.date_range(proj_start, end=df.End.max()).strftime("%m/%d")
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
    #df['color'] = df.apply(color, axis=1)

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Machine", width=2340, height=1080, color="Resource", text="Resource", title="Job shop Schedule",hover_data=['Machine', 'Start', 'Finish', 'Resource', 'Task'])
    #fig = ff.create_gantt(df,colors="color", index_col='Resource', show_colorbar=True, group_tasks=True,task_names="Task", showgrid_x=True,title='Job shop Schedule', hover_data=['Machine', 'Start', 'Finish', 'Resource', 'Task'])
    #fig.show(fig, filename='GA_job_shop_scheduling', world_readable=True)
    return fig.show()

def gant_chart_fail(G_sequence_best):
    '''--------plot gantt chart-------'''
    from matplotlib.patches import Patch
    import plotly.figure_factory as ff
    import datetime

    j_keys = [j for j in range(open.num_job)]
    key_count = {key: 0 for key in j_keys}
    j_count = {key: 0 for key in j_keys}
    m_keys = [j + 1 for j in range(open.num_mc)]
    m_count = {key: 0 for key in m_keys}
    j_record = {}
    today = datetime.datetime.today()
    for i in G_sequence_best:
        gen_t = int(open.pt[i][key_count[i]])
        gen_m = int(open.ms[i][key_count[i]])
        j_count[i] = j_count[i] + gen_t
        m_count[gen_m] = m_count[gen_m] + gen_t

        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]

        start_time = str(today + datetime.timedelta(seconds=(j_count[i] - open.pt[i][key_count[i]]) * 3600))  # convert seconds to hours, minutes and seconds
        end_time = str(today + datetime.timedelta(seconds=j_count[i] * 3600))
        j_record[(i, gen_m)] = [start_time, end_time]

        key_count[i] = key_count[i] + 1

    df = []
    r = lambda: random.randint(0, 255)
    colors = ['#%02X%02X%02X' % (r(), r(), r())]
    for j in j_keys:
        colors.append('#%02X%02X%02X' % (r(), r(), r()))
        for i in range(3):
            m = int(open.ms[j][i])
            df.append(dict(Task='Machine %s' %(m), Start='%s' %(str(j_record[(j, m)][0])), Finish='%s' % (str(j_record[(j, m)][1])), Resource='Job %s' % (j + 1)))

    #print(colors)
    fig = ff.create_gantt(df,colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True,title='Job shop Schedule')
    #fig.show(fig, filename='GA_job_shop_scheduling', world_readable=True)
    return fig.show()
