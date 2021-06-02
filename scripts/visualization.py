import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from tempfile import NamedTemporaryFile#module creates temporary files and directories
from matplotlib.colors import LinearSegmentedColormap

import jinja2
import re
import os
from subprocess import run,call #allows you to spawn new processes


def attack_rgba(attack_dev):
    '''
    Given an array of deviations, provide corresponding coloration. As the deviation gets bigger, the color will be more red.
    '''
    cdict = {'red': [(0.0, 0.0, 0.0),
                     (0.04, 0.0, 0.0),
                     (0.07, 0.99653979, 0.99653979),
                     (0.24, 0.64705882, 0.64705882),
                     (1, 0.64705882, 0.64705882)],

             'green': [(0.0, 0.40784314, 0.40784314),
                       (0.04, 0.40784314, 0.40784314),
                       (0.07, 0.89273356, 0.89273356),
                       (0.24, 0, 0),
                       (1, 0, 0)],

             'blue': [(0.0, 0.21568627, 0.21568627),
                      (0.04, 0.21568627, 0.21568627),
                      (0.07, 0.56908881, 0.56908881),
                      (0.24, 0.14901961, 0.14901961),
                      (1, 0.14901961,0.14901961)]}
    
    attacks_cm = LinearSegmentedColormap("attack_colors", cdict) #create a colormap to interpolate colors
    sm = cm.ScalarMappable(norm=plt.Normalize(0, 1, clip=True), cmap=attacks_cm) #scalar data to RGBA map
    if type(attack_dev) is not np.ndarray:
        attack_dev = np.array([attack_dev])
    abs_devs = np.abs(attack_dev)
    return sm.to_rgba(abs_devs)



class ColorIterator: #Iterator that changes the color of the nodes in lilypond
    def __init__(self, rgbas):
        self.rgbas = rgbas
        self.pos = -1

    def current(self):
        return " %.3f %.3f %.3f" %\
               (self.rgbas[self.pos, 0],
                self.rgbas[self.pos, 1],
                self.rgbas[self.pos, 2])

    def __iter__(self):
        return self

    def next(self):
        self.pos += 1
        if self.pos < len(self.rgbas):
            return self.current()
        else:
            raise StopIteration

def size_from_color(rgbas):
    '''
    This functin takes the notes that are miss played and augments its size when plotting.
    '''
    sizes = []
    for rgb in rgbas:
        if rgb[0]> 0.4:
            sizes.append(5)
        else:
            sizes.append(0)
    return sizes
        
class SizeIterator:#----Iterator to change the size of miss played notes in lilypond
    def __init__(self, sizes):
        self.size = sizes
        self.pos = -1

    def current(self):
        return self.size[self.pos]

    def __iter__(self):
        return self

    def next(self):
        self.pos += 1
        if self.pos < len(self.size):
            return self.current()
        else:
            raise StopIteration

class TempWaveFormsGenerator: #macro that prints the waveform in the lilyond
    def __init__(self, gen_lambda):
        self.gen_lambda = gen_lambda
        self.all_files = []

    def eps(self, first_bar, last_bar, w = 1, h = 0.1, left_border_shift=-0.15, right_border_shift=-0.2):
        f = self.gen_lambda(first_bar, last_bar, w, h, left_border_shift, right_border_shift)
        self.all_files.append(f)
        return f

    def clear(self):
        for f in self.all_files:
            os.unlink(f)



#----------------------------------Plotting functions-------------------------------------------------------#
def save_bar_plot(audio,expected_attacks,left_time=0.0,right_time=None,fs=44100, w=1, h=0.1, dpi=300,
        actual_attacks=np.array([], dtype=float),color_func=attack_rgba):
    '''
    INPUT: 
    audio: targe audio to be assessed
    expected_attacks: ground trugh beats
    left: initial time of the music score
    actual_attacks: predicted onsets (predicted guitar strokes)
    
    OUTPUT: 
    '''
    # half-beat margin
    left = int(left_time * fs)
    if right_time is None: ## if the finalization time of the score not provided calculate it
        right = len(audio)
        right_time = float(right) / fs
    else:
        right = int(right_time * fs) ## if provided put it in frames instead of seconds
        
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_xlim(left=0, right=right-left) #set the x axis size (from the start frame to the last frame of audio)
    fig.add_axes(ax)
    
    if left < 0: #------------------------plot the audio wave 
        addon = np.zeros(-left)
        left = 0
        ax.plot(np.concatenate((addon, audio[left:right]), axis = None), lw=0.5)
    else:
        ax.plot(audio[left:right], lw=0.5)

    # filter events.
    expected_attacks = expected_attacks + left_time
    filtered_expected = expected_attacks[(expected_attacks >= left_time) & (expected_attacks < right_time)] #take the onsets that are inside your range (start - final of the score)
    filtered_actual = actual_attacks[(actual_attacks >= left_time) & (actual_attacks < right_time)] #take the gt beats that are inside your range (start - final of the score)
    ## print the GT beats above the wave and print a region colored according to the deviation of the predicted onset from the GT beat
    for x in actual_attacks: #print you predicted onset with a black line
        plt.axvline((x-left_time) * fs,0,0.25, color='k', lw=0.5)
    for x in expected_attacks:
        plt.axvline((x-left_time) * fs,0.25,0.75, color='c', lw=0.5)


    #
    for x in filtered_actual:
        i = np.searchsorted(expected_attacks, x)
        if i >= len(expected_attacks) or (0 < i and x - expected_attacks[i - 1] < expected_attacks[i] - x):
            expected_attack = expected_attacks[i - 1]
            x1 = expected_attacks[i - 1]
            x2 = x
            attack_dev = x - expected_attacks[i - 1] #calculate the deviation from the GT beat
        else:
            expected_attack = expected_attacks[i]
            x1 = x
            x2 = expected_attacks[i]
            attack_dev = x - expected_attacks[i]
        if (expected_attack in filtered_expected): #color the deviation with a region and attack_rgba fucntion
            rgbas = color_func(attack_dev)
            """if rgbas[0][0] > 0.4: #----------------------------If the deviation of the tempo is pretty high, highlight it
                N=5000
                plt.axvspan(((x1 - left_time) * fs)-N,((x2 - left_time)* fs)+N,ymax=1, color='yellow',alpha=0.05) #color the tempo deviations
            """
            plt.axvspan((x1 - left_time) * fs, (x2 - left_time) * fs,facecolor=rgbas[0])
    
    #Save file
    fname = NamedTemporaryFile(suffix='.eps', delete=False)
    plt.savefig(fname, dpi=dpi, format='eps')
    plt.close(fig)
    fname.close()
    
    return fname.name



def score_image(template_dir, tamplate_name, normLu, eps_lambda, image_format='png'):
    '''
    INPUT:
    normLu = nlu = chroma scores #Chroma estimated
    template_name: .ly file name
    template_dir: directory where the .ly file is stored
    eps_lambda: macro that will plot the wave form using lilypond
    Create the final _assessment.png visualization using lilypond software.
    '''
    #m = np.exp(normLu)
    sm = cm.ScalarMappable(
        norm=plt.Normalize(0, 1, clip=True), cmap=plt.get_cmap('RdYlGn')) #define a colormap (for deviations and notes)
    rgbas = sm.to_rgba(normLu)
    
    it = ColorIterator(rgbas) #Create an iterable for the lilypond macro in charge of coloring the notes
    size = SizeIterator(size_from_color(rgbas))#Create an iterable for the lilypond macro in charge of augmenting the size of miss played notes
    
    #Read the lilypond template
    latex_jinja_env = jinja2.Environment(
        variable_start_string='%{',
        variable_end_string='%}',
        trim_blocks=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(template_dir))
    template = latex_jinja_env.get_template(tamplate_name)
    
    eps_generator = TempWaveFormsGenerator(eps_lambda) #macor that will plot the wave form in lilypond
    #Assign lilypond macros to iterables created
    res = template.render({
        'next_color': lambda: it.next(),
        'current_color': lambda: it.current(),
        'next_size': lambda: size.next(),#--------------added
        'eps_waveform': eps_generator.eps
    })
    
    #Compile lilypond file with the macros
    fin = NamedTemporaryFile(suffix='.ly', delete=False)
    fin.write(res.encode('UTF8'))
    fin.close()
    #Generate using lilypond a .png image containing the visualization of the assessment
    image_format == 'png'
    suffix = 'png'
    lily_format_option = '--png'
    ext_cutter = re.compile('\.' + suffix + '$')
    fout = NamedTemporaryFile(suffix='.' + suffix,delete=False)
    fout.close()
    os.unlink(fout.name)

    call(["lilypond", lily_format_option, "-dcrop=#t", '-o', ext_cutter.sub('', fout.name),
         "-dresolution=300", fin.name])
    
    os.unlink(fin.name)
    eps_generator.clear()  

    return fout.name