import numpy as np
from getkey import getkey, keys
import random, sys, pickle
from scamp import Session, Ensemble, current_clock
from compute_knn_model import *
from music21.pitch import Pitch

#print([pitch.Pitch(midi=int(p)) for p in pitchlist])



from scamp import Session, Ensemble
from scamp._soundfont_host import get_best_preset_match_for_name

def get_general_midi_number_for_instrument_name(p,sf):
    ensemble = Ensemble(default_soundfont=sf)
    return (get_best_preset_match_for_name(p,which_soundfont=ensemble.default_soundfont)[0]).preset

def construct_ensemble(sf,std):
    global piano_clef,piano_bass, flute, strings, session
    ensemble = Ensemble(default_soundfont=sf)

    ensemble.print_default_soundfont_presets()

    return [(ensemble.new_part(p),get_best_preset_match_for_name(p,which_soundfont=ensemble.default_soundfont)) for p in std] 




def play_piano(pitch,volume,instr_to_play):
    global started_notes,instrs
    #print("pitch,volume:",pitch,volume)
    #print("notes:",pressed_notes)
    note_id = s.instruments[instr_to_play].start_note(pitch,volume/128.0)
    started_notes.append(note_id)    
    
def piano_play_note(pitch,volume,duration,isPause,instr_to_play):
    global instrs,s,tempo,instrumentName,knn_is_reverse,knn_nr_neighbors
    if isPause:
        return
    length = duration
    print("playing "+instrumentName+" [",instr_to_play,"]: ", Pitch(midi=int(pitch)),volume,duration,length)
    print("knn_reverse = ",knn_is_reverse)
    print("knn_nr_neighbors = ", knn_nr_neighbors)
    s.instruments[instr_to_play].play_note(pitch,volume/128.0,length,blocking=True)



def append_knn_notes(note):
    global knn_notes_to_play,global_counter_midi_pressed,instrs,list_nbrs_notes_per_two_octaves,knn_is_reverse,knn_nr_neighbors
    #instr_to_play = global_counter_midi_pressed % len(instrs)
    pitch,duration,volume,isPause = note
    play_left_hand = 1*(pitch<128//2)
    instr_to_play = 0 #play_left_hand
    new_row = np.array([pitch,duration,volume,isPause])
    two_octaves = pitch//24
    nbrs,notes = list_nbrs_notes_per_two_octaves[two_octaves]
    bm = findBestMatches(nbrs,new_row,n_neighbors=knn_nr_neighbors,reverse=knn_is_reverse)
    print([notes[b] for b in bm])
    for b in bm:
        pitch,duration,volume,isPause = notes[b]
        knn_notes_to_play[instr_to_play].append(notes[b])   
    
def callback_midi(midi,dt):
    global inds,s,piano,nbrs,notes,tempo, started_notes,knn_notes_to_play_for_marimba,knn_notes_to_play
    global instr_to_play, global_counter_midi_pressed,instrs
    
    code,pitch,volume = midi
    play_left_hand = 1*(pitch<128//2)
    instr_to_play = 0 #play_left_hand
    #print(midi,dt)
    if code == 144: # note on
        # flush old notes to play for harp, if any:
        #knn_notes_to_play[instr_to_play] = []
        pressed_notes.append([pitch,volume,-1])
        s.fork(play_piano,(pitch,volume,instr_to_play ))
        
    elif code==128: # note off
        if len(pressed_notes)>0:           
            #knn_notes_to_play[instr_to_play] = []
            global_counter_midi_pressed += 1
            #instr_to_play = global_counter_midi_pressed % len(instrs)
            pressed_notes[-1][-1] = dt
            pitch,volume,dt = pressed_notes.pop(0)
            print("pitch,volume,dt",pitch,volume,dt)
            duration = dt
            print(duration,dt)
            isPause = (False)*1
            note = pitch,duration,volume,isPause
            inds[instr_to_play].append(note)
            append_knn_notes(note)           
        note_id = started_notes.pop(-1)        
        s.instruments[instr_to_play].end_note(note_id)


        
def scamp_loop():
    global inds,tracks,s,started_transcribing,knn_notes_to_play_for_marimba,knn_notes_to_play,global_counter_midi_pressed,instrs
    global instr_to_play
   
    while True:
        #instr_to_play = global_counter_midi_pressed % len(instrs)   
        if len(knn_notes_to_play[instr_to_play])>0:
            note = knn_notes_to_play[instr_to_play].pop(0)
            pitch,duration,volume,isPause = note
            if len(knn_notes_to_play[instr_to_play])==0:
                append_knn_notes(note)
            inds[instr_to_play].append(note)    
            current_clock().fork(piano_play_note,(pitch,volume,duration,isPause,instr_to_play))        
        if len(current_clock().children()) > 0:
            current_clock().wait_for_children_to_finish()
        #    #pass
        else:
            # prevents hanging if nothing has been forked yet
            current_clock().wait(1.0)    

def main():
    global s, s_forked,midiFileName,instrumentMidiNumber,knn_is_reverse,knn_nr_neighbors,inds,instr_to_play
    
    # https://stackoverflow.com/questions/24072790/how-to-detect-key-presses
    #try:  # used try so that if user pressed other than the given key error will not be shown
    #    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
    #        print('You Pressed A Key!')
    #        break  # finishing the loop
    #except:
    #    break 
        
    while True:
        if not s_forked:
            s_forked = True
            s.fork(scamp_loop)
        key = getkey() 
        if key=="f":
            inds[instr_to_play] = []
        if key=="r":
            knn_is_reverse = not knn_is_reverse
        if key==keys.UP:
            knn_nr_neighbors += 1
        if key==keys.DOWN and knn_nr_neighbors > 1:
            knn_nr_neighbors -= 1
        if key==keys.RIGHT:
            knn_nr_neighbors *= 2
        if key==keys.LEFT and knn_nr_neighbors > 1:
            knn_nr_neighbors = knn_nr_neighbors//2    
            
        if key=="q":
            print("You pressed q. Quitting the program and writing the midi file")
            fn = midiFileName
            print(fn)
            writePitches(fn,inds,tempo=tempo,instrument=[instrumentMidiNumber],add21=False,start_at= [0,0],durationsInQuarterNotes=True)
            return

instr_to_play = 0            
inds = []
global_counter_midi_pressed = 0            
started_transcribing = False
s_forked = False
knn_notes_to_play_for_marimba = []        
       
pressed_notes = []
started_notes = []


# soundfonts from https://sites.google.com/site/soundfonts4u/home
generalSF = "/usr/share/sounds/sf3/MuseScore_General.sf3"
pianoSF = "~/Dokumente/MuseScore3/SoundFonts/4U-Mellow-Steinway-v3.6.sf2"
    
if len(sys.argv) == 2:
    conf = readConfiguration(sys.argv[1])
else:
    conf = {
        "knn_nr_neighbors" : 5,
        "knn_is_reverse" : True,
        "instrumentName" : "Grand Piano",
        "midiFileName" : "./midi/live_music_grand_piano.mid",
        "knn_model" : "./knn_models/knn.pkl"
    }
    writeConfiguration("./scamp-piano-configurations/start-conf.yaml",conf)

knn_nr_neighbors = conf["knn_nr_neighbors"]
knn_is_reverse=conf["knn_is_reverse"]    
instrumentName = conf["instrumentName"]    
midiFileName = conf["midiFileName"]
instrumentMidiNumber = get_general_midi_number_for_instrument_name(instrumentName,generalSF)    
std = [instrumentName]
sf = generalSF 
list_nbrs_notes_per_two_octaves = load_knn(conf["knn_model"])
    
tracks = construct_ensemble(sf,std)

print(tracks)
print(len(tracks))

tempo = 80
s = Session(tempo=tempo,default_soundfont=sf).run_as_server()

s.print_available_midi_output_devices()
#print(dir(s))

s.print_available_midi_input_devices()

    
for t in tracks:
    print(t[1][0].preset)
    s.add_instrument(t[0])

#piano = s.new_part("Mellow Steinway")
#harp = s.new_part("Mellow Steinway")
#marimba = s.new_part("Marimba")

instrs = s.instruments #[piano,harp]

knn_notes_to_play = [] 

for i in instrs:
    inds.append([])
    knn_notes_to_play.append([])

#s.add_instrument(harp)
#s.add_instrument(piano)
#s.add_instrument(marimba)

   
s.register_midi_listener(port_number_or_device_name="LPK25", callback_function=callback_midi)

#s.start_transcribing()

main()
#s.wait_forever()
#s.run_as_server()



