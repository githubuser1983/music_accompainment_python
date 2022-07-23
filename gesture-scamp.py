import numpy as np
from getkey import getkey, keys
import random, sys, pickle
from scamp import Session, Ensemble, current_clock
from compute_knn_model import *
from music21.pitch import Pitch
from cvzone.HandTrackingModule import HandDetector
import cv2,numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
from sklearn.decomposition import PCA

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


currentLmList = []
daumen_note,zeigefinger_note,mittelfinger_note,ringfinger_note,kleinerfinger_note = 5*[None]

d,zf,mf,rf,kf = [],[],[],[],[]

currentFingers1 = []
currentFingers2 = []
def imageInLoop():
    global cap, detector, currentLmList, pcaHand,daumen_note,zeigefinger_note,mittelfinger_note,ringfinger_note,kleinerfinger_note
    global knn_nr_neighbors, currentFingers,s
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        #print(lmList1)
        #currentLmList.append(lmList1)
            
            
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)
        
        currentFingers1.append(fingers1)
        
        if len(currentFingers1)>=2:
            if sum(np.array(currentFingers1[-1])-np.array(currentFingers1[-2]))>0.0: #change in finger movement:
                code = 144 # note on
                dt = 0.1
                pitch, volume = sum([2**i*fingers1[i] for i in range(5)])+42, sum(fingers1)/5.0
                midi = code,pitch,volume
                duration = {0:0.25,1:0.25,2:0.25,3:0.5,4:0.5,5:1.0}[sum(fingers1)]
                isRest = False
                note = pitch,duration,volume,isRest
                append_knn_notes(0,note)       
        print("fingers up, hand one =",fingers1)

        if hands[0]["type"]=="Left":
            knn_nr_neighbors = sum(fingers1)+1
        
        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)

            currentFingers2.append(fingers2)
        
            if len(currentFingers2)>=2:
                if sum(np.array(currentFingers2[-1])-np.array(currentFingers2[-2]))>0.0: #change in finger movement:
                    code = 144 # note on
                    dt = 0.1
                    pitch, volume = sum([2**i*fingers2[i] for i in range(5)])+42, sum(fingers2)/5.0
                    midi = code,pitch,volume
                    duration = np.power(2.0,-(sum(fingers1)-1))
                    isRest = False
                    note = pitch,duration,volume,isRest
                    append_knn_notes(1,note)
                #callback_midi(midi,dt)
                
                #print(midi)
            #knn_nr_neighbors = 1+sum(fingers1)+sum(fingers2)
            # Find Distance between two Landmarks. Could be same hand or different hands
            #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    


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



def append_knn_notes(instr_to_play,note):
    global knn_notes_to_play,global_counter_midi_pressed,instrs,list_nbrs_notes_octaves,knn_is_reverse,knn_nr_neighbors
    global jump
    #instr_to_play = global_counter_midi_pressed % len(instrs)
    pitch,duration,volume,isPause = note
    play_left_hand = 1*(pitch<128//2)
    #instr_to_play = 0 #play_left_hand
    new_row = np.array([pitch,duration,volume,isPause])
    octaves = pitch//36
    nbrs,notes = list_nbrs_notes_octaves[octaves]
    if knn_nr_neighbors > 0:
        bm = findBestMatches(nbrs,new_row,n_neighbors=knn_nr_neighbors,reverse=knn_is_reverse)
        if jump:
            jump = False
            bm = [bm[-1]]    
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
    print(midi,dt)
    if code == 144: # note on
        # flush old notes to play for harp, if any:
        #knn_notes_to_play[instr_to_play] = []
        pressed_notes.append([pitch,volume,-1])
        s.fork(play_piano,(pitch,volume,instr_to_play ))
        print("forked")
        
    elif code==128: # note off
        if len(pressed_notes)>0:           
            #knn_notes_to_play[instr_to_play] = []
            global_counter_midi_pressed += 1
            #instr_to_play = global_counter_midi_pressed % len(instrs)
            pressed_notes[-1][-1] = dt
            pitch,volume,dt = pressed_notes.pop(0)
            print("pitch,volume,dt",pitch,volume,dt)
            duration = dt*4
            print(duration,dt)
            isPause = (False)*1
            note = pitch,duration,volume,isPause
            inds[instr_to_play].append(note)
            append_knn_notes(note)           
        note_id = started_notes.pop(-1)        
        s.instruments[instr_to_play].end_note(note_id)


        
def scamp_loop():
    global inds,tracks,s,started_transcribing,knn_notes_to_play_for_marimba,knn_notes_to_play,global_counter_midi_pressed,instrs
    global instr_to_play,loopKnn
   
    while True:
        #instr_to_play = global_counter_midi_pressed % len(instrs)   
        if len(knn_notes_to_play[instr_to_play])>0:
            note = knn_notes_to_play[instr_to_play].pop(0)
            pitch,duration,volume,isPause = note
            if len(knn_notes_to_play[instr_to_play])==0 and loopKnn:
                append_knn_notes(instr_to_play,note)
            inds[instr_to_play].append(note)    
            current_clock().fork(piano_play_note,(pitch,volume,duration,isPause,instr_to_play))        
        if len(current_clock().children()) > 0:
            current_clock().wait_for_children_to_finish()
        #    #pass
        else:
            # prevents hanging if nothing has been forked yet
            current_clock().wait(1.0)    

def main():
    global s, s_forked,midiFileName,instrumentMidiNumber,knn_is_reverse,knn_nr_neighbors,inds,instr_to_play,loopKnn
    global jump 
    # https://stackoverflow.com/questions/24072790/how-to-detect-key-presses
    #try:  # used try so that if user pressed other than the given key error will not be shown
    #    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
    #        print('You Pressed A Key!')
    #        break  # finishing the loop
    #except:
    #    break 
        
    while True:
        imageInLoop()
        if not s_forked:
            s_forked = True
            s.fork(scamp_loop)
            
        key = "-1" #getkey() 
        if key=="f":
            inds[instr_to_play] = []
        if key=="l":
            loopKnn = not loopKnn
        if key=="j":
            jump = not jump
        if key=="r":
            knn_is_reverse = not knn_is_reverse
        if key==keys.UP:
            knn_nr_neighbors += 1
        if key==keys.DOWN and knn_nr_neighbors >= 1:
            knn_nr_neighbors -= 1
        if key==keys.RIGHT:
            knn_nr_neighbors *= 2
        if key==keys.LEFT and knn_nr_neighbors >= 1:
            knn_nr_neighbors = knn_nr_neighbors//2    
            
        if key=="q":
            print("You pressed q. Quitting the program and writing the midi file")
            fn = midiFileName
            print(fn)
            writePitches(fn,inds,tempo=tempo,instrument=[instrumentMidiNumber],add21=False,start_at= [0,0],durationsInQuarterNotes=True)
            return
        

            
        #if not zeigefinger_note is None:
        #    append_knn_notes(transformNote(*zeigefinger_note))
        #if not mittelfinger_note is None:
        #    append_knn_notes(transformNote(*mittelfinger_note))
    
    
    cap.release()
    cv2.destroyAllWindows()
        

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
        "soundfont": generalSF, 
        "loopKnn" : True,
        "knn_nr_neighbors" : 5,
        "knn_is_reverse" : True,
        "instrumentName" : "Mellow Steinway",
        "midiFileName" : "./midi/live_music_grand_piano.mid",
        "knn_model" : "./knn_models/knn.pkl",
        "jump" : False # will jump directly to the k-th nearest neighbor, without looking at the other nearest neighbors.
    }
    writeConfiguration("./start-conf.yaml",conf)

loopKnn = conf["loopKnn"]    
knn_nr_neighbors = conf["knn_nr_neighbors"]
knn_is_reverse=conf["knn_is_reverse"]    
instrumentName = conf["instrumentName"]    
midiFileName = conf["midiFileName"]
instrumentMidiNumber = get_general_midi_number_for_instrument_name(instrumentName,generalSF)    
std = [instrumentName,"Harp"]
sf = conf["soundfont"]
list_nbrs_notes_octaves = load_knn(conf["knn_model"])
jump = conf["jump"]
    
tracks = construct_ensemble(sf,std)

print(tracks)
print(len(tracks))

tempo = 80
s = Session(tempo=tempo,default_soundfont=sf).run_as_server()

s.print_available_midi_output_devices()
#print(dir(s))

s.print_available_midi_input_devices()

    
#for t in tracks:
#    print(t[1][0].preset)
#    s.add_instrument(t[0])

piano = s.new_part("Mellow Steinway")
harp = s.new_part("Harp")
#marimba = s.new_part("Marimba")

instrs = [piano,harp] #s.instruments

print(instrs)

knn_notes_to_play = [] 

for i in instrs:
    s.add_instrument(i)
    inds.append([])
    knn_notes_to_play.append([])

#s.add_instrument(harp)
#s.add_instrument(piano)
#s.add_instrument(marimba)

   
#s.register_midi_listener(port_number_or_device_name="LPK25", callback_function=callback_midi)

#s.start_transcribing()

main()
#s.wait_forever()
#s.run_as_server()



