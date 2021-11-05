from sage.all import *
import numpy as np
import random

 
def parseMidi(fp,part=0,volume_bounded_by_one=False,subtract21=False):
    import os
    from music21 import converter
    score = converter.parse(fp,quantizePost=True)
    score.makeVoices()
    from music21 import chord
    durs = []
    ll0 = []
    vols = []
    isPauses = []
    if subtract21:
        subtractThis = 21
    else:
        subtractThis = 0
    for p in score.elements[part].recurse().notesAndRests: #parts[0].streams()[part].notesAndRests:
        if type(p)==chord.Chord:
            pitches = sorted([e.pitch.midi-subtractThis for e in p])[0] # todo: think about chords
            vol = sorted([e.volume.velocity for e in p])[0]
            dur = float(p.duration.quarterLength)
            isPause = 0
        elif (p.name=="rest"):
            pitches = 64
            vol = 64
            dur = float(p.duration.quarterLength)
            isPause = 1
        else:
            pitches = p.pitch.midi-subtractThis
            vol = p.volume.velocity
            dur = float(p.duration.quarterLength)
            isPause =  0
        if not dur>0 and vol>0:
            continue
        ll0.append(min(87,pitches))    
        durs.append(dur)
        if vol is None or vol == 0:
            vol = 64
        if volume_bounded_by_one:    
            vols.append(vol*1.0/127.0)
        else:
            vols.append(vol*1.0)
        isPauses.append(isPause)
    return ll0,durs,vols,isPauses

def kernPause(a1,a2):
    return  1*(a1==a2)

def kernPitch(k1,k2):
    q = getRational(k2-k1)
    a,b = q.numerator(),q.denominator()
    return gcd(a,b)**2/(a*b)


import portion as PP
def muInterval(i):
    if i.empty:
        return 0
    return i.upper-i.lower

def jaccard(i1,i2):
    return muInterval(i1.intersection(i2))/muInterval(i1.union(i2))

def kernJacc(x,y):
    if 0.0<= x  and 0.0<= y :
        if x==y==0.0:
            return 1
        X = PP.closed(0,x)
        Y = PP.closed(0,y)
        return jaccard(X,Y)

def kernDurationQuotient(d1,d2):
    q = QQ(d1/d2)
    a,b = (q).numerator(),q.denominator()
    return gcd(a,b)**2/(a*b)    
    
def kernDuration(k1,k2):
    #return kernJacc(k1,k2)
    return min(k1,k2)/max(k1,k2)
    #return kernDurationQuotient(k1,k2)

def kernVolume(v1,v2):
    #return kernJacc(v1,v2)
    return min(v1,v2)/max(v1,v2)

def kernAdd(t1,t2,alphaPitch=0.25):
    pitch1,duration1,volume1,isPause1 = t1
    pitch2,duration2,volume2,isPause2 = t2
    #return 1.0/3*(1-alphaPitch)*kernPause(isPause1,isPause2)+alphaPitch*kernPitch(pitch1,pitch2)+1.0/3*(1-alphaPitch)*kernDuration(duration1,duration2)+1.0/3*(1-alphaPitch)*kernVolume(volume1,volume2)
    apa = alphaPitch["pause"]
    api = alphaPitch["pitch"]
    adu = alphaPitch["duration"]
    avo = alphaPitch["volume"]
    if np.abs(apa+api+adu+avo-1)<10**-5:
        return apa*kernPause(isPause1,isPause2)+api*kernPitch(pitch1,pitch2)+adu*kernDuration(duration1,duration2)+avo*kernVolume(volume1,volume2)
    else:
        return None

def kernMul(t1,t2):
    pitch1,duration1,volume1,isPause1 = t1
    pitch2,duration2,volume2,isPause2 = t2
    alpha = 0.1
    return (1-alpha)*kernPause(isPause1,isPause2)+alpha*(kernPitch(pitch1,pitch2)*kernDuration(duration1,duration2)*kernVolume(volume1,volume2))

def kern(alphaPitch):
    def f(t1,t2):
        return kernAdd(t1,t2,alphaPitch)
    return f


def getRational(k):
    alpha = 2**(1/12.0)
    x = RDF(alpha**k).n(50)
    return x.nearby_rational(max_error=0.01*x)


def ngrams(input, n):
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

def kernNgram(ngrams1,ngrams2,alphaPitch=0.25):
    return 1.0/len(ngrams1)*sum([ kern(alphaPitch)(ngrams1[i], ngrams2[i]) for i in range(len(ngrams1))]) 

def writePitches(fn,inds,tempo=82,instrument=[0,0],add21=True,start_at= [0,0],durationsInQuarterNotes=False):
    from MidiFile import MIDIFile

    track    = 0
    channel  = 0
    time     = 0   # In beats
    duration = 1   # In beats # In BPM
    volume   = 116 # 0-127, as per the MIDI standard

    ni = len(inds)
    MyMIDI = MIDIFile(ni,adjust_origin=False) # One track, defaults to format 1 (tempo track
                     # automatically created)
    MyMIDI.addTempo(track,time, tempo)


    for k in range(ni):
        MyMIDI.addProgramChange(k,k,0,instrument[k])


    times = [0.0,0.0] #start_at
    for k in range(len(inds)):
        channel = k
        track = k
        for i in range(len(inds[k])):
            pitch,duration,volume,isPause = inds[k][i]
            #print(pitch,duration,volume,isPause)
            track = k
            channel = k
            if not durationsInQuarterNotes:
                duration = 4*duration#*maxDurations[k] #findNearestDuration(duration*12*4)            
            print(k,pitch,times[k],duration,volume)
            if not isPause: #rest
                #print(volumes[i])
                # because of median:
                pitch = int(floor(pitch))
                if add21:
                    pitch += 21
                #print(pitch,times[k],duration,volume,isPause)    
                MyMIDI.addNote(track, channel, int(pitch), float(times[k]) , float(duration), int(volume))
                times[k] += duration*1.0  
            else:
                times[k] += duration*1.0
       
    with open(fn, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    print("written")    
    


# Idee: Benutze den Jaccard Koeff als positiv definiten Kernel um zwei Intervalle miteinander zu vergleichen.
# https://en.wikipedia.org/wiki/Jaccard_index

def kernIntervalMult(x1,x2):
    i1,n1 = x1
    i2,n2 = x2
    return jaccard(i1,i2)*kernNgram(n1,n2)

def kernIntervalAdd(x1,x2):
    i1,n1 = x1
    i2,n2 = x2
    alpha = 0.750
    return alpha*(jaccard(i1,i2))+(1-alpha)*kernNgram(n1,n2)

def kernInterval(x1,x2):
    return kernIntervalAdd(x1,x2)


def distInterval1(interval):
    return lambda x1,x2 : np.sqrt(2-2*kernInterval(interval[int(x1)],interval[int(x2)]))

def distNgram(interval,alphaPitch=0.25):
    return lambda x1,x2 : np.sqrt(2-2*kernNgram(interval[int(x1)],interval[int(x2)],alphaPitch))


def distInterval2(x1,x2):
    return np.sqrt(2*(1-kernInterval(x1,x2)))

def distKern(x,y,alphaPitch=0.25):
    #print(alphaPitch)
    return np.sqrt(2-2*kern(alphaPitch)(x,y))

def generateNotes(two_octaves):
    from itertools import product
    from music21 import pitch
    pitchlist = [p for p in list(range(two_octaves*24,two_octaves*24+24))]
    #distmat = np.array(matrix([[np.sqrt(2*(1.0-kernPitch(x,y))) for x in pitchlist] for y in pitchlist]))
    #permutation,distance = tspWithDistanceMatrix(distmat,exact=False)
    #pitchlist = [pitchlist[permutation[k]] for k in range(len(pitchlist))]
    print([pitch.Pitch(midi=int(p)) for p in pitchlist])
    durationlist = [1,1/2,1/4,1/8,1/16]
    #if len(durs)>2:
    #    distmat = np.array(matrix([[np.sqrt(2*(1.0-kernDuration(x,y))) for x in durationlist] for y in durationlist]))
    #    permutation,distance = tspWithDistanceMatrix(distmat)
    #    durationlist = [durationlist[permutation[k]] for k in range(len(durationlist))]
    print(durationlist)
    volumelist = vols = [(128//8)*(k+1) for k in range(8)] #[x*127 for x in [1.0/6.0,1.0/3.0,1.0/2.0,2.0/3.0 ]]
    #distmat = np.array(matrix([[np.sqrt(2*(1.0-kernVolume(x,y))) for x in volumelist] for y in volumelist]))
    #permutation,distance = tspWithDistanceMatrix(distmat)
    #volumelist = [volumelist[permutation[k]] for k in range(len(volumelist))]
    print(volumelist)
    pauselist = [False,True]
    ll = list(product(durationlist,volumelist,pauselist,pitchlist))
    #distmat = np.array(matrix([[distKern(x,y,alphaPitch) for x in ll] for y in ll]))
    #np.random.seed(43)
    #permutation,distance = tspWithDistanceMatrix(distmat,exact=False)
    #ll = [ll[permutation[k]] for k in range(len(ll))]
    print(len(ll))
    #print(ll)
    pitches = [p[3] for p in ll]
    durations = [d[0] for d in ll]
    volumes = [v[1] for v in ll]
    isPauses = [p[2] for p in ll]
    return pitches,durations,volumes,isPauses    


def findBestMatches(nbrs,new_row,n_neighbors=3,reverse=False):
    distances,indices = nbrs.kneighbors([np.array(new_row)],n_neighbors=n_neighbors)
    dx = sorted(list(zip(distances[0],indices[0])),reverse=reverse)
    print(dx)
    indi = [d[1] for d in dx]
    print(indi)
    #print(distances)
    #distances,indices = nbrs.query([np.array(new_row)],k=n_neighbors)
    return indi


def writeConfiguration(filename,configuration):    
    import yaml
    import io

    # Write YAML file
    with io.open(filename, 'w', encoding='utf8') as outfile:
        yaml.dump(configuration, outfile, default_flow_style=False, allow_unicode=True)

def readConfiguration(filename):        
    # Read YAML file
    import yaml
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded    

def get_notevalues():
    durationslist = [[sum([(QQ(2)**(n-i)) for i in range(d+1)]) for n in range(-8,3+1)] for d in range(0,3+1)]
    notevals = []
    for i in range(len(durationslist)):
        notevals.extend(durationslist[i])
    notevals = sorted(notevals)    
    return notevals

def distKern(kern):
    def f(a,b):
        return np.sqrt(2*(1-kern(a,b)))
    return f

def get_knn_model_durations(durations):
    #notevals = np.array([[x*1.0] for x in durations])
    #print(notevals)
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernDuration)).fit(durations)
    return nbrs,durations
                   
def get_knn_model_pitches(pitches):
    #pitches = np.array([[x*1.0] for x in pitches])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernPitch)).fit(pitches)
    return nbrs,pitches                    

def get_knn_model_volumes(volumes):
    #volumes = np.array([[x*1.0] for x in volumes])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernVolume)).fit(volumes)
    return nbrs,volumes                    

def get_knn_model_rests(rests):
    #rests = np.array([[x*1.0] for x in rests])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=distKern(kernPause)).fit(rests)
    return nbrs,rests     

def get_knn_model_notes(notes,alphaPitch):
    #notes = np.array([[x*1.0 for x in n] for n in notes])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    dK = distKern(kern(alphaPitch))
    nbrs = NearestNeighbors( algorithm='ball_tree',metric=dK).fit(notes)
    return nbrs,notes
                        
def get_nearest_note_by_radius(notes_nbrs,notes_list,note,radius):
    note = notes_list[findBestMatch(notes_nbrs,np.array([1.0*x for x in note]),radius)]
    return note


def getProbsFromWeights(weights):
    sW = sum(weights)
    probs = [w*1.0/sW for w in weights]
    alphaPitch = dict(zip(["pitch","duration","volume","pause"],probs))
    return alphaPitch

def make_np_array(ll):
    return np.array([[x*1.0] for x in ll])

def plot_graph(typeOfGraph,startRadius=0.25,plotInFile = True):
    import numpy as np, networkx as nx
    dd = {"pitches": (make_np_array(range(128)), get_knn_model_pitches),
          "durations": (make_np_array(get_notevalues()), get_knn_model_durations),
          "volumes":  (make_np_array([(128//8)*(k+1) for k in range(8)]), get_knn_model_volumes),
          "rests": (make_np_array([True,False]),get_knn_model_rests),
          }
    ll = dd[typeOfGraph][0]
    func = dd[typeOfGraph][1]
    nbrs,lls = func(ll)
    print("constructing graph(s)...for type = ",typeOfGraph)
    n_neighbors = 2
    connected = False
    radius = startRadius
    while not connected:
        print("constructing graph with ",radius," radius, until connected ..")
        #A_knn = nbrs.kneighbors_graph(n_neighbors=n_neighbors, mode='distance')
        A_radius = nbrs.radius_neighbors_graph(radius=radius, mode='distance',sort_results=True)
        G = nx.from_numpy_array(A_radius,create_using=nx.Graph)
        print((nx.number_connected_components(G)))
        connected = nx.is_connected(G)
        n_neighbors += 1
        radius = 1.1*radius
    print("graph connected with ",radius/1.1," radius")
    if plotInFile:
        Gr = Graph(G,loops=True)
        Gr.plot().save("./plots/"+typeOfGraph+"_graph_radius_"+str(np.round(radius/1.1,2))+".png")
    return G,ll
    

   
     
def dump_knn(nbrs,fn):
    import joblib, dill
    import datetime
    
    n = str(datetime.datetime.now())
    
    with open(fn,"wb") as f:
        dill.dump(nbrs,f)
    
    
def load_knn(fn):
    import joblib, dill
    with open(fn,"rb") as f:
        nbrs = dill.load(f)
    return nbrs   



if __name__=="__main__":
    weights = [0.5,1,3,4]
    #weights = [1,2,3,4]
    ll = []
    for two_octaves in range(5):
        alphaPitch = getProbsFromWeights(weights)
        pitches,durations,volumes,isPauses = generateNotes(two_octaves)
        notes = list(zip(pitches,durations,volumes,isPauses))
        nbrs,notes = get_knn_model_notes(notes,alphaPitch)
        print(nbrs)
        print(len(notes))
        ll.append((nbrs,notes))
    fn = "./knn_models/knn-"+"_".join([str(x) for x in weights])+".pkl"    
    dump_knn(ll,fn)
    