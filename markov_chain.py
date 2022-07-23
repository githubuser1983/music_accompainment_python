import music21 as m21
from itertools import product

pitchToZ12 = dict(zip(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],range(12)))
Z12ToPitch = dict(zip(range(12),["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]))

import numpy as np
def kernPause(a1,a2):
    return  1*(a1==a2)

def kernPitch(k1,k2):
    q = getRational(k2-k1)
    a,b = q.numerator(),q.denominator()
    #print(a,b)
    return gcd(a,b)**2/(a*b)

def kernDuration(k1,k2):
    return  min(k1,k2)/max(k1,k2)

def kernVolume(k1,k2):
    return min(k1,k2)/max(k1,k2)

def getRational(k):
    alpha = 2**(1/12.0)
    x = RDF(alpha**k).n(50)
    return x.nearby_rational(max_error=0.01*x)

def kern(t1,t2):
    import numpy as np
    pitch1,duration1,volume1,isPause1 = t1
    pitch2,duration2,volume2,isPause2 = t2
    weights = [1,2,3,4]
    #weights = np.array(weights)/np.sum(weights)
    #print(weights)
    tt = list(zip(t1,t2))
    kerns = [kernPitch, kernDuration, kernVolume, kernPause]
    x = np.sum([weights[i]*kerns[i](tt[i][0],tt[i][1]) for i in range(4)])
    #print(x)
    return x

def kernChord(c1,c2):
    return 1/(len(c2)*len(c1))*sum([kern(x,y) for x in c1 for y in c2])

def findNearestDuration(duration,durationslist):
    return sorted([(abs(duration-nv),nv) for nv in durationslist])[0][1]

def xml_to_list(xml):
    xml_data = m21.converter.parse(xml)
    score = []
    for part in xml_data.parts:
        parts = []
        for note in part.recurse().notesAndRests:
            if note.isRest:
                start = note.offset
                duration = float(note.quarterLength)/4.0
                vol = 32 #note.volume.velocity
                pitches= tuple([-1])
                parts.append(tuple([pitches,duration,vol,1]))
            elif type(note)==m21.chord.Chord:
                pitches = sorted([e.pitch.midi for e in note]) # todo: think about chords
                vol = int(note[0].volume.velocity)
                duration = float(note.duration.quarterLength)/4.0
                parts.append(tuple([tuple(pitches),duration,vol,0]))
            else:
                #print(note)
                start = note.offset
                duration = float(note.quarterLength)/4.0
                pitches = tuple([note.pitch.midi])
                #print(pitch,duration,note.volume)
                vol = note.volume.velocity
                if vol is None:
                    vol = int(note.volume.realized * 127)
                parts.append(tuple([pitches,duration,vol,0]) )
        score.append(parts)        
    return score



def getNearestVolume(vol):
    return sorted([(abs(vol-v),v) for v in volumelist])[0][1]

def parseXml(fp):
    return xml_to_list(fp)

def ngrams(inp, n):
    output = []
    for i in range(len(inp)-n+1):
        output.append(tuple(inp[i:(i+n)]))
    return output

#print(ngrams(list(range(7)),3))


from collections import Counter
import numpy as np


def zeroMatDict(ll):
    from itertools import product
    possible_ll = list(product(ll,ll))
    dd = dict([])
    for x in possible_ll:
        dd[x] = 0
    return dd    
    
    

def getCountValue(counter,possibilities,key):
    x,y = key
    if x in possibilities and y in possibilities:
        if key in counter.keys():
            return counter[key]
        else:
            return 0
    #print(possibilities,key)    
    return 0   



def getProbValue(counter,possibilities,key):
    x,y = key
    #print(x,y,key)
    cntv = getCountValue(counter,possibilities,key)
    s = sum([getCountValue(counter,possibilities,(x,Y)) for Y in possibilities])
    if s ==0:
        s=1
    val = float(cntv*1.0/s)
    return val

def getProbValues(counter,possibilities,x):
    return [getProbValue(counter,possibilities,(x,y)) for y in possibilities]



#print(bigram_pitches)
#print(bigram_durations)
def power(a,b):
    return np.exp(b*np.log(a))


def generateNext(counter,possibilities,current):
    probs = getProbValues(counter,possibilities,current)
    #print(current,probs)
    N = len(probs)
    if abs(np.sum(probs))<0.01:
        probs = [1.0/N for n in range(N)]
    #print(probs)    
    ch  = np.random.choice(range(N),p=probs)
    x = possibilities[ch]
    #if type(x)==type((1,2)) and len(x)>1:
    #    while current[-1]!=x[0]:
    #        ch  = np.random.choice(range(N),p=probs)
    #        x = possibilities[ch]
    #        #print(x)
    return x
        


def getPossibilities(ll,NMarkov=2):
    from itertools import product
    return [tuple(x) for x in list(product(*((NMarkov)*[ll])))]

def getNextWithKnn(bigram,possibilities,start_notes,NMarkov,NCandidates=10,rev=False):
    nxts = []
    possibs = [ p for p in possibilities if start_notes[-(NMarkov-1):]==p[0:(NMarkov-1)]]
    inv_possibs  = [ p for p in possibilities if start_notes[-(NMarkov-1):]!=p[0:(NMarkov-1)]]
    for k in range(NCandidates):
        print(len(possibs))
        if len(possibs)>0:
            nxt = generateNext(bigram,possibs,start_notes)
        else:
            nxt = choice(inv_possibs)
        nxts.append(nxt)
    return sorted([(kernChord(nxt,start_notes),nxt) for  nxt in nxts],reverse=rev)[0][1]

def getNext(bigram,possibilities,start_notes,NMarkov):
    import numpy as np
    possibs = [ p for p in possibilities if start_notes[-(NMarkov-1):]==p[0:(NMarkov-1)]]
    inv_possibs  = [ p for p in possibilities if start_notes[-(NMarkov-1):]!=p[0:(NMarkov-1)]]
    print(len(possibs))
    if len(possibs)>0:
        nxt = generateNext(bigram,possibs,start_notes)
    else:
        rng = range(len(inv_possibs))
        rng_nxt = np.random.choice(rng)
        nxt = inv_possibs[rng_nxt]
    return nxt
    

def chordOrNoteOrRest(pitches):
    r = pitches
    if len(r)==1 and r[0]!=-1:
        pitch = r[0]%12
        octave = r[0]//12
        n0 = m21.note.Note(Z12ToPitch[pitch]+str(octave))
    elif len(r)==1 and r[0]==-1:
        n0 = m21.note.Rest()
    else:
        octave = r[0]//12
        n0 = m21.chord.Chord([Z12ToPitch[rr%12]+str(octave) for rr in r])
    return n0       

def generate_from_file(NMarkov=2, tempo=70,nrMinutes=12,inputfn="./midi/una_mattina.mid",outputfn="./midi/markov.mid"):
    import music21 as m21
    scores = parseXml(inputfn)
    from itertools import product

    score = m21.stream.Score()
    tm = m21.tempo.MetronomeMark(number=tempo)
    score.append(tm)
    
    for j in range(len(scores)):

        sc = [tuple(x) for x in scores[j]]
        #print(ngrams(sc,NMarkov))
        possibilities = list(sorted(frozenset(ngrams(sc,NMarkov))))
    
        bigram = Counter(ngrams(sc,NMarkov+1))
        #print(bigram)
        start_notes = tuple(sc[0:(NMarkov)])
    
        print(start_notes)
        #counters = [bigram_pitches,bigram_octaves,bigram_durations,bigram_volumes,bigram_pauses]
        #possibs = [pitchlist,octavelist,durationslist,volumelist,pauseslist]

        ll = []

        minBars = int(tempo*nrMinutes/4)
        sumBars = 0
        while sumBars < minBars:
            note = []
            Note = []
            nxt = getNext(bigram,possibilities,start_notes,NMarkov)
            #nxt = generateNext(bigram,possibilities,start_notes)
            #print(start_notes,nxt)
            start_notes = nxt
            duration = nxt[-1][1]
            print(j,duration,sumBars)
            sumBars += duration
            ll.append(nxt[-1])    
        #print(ll) 

        lh = m21.stream.Part()
        lh.append(m21.instrument.Piano())       
        notesLH = []
        for i in range(len(ll)):
            #print(ll[i])
            pitches,duration,volume,pause = ll[i]
            n0 = chordOrNoteOrRest(pitches)
            n0.duration = m21.duration.Duration(duration*4.0)
            if not type(n0)==m21.note.Rest:
                n0.volume.velocity = int(volume)
            #n0.duration.quarterLength = float(choice([0.5,0.25,1]))
            notesLH.append(n0)
        for n in notesLH:
            lh.append(n)
        score.append(lh)
    print(len(score.parts))    
    score_name = outputfn
    score.write('midi', fp=score_name)  
    score.write("musicxml",fp=score_name+".xml")

import sys
NMarkov, tempo, nrMinutes, inputfn, outputfn = sys.argv[1:]
generate_from_file(NMarkov=int(NMarkov),tempo=int(tempo),nrMinutes=int(nrMinutes),inputfn=inputfn,outputfn=outputfn)        