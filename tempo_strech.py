import math
import numpy as np
import wave
import struct
from matplotlib import pyplot as plt


def Tempo_Analyse(wf0):
    wf0.rewind()
    data0 = wf0.readframes(-1)

    s0 = np.frombuffer(data0, dtype = 'int16')
    dt = s0 / (2 ** 15)

    fs = wf0.getframerate()
    samples = wf0.getnframes()

    frames = 256
    sample_max = samples - (samples % frames)
    frame_max = sample_max // frames

    frame_list = np.split(dt[:sample_max], frame_max)
    amp_list   = np.array([np.sqrt(sum(x ** 2)) for x in frame_list])

    plt.plot(amp_list)
    plt.show()

    amp_diff_list = amp_list[1:] - amp_list[:-1]
    amp = np.vectorize(max)(amp_diff_list, 0)

    match_list = []
    N = len(amp)
    f_frame = fs / frames

    for bpm in range(60, 300):
        f_bpm   = bpm / 60

        phase_array = 2 * np.pi * f_bpm * np.arange(N) / f_frame
        phase_list = phase_array.tolist()
        sin_match   = (1 / N) * sum(amp * np.sin(phase_list))
        cos_match   = (1 / N) * sum(amp * np.cos(phase_list))

        match = np.sqrt(sin_match ** 2 + cos_match ** 2)
        match_list.append(match)


    most_match = match_list.index(max(match_list))
    bpm = most_match + 60

    plt.plot(np.arange(60, 60 + len(match_list)), match_list)
    plt.xlabel('BPM')
    plt.ylabel('Match Level')
    plt.show()

    return bpm

def TimeStretch_MONO_HI(wf0, rate, ch):
    wf0.rewind()
    data0 = wf0.readframes(-1)

    s0 = np.frombuffer(data0, dtype = 'int16')

    fname = 'out_mono_hi.wav'
    width = wf0.getsampwidth()
    fs = wf0.getframerate()
    samples = int(wf0.getnframes() / rate) + 1
    s1 = np.zeros(samples, dtype = 'int16')

    template_size = int(fs * 0.01)
    pmin = int(fs * 0.005)
    pmax = int(fs * 0.02)

    x = [0] * template_size
    y = [0] * template_size
    r = [0] * (pmax + 1)

    offset0 = 0
    offset1 = 0

    print(len(s0))

    while offset0 + pmax * 2 < len(s0):
        for n in range(0, template_size):
            x[n] = s0[offset0 + n]

        rmax = 0
        p = pmin
        for m in range(pmin, pmax + 1):
            for n in range(0, template_size):
                y[n] = s0[offset0 + m + n]
            r[m] = 0
            for n in range(0, template_size):
                r[m] += int(x[n]) * int(y[n])
            if r[m] > rmax:
                rmax = r[m]
                p = m
        for n in range(0, p):
            s1[offset1 + n] = s0[offset0 + n] * (p - n) / p
            s1[offset1 + n] += s0[offset0 + p + n] * n / p 

        q = np.round(p / (rate - 1))
        for n in range(int(p), int(q)):
            if offset0 + p + n >= len(s0):
                break
            s1[offset1 + n] = s0[offset0 + p + n]

        offset0 += int(p + q)
        offset1 += int(q)

        print(offset0)

    s1 = np.rint(s1)
    s1 = s1.astype(np.int16)
    data1 = struct.pack("h" * samples, *s1)

    wf1 = wave.open(fname, 'w')

    wf1.setparams((
        ch,
        width,
        fs, 
        samples,
        "NONE", "NONE"
    ))
    wf1.writeframes(data1)

    wf0.close()
    wf1.close()

    plt.plot(s1)
    plt.show()

def TimeStretch_MONO_LOW(wf0, rate, ch):
    wf0.rewind()
    data0 = wf0.readframes(-1)

    s0 = np.frombuffer(data0, dtype = 'int16')

    fname = 'out_mono_low.wav'
    ch = wf0.getnchannels()
    width = wf0.getsampwidth()
    fs = wf0.getframerate()
    samples = int(wf0.getnframes() / rate) + 1
    s1 = np.zeros(samples, dtype = 'int16')

    template_size = int(fs * 0.01)
    pmin = int(fs * 0.005)
    pmax = int(fs * 0.02)

    x = [0] * template_size
    y = [0] * template_size
    r = [0] * (pmax + 1)

    offset0 = 0
    offset1 = 0

    print(len(s0))

    while offset0 + pmax * 2 < len(s0):
        for n in range(0, template_size):
            x[n] = s0[offset0 + n]

        rmax = 0
        p = pmin
        for m in range(pmin, pmax + 1):
            for n in range(0, template_size):
                y[n] = s0[offset0 + m + n]
            r[m] = 0
            for n in range(0, template_size):
                r[m] += int(x[n]) * int(y[n])
            if r[m] > rmax:
                rmax = r[m]
                p = m

        for n in range(0, p):
            s1[offset1 + n] = s0[offset0 + n]

        for n in range(0, p):
            s1[offset1 + p + n] = s0[offset0 + p + n] * (p - n) / p
            s1[offset1 + p + n] += s0[offset0 + n] * n / p

        q = np.round(p * rate / (1 - rate))
        for n in range(int(p), int(q)):
            if offset0 + n >= len(s0):
                break
            s1[offset1 + p + n] = s0[offset0 + n]

        offset0 += int(q)
        offset1 += int(p + q)

        print(offset0)

    s1 = np.rint(s1)
    s1 = s1.astype(np.int16)
    data1 = struct.pack("h" * samples, *s1)

    wf1 = wave.open(fname, 'w')

    wf1.setparams((
        ch,
        width,
        fs,
        samples,
        "NONE", "NONE"
    ))
    wf1.writeframes(data1)

    wf0.close()
    wf1.close()

    plt.plot(s1)
    plt.show()

def TimeStretch_STEREO_HI(wf0, rate, ch):
    wf0.rewind()
    data0 = wf0.readframes(-1)

    s0 = np.frombuffer(data0, dtype = 'int16')

    ls = s0[::2]
    rs = s0[1::2]

    fname = 'out_stereo_hi.wav'
    width = wf0.getsampwidth()
    fs = wf0.getframerate()
    samples = int(wf0.getnframes() / rate) + 1
    ls1 = np.zeros(samples, dtype = 'int16')
    rs1 = np.zeros(samples, dtype = 'int16')

    template_size = int(fs * 0.01) 
    pmin = int(fs * 0.005)
    pmax = int(fs * 0.02)

    lx = [0] * template_size
    rx = [0] * template_size
    ly = [0] * template_size
    ry = [0] * template_size
    r = [0] * (pmax + 1)

    offset0 = 0
    offset1 = 0
    print('Left: ', len(ls))

    
    while offset0 + pmax * 2 < len(ls):
        for n in range(0, template_size):
            lx[n] = ls[offset0 + n]  

        rmax = 0
        p = pmin
        for m in range(pmin, pmax + 1):
            for n in range(0, template_size):
                ly[n] = ls[offset0 + m + n] 
            r[m] = 0
            for n in range(0, template_size):
                r[m] += int(lx[n]) * int(ly[n])
            if r[m] > rmax:
                rmax = r[m]
                p = m

        for n in range(0, p):
            ls1[offset1 + n] = ls[offset0 + n] * (p - n) / p
            ls1[offset1 + n] += ls[offset0 + p + n] * n / p

        q = np.round(p / (rate - 1))
        for n in range(int(p), int(q)):
            if offset0 + p + n >= len(ls):
                break
            ls1[offset1 + n] = ls[offset0 + p + n]

        offset0 += int(p + q)
        offset1 += int(q)

        print(offset0)


    offset0 = 0
    offset1 = 0
    print('Right : ', len(rs))

    while offset0 + pmax * 2 < len(rs):
        for n in range(0, template_size):
            rx[n] = rs[offset0 + n]

        rmax = 0
        p = pmin
        for m in range(pmin, pmax + 1):
            for n in range(0, template_size):
                ry[n] = rs[offset0 + m + n]
            r[m] = 0
            for n in range(0, template_size):
                r[m] += int(rx[n]) * int(ry[n])
            if r[m] > rmax:
                rmax = r[m]
                p = m

        for n in range(0, p):
            rs1[offset1 + n] = rs[offset0 + n] * (p - n) / p 
            rs1[offset1 + n] += rs[offset0 + p + n] * n / p

        q = np.round(p / (rate - 1))
        for n in range(int(p), int(q)):
            if offset0 + p + n >= len(rs):
                break
            rs1[offset1 + n] = rs[offset0 + p + n]

        offset0 += int(p + q)
        offset1 += int(q)

        print(offset0)


    ls1 = np.rint(ls1)
    rs1 = np.rint(rs1) 
    s1 = np.zeros(samples * 2, dtype = 'int16')
    s1[::2] = ls1
    s1[1::2] = rs1 
    s1 = s1.astype(np.int16)
    data1 = struct.pack("h" * samples * 2, *s1)

    wf1 = wave.open(fname, 'w')

    wf1.setparams((
        ch,
        width, 
        fs,
        samples * 2,
        "NONE", "NONE"
    ))
    wf1.writeframes(data1)

    wf0.close()
    wf1.close()

    plt.plot(s1)
    plt.show()

def TimeStretch_STEREO_LOW(wf0, rate, ch):
    wf0.rewind()
    data0 = wf0.readframes(-1)

    s0 = np.frombuffer(data0, dtype = 'int16')

    ls = s0[::2]
    rs = s0[1::2]

    fname = 'out_stereo_low.wav'
    width = wf0.getsampwidth()
    fs = wf0.getframerate()
    samples = int(wf0.getnframes() / rate) + 1 
    ls1 = np.zeros(samples, dtype = 'int16') 
    rs1 = np.zeros(samples, dtype = 'int16')

    template_size = int(fs * 0.01)
    pmin = int(fs * 0.005)
    pmax = int(fs * 0.02)

    lx = [0] * template_size
    rx = [0] * template_size
    ly = [0] * template_size
    ry = [0] * template_size
    r = [0] * (pmax + 1)

    offset0 = 0
    offset1 = 0
    print('Left : ', len(ls))

    # Left
    while offset0 + pmax * 2 < len(ls):
        for n in range(0, template_size):
            lx[n] = ls[offset0 + n]

        rmax = 0
        p = pmin
        for m in range(pmin, pmax + 1):
            for n in range(0, template_size):
                ly[n] = ls[offset0 + m + n] 
            r[m] = 0
            for n in range(0, template_size):
                r[m] += int(lx[n]) * int(ly[n]) 
            if r[m] > rmax:
                rmax = r[m]
                p = m

        for n in range(0, p):
            ls1[offset1 + p + n] = ls[offset0 + p + n] * (p - n) / p
            ls1[offset1 + p + n] += ls[offset0 + n] * n / p

        q = np.round(p * rate / (1 - rate))
        for n in range(int(p), int(q)):
            if offset0 + n >= len(ls):
                break
            ls1[offset1 + p + n] = ls[offset0 + n]

        offset0 += int(q)
        offset1 += int(p + q)

        print(offset0)


    offset0 = 0
    offset1 = 0
    print('Right : ', len(rs))

    while offset0 + pmax * 2 < len(rs):
        for n in range(0, template_size):
            rx[n] = rs[offset0 + n] 

        rmax = 0
        p = pmin
        for m in range(pmin, pmax + 1):
            for n in range(0, template_size):
                ry[n] = rs[offset0 + m + n]
            r[m] = 0
            for n in range(0, template_size):
                r[m] += int(rx[n]) * int(ry[n])
            if r[m] > rmax:
                rmax = r[m]
                p = m

        for n in range(0, p):
            rs1[offset1 + p + n] = rs[offset0 + p + n] * (p - n) / p
            rs1[offset1 + p + n] += rs[offset0 + n] * n / p

        q = np.round(p * rate / (1 - rate))
        for n in range(int(p), int(q)):
            if offset0 + n >= len(rs):
                break
            rs1[offset1 + p + n] = rs[offset0 + n]

        offset0 += int(q)
        offset1 += int(p + q)

        print(offset0)


    ls1 = np.rint(ls1)
    rs1 = np.rint(rs1)
    s1 = np.zeros(samples * 2, dtype = 'int16')
    s1[::2] = ls1
    s1[1::2] = rs1
    s1 = s1.astype(np.int16)
    data1 = struct.pack("h" * samples * 2, *s1)

    wf1 = wave.open(fname, 'w')

    wf1.setparams((
        ch,
        width,
        fs,
        samples * 2,
        "NONE", "NONE"
    ))
    wf1.writeframes(data1)

    wf0.close()
    wf1.close()

    plt.plot(s1)
    plt.show()
    

if (__name__ == '__main__'):
    fname = 'crazy.wav'
    wf = wave.open(fname, 'r')
    ch = wf.getnchannels() 

    bpm = Tempo_Analyse(wf)
    print("BPM : ", bpm)

    x = input('New BPM : ')
    nw_bpm = int(x)
    rate = nw_bpm / bpm
    print('rate : ', rate)

    if rate > 1:
        if ch == 1:
            TimeStretch_MONO_HI(wf, rate, ch)
        else:
            TimeStretch_STEREO_HI(wf, rate, ch)
    elif rate >= 0.5 and rate < 1:
        if ch == 1:
            TimeStretch_MONO_LOW(wf, rate, ch)
        else:
            TimeStretch_STEREO_LOW(wf, rate, ch)
    else:
        print('Error: invalid literal for Tempo')
