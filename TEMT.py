#!/usr/bin/python

"""Description: TEMT

Copyright (c) 2012 Yi Li <yil8@uci.edu>

This code is free software; you can redistribute it and/or modify it
under the terms of the Artistic License 2.0 (see the file LICENSE included
with the distribution).

"""

######################################################
#                   python modules                   #
######################################################


import numpy
import time
from optparse import OptionParser
from pygr import seqdb
from scipy import float32
from scipy.stats import norm
from math import pow, log, log10

#ORDER = 3
NUC_NUM = 4
WIN_SIZE = 21
SURROUND = 10
NT_dict = {'A':0, 'C':1, 'G':2, 'T':3}
LEN_BINS = [0, 791, 1265, 1707, 2433, 999999999]
LEN_BINS_NUM = 5
POS_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
POS_BINS_NUM = 10

step_size = 0.85
frag_len_mean = 200
frag_len_std = 80
frag_len_max = 800
prior_intens = 10


######################################################
#              Bias related functions                #
######################################################





def reverseSeq(seq):
    return seq[::-1]
    
def getBinIndex(val, bin_type):
    
    bin_index = 0
    if bin_type == 0:
        bins = LEN_BINS
        bins_num = LEN_BINS_NUM
    elif bin_type == 1:
        bins = POS_BINS
        bins_num = POS_BINS_NUM
        
    for i in range(0, bins_num):
        if val >= bins[i] and val < bins[i+1]:
            bin_index = i
            break
    
    return bin_index


def getStartEnd(pos, seq_len, tr_len, flag): # pos: position index within transcript, if 5'end read then leftmost coordinate; 
                                                       # if 3'end read then rightmost coordinate
    if pos - SURROUND < 0:
        start_5_end = WIN_SIZE - seq_len
        end_5_end = WIN_SIZE
        start_3_end = 0
        end_3_end = seq_len            
    elif pos + SURROUND > tr_len - 1:
        start_5_end = 0
        end_5_end = seq_len
        start_3_end = WIN_SIZE - seq_len
        end_3_end = WIN_SIZE
    else:
        start_5_end = 0
        start_3_end = 0
        end_5_end = WIN_SIZE
        end_3_end = WIN_SIZE
        
    # start and end are position indexs within window
    if flag == 0: # flag == 0 means 5'end
        start = start_5_end 
        end = end_5_end
    elif flag == 1: # flag == 1 means 3'end
        start = start_3_end
        end = end_3_end
        
    return (start, end)
    

####sequence specific bias class#################


class seqBias:    
    def __init__(self):
        
        self._counts_bg = numpy.ones((1, NUC_NUM), dtype=float32) #expected counts, which is counted based on transcripts background
        self._counts_fg_5_end = numpy.ones((WIN_SIZE, NUC_NUM), dtype=float32) #observed counts, which is counted based on 5'end of reads set forground
        self._counts_fg_3_end = numpy.ones((WIN_SIZE, NUC_NUM), dtype=float32) #observed counts, which is counted based on 3'end of reads set forground
        self._sums_bg = numpy.ones(1, dtype=float32)*4 #expected conditional sums, which is counted based on transcripts background
        self._sums_fg_5_end = numpy.ones(WIN_SIZE, dtype=float32)*4 #observed sums, which is counted based on 5'end of reads set forground
        self._sums_fg_3_end = numpy.ones(WIN_SIZE, dtype=float32)*4 #observed sums, which is counted based on 3'end of reads set forground        
        
    
    def getCountsBG(self):
        return self._counts_bg
    
    def getSumsBG(self):
        return self._sums_bg
    
    def getCountsFG(self, flag):
        if flag == 0:
            return self._counts_fg_5_end
        elif flag == 1:
            return self._counts_fg_3_end
        
    def getSumsFG(self, flag):
        if flag == 0:
            return self._sums_fg_5_end
        elif flag == 1:
            return self._sums_fg_3_end
    
    
    def updateCountsBG(self, seq):
        
        seq_len = len(seq)
        for i in range(0, seq_len):
            NT_char = seq[i]
            if NT_char != 'N': #pass 'N' in sequence
                NT_int = NT_dict[NT_char]
                self._counts_bg[0, NT_int] = self._counts_bg[0, NT_int] + 1
                self._sums_bg[0] = self._sums_bg[0] + 1
    
    
    def copyCountsBG(self, bias):
        self._counts_bg = numpy.array(bias.getCountsBG())
        self._sums_bg = numpy.array(bias.getSumsBG())
    
                
    def updateCountsFG(self, seq, pos, tr_len, flag, mass = 1):
        
        seq_len = len(seq)
        seq_rev = reverseSeq(seq)
        start, end = getStartEnd(pos, seq_len, tr_len, flag) # start and end are position indexs within window
        i = 0 #i is the position index for seq
        for j in range(start, end): #j is the position index for window
            if flag == 0: # flag == 0 means 5'end
                NT_char = seq[i]
                if NT_char != 'N':
                    NT_int = NT_dict[NT_char]
                    self._counts_fg_5_end[j, NT_int] = self._counts_fg_5_end[j, NT_int] + mass
                    self._sums_fg_5_end[j] = self._sums_fg_5_end[j] + mass
            elif flag == 1: # flag == 1 means 3'end
                NT_char = seq_rev[i]
                if NT_char != 'N':
                    NT_int = NT_dict[NT_char]
                    self._counts_fg_3_end[j, NT_int] = self._counts_fg_3_end[j, NT_int] + mass
                    self._sums_fg_3_end[j] = self._sums_fg_3_end[j] + mass
            
            i = i + 1
            
    
    def seqLogProbBG(self, seq, pos, tr_len): #log10 space, 
        
        seq_len = len(seq)
        p_seq_log = 0 + log10(0.25)*(WIN_SIZE - seq_len)
        for i in range(0, seq_len):
            NT_char = seq[i]
            if NT_char == 'N':
                p_cond = 0.25
            else:
                NT_int = NT_dict[NT_char]
                p_cond = self._counts_bg[0, NT_int]/self._sums_bg[0]
            
            p_seq_log = p_seq_log + log10(p_cond)    
                
        return p_seq_log
    
    def seqLogProbFG(self, seq, pos, tr_len, flag): #log10 space, pos: position index within transcript
        
        seq_len = len(seq)
        seq_rev = reverseSeq(seq)
        start, end = getStartEnd(pos, seq_len, tr_len, flag)
        i = 0 #i is the position index for seq
        p_seq_log = 0 + log10(0.25)*(WIN_SIZE - seq_len)
        for j in range(start, end): #j is the position index for window
            if flag == 0:
                NT_char = seq[i]
                if NT_char == 'N':
                    p_cond = 0.25
                else:
                    NT_int = NT_dict[NT_char]
                    p_cond = self._counts_fg_5_end[j, NT_int]/self._sums_fg_5_end[j]
            elif flag == 1:
                NT_char = seq_rev[i]
                if NT_char == 'N':
                    p_cond = 0.25
                else:
                    NT_int = NT_dict[NT_char]
                    p_cond = self._counts_fg_3_end[j, NT_int]/self._sums_fg_3_end[j]

            p_seq_log = p_seq_log + log10(p_cond)
            i = i + 1
                
        return p_seq_log
    
    def seqWeight(self, seq, pos, tr_len, flag):
        
        p_BG_log = self.seqLogProbBG(seq, pos, tr_len)
        p_FG_log = self.seqLogProbFG(seq, pos, tr_len, flag)
        
        weight = pow(10, p_FG_log - p_BG_log)
        
        return weight
    

####positional bias class#################


class posBias: 
    def __init__(self):
        
        self._counts_fg_5_end = numpy.ones((LEN_BINS_NUM, POS_BINS_NUM), dtype=float32) #observed counts, which is counted based on 5'end of reads set forground
        self._counts_fg_3_end = numpy.ones((LEN_BINS_NUM, POS_BINS_NUM), dtype=float32) #observed counts, which is counted based on 3'end of reads set forground

    def getCountsFG(self, flag):
        if flag == 0:
            return self._counts_fg_5_end
        elif flag == 1:
            return self._counts_fg_3_end


    def updateCountsFG(self, pos, tr_len, flag, mass = 1):
        
        tr_len_bin_i = getBinIndex(tr_len, 0)
        tr_pos_bin_i = getBinIndex(pos/float32(tr_len), 1)
        if flag == 0:
            self._counts_fg_5_end[tr_len_bin_i, tr_pos_bin_i] = self._counts_fg_5_end[tr_len_bin_i, tr_pos_bin_i] + mass
        elif flag == 1:
            self._counts_fg_3_end[tr_len_bin_i, tr_pos_bin_i] = self._counts_fg_3_end[tr_len_bin_i, tr_pos_bin_i] + mass
            
    def posProbFG(self, pos, tr_len, flag): #linear space, pos: position index within transcript
        
        tr_len_bin_i = getBinIndex(tr_len, 0)
        tr_pos_bin_i = getBinIndex(pos/float32(tr_len), 1)
        if flag == 0:
            p = self._counts_fg_5_end[tr_len_bin_i, tr_pos_bin_i]/numpy.sum(self._counts_fg_5_end[tr_len_bin_i, :])
        elif flag == 1:
            p = self._counts_fg_3_end[tr_len_bin_i, tr_pos_bin_i]/numpy.sum(self._counts_fg_3_end[tr_len_bin_i, :])
            
        return p
    
    def posWeight(self, pos, tr_len, flag):
        
        p_FG = self.posProbFG(pos, tr_len, flag)
        p_BG = 1/float32(POS_BINS_NUM)
        weight = p_FG/p_BG
        
        return weight



######################################################
#                  major functions                   #
######################################################


def getEffLen(tr_len):
    
    tr_len_eff = 0
    normal_const = numpy.sum(norm.pdf(range(1, min(frag_len_max, tr_len)+1), frag_len_mean, frag_len_std))
    for l in range(1, min(frag_len_max, tr_len)+1):
        tr_len_eff = tr_len_eff + norm.pdf(l, frag_len_mean, frag_len_std)*(tr_len - l + 1)/normal_const

    return tr_len_eff

def getWinSeq(left, tr_seq, read_len, frag_end):
    right = left + int(read_len) - 1
    if frag_end == 0: # 5'end read
        pos = left
    elif frag_end == 1: # 3'end read
        pos = right
            
    tr_len = len(tr_seq)
    if pos - SURROUND < 0:
        win_seq = tr_seq[0:pos+SURROUND+1]
    elif pos + SURROUND > tr_len - 1:
        win_seq = tr_seq[pos-SURROUND:tr_len]
    else:
        win_seq = tr_seq[pos-SURROUND:pos+SURROUND+1]    
    
    return (win_seq, pos)

def getReadWeight(left_read, frag_end_read, read_len, tr_ID, tr_num, intranscripts, bias_seq, bias_pos):
    w_i = numpy.array(numpy.ones(tr_num), dtype=float32) #initial bias weight of read_i for each transcript
        
    for index in left_read.keys():
        left_i = left_read[index]
        frag_end_i = frag_end_read[index]
        tr_seq_i = str(intranscripts[tr_ID[index]]).upper()
        tr_len_i = len(tr_seq_i)
        win_seq_i, pos_i = getWinSeq(left_i, tr_seq_i, read_len, frag_end_i)
        w_i[index] = bias_seq.seqWeight(win_seq_i, pos_i, tr_len_i, frag_end_i)*bias_pos.posWeight(pos_i, tr_len_i, frag_end_i)
    
    return w_i

def updateBiasFG(left_read, frag_end_read, read_len, tr_ID, tr_num, intranscripts, mass, bias_seq, bias_pos):
    
    for index in left_read.keys():
        left_i = left_read[index]
        frag_end_i = frag_end_read[index]
        tr_seq_i = str(intranscripts[tr_ID[index]]).upper()
        tr_len_i = len(tr_seq_i)
        win_seq_i, pos_i = getWinSeq(left_i, tr_seq_i, read_len, frag_end_i)
        bias_seq.updateCountsFG(win_seq_i, pos_i, tr_len_i, frag_end_i, mass)
        bias_pos.updateCountsFG(pos_i, tr_len_i, frag_end_i, mass)
    
    return (bias_seq, bias_pos)

def getAvgBias(tr_seq, bias_seq, bias_pos):
    
    tr_len = len(tr_seq)
    weight_sum_5_end = 0
    weight_sum_3_end = 0
        
    for pos in range(0, tr_len):
        if pos - SURROUND < 0:
            win_seq = tr_seq[0:pos+SURROUND+1]
        elif pos + SURROUND > tr_len - 1:
            win_seq = tr_seq[pos-SURROUND:tr_len]
        else:
            win_seq = tr_seq[pos-SURROUND:pos+SURROUND+1]
         
        weight_sum_5_end = weight_sum_5_end + bias_seq.seqWeight(win_seq, pos, tr_len, 0)*bias_pos.posWeight(pos, tr_len, 0)
        weight_sum_3_end = weight_sum_3_end + bias_seq.seqWeight(win_seq, pos, tr_len, 1)*bias_pos.posWeight(pos, tr_len, 1)
                       
    tr_len_bias = (weight_sum_5_end + weight_sum_3_end)/2
    tr_avg_bias = weight_sum_5_end*weight_sum_3_end/(tr_len*tr_len)
        
    return (tr_avg_bias, tr_len_bias)

def pasSAMLine(line):
    if line[0:1] == '@':
        line_type = 'h'
        return (None, None, None, None, None, line_type)
    elif line[0:1] != '@':
        line_type = 'r'
        fields = line.strip('\n').split('\t')
        read_ID, FLAG, tr_ID, left = fields[0:4] # left: the leftmost coordinate of read within transcript from SAM line 1-base
        FLAG = int(FLAG)
        left = int(left) - 1 # all calculation in python is 0-based 
        read_seq = fields[9]
        if FLAG == 0:
            frag_end = 0 # 5'end read of the fragment
        elif FLAG == 16:
            frag_end = 1 # 3'end read of the fragment
        else:
            frag_end = None
        return (read_ID, tr_ID, frag_end, left, read_seq, line_type)
        

def pasRead(inreads, read_ID_current, tr_ID, tr_num):
    
    Y_i = numpy.array(numpy.zeros(tr_num))
    left_read_i = {}
    frag_end_read_i = {}
    while True:
        line = inreads.readline()
        if len(line) == 0:
            break
        
        read_ID_i, tr_ID_i, frag_end_i, left_i, read_seq_i, line_type_i = pasSAMLine(line) 
        if line_type_i == 'r' and tr_ID_i != '*': # read line and is not non-mappable read
            if read_ID_i != read_ID_current:
                if read_ID_current == '':
                    read_ID_current = read_ID_i
                    index_i = tr_ID.index(tr_ID_i)
                    Y_i[index_i] = 1
                    left_read_i[index_i] = left_i
                    frag_end_read_i[index_i] = frag_end_i
                    continue
                
                read_ID_current = read_ID_i
                inreads.seek(-1*len(line), 1)
                return (Y_i, read_ID_current, left_read_i, frag_end_read_i, 0) #flag = 0 means not end of file
            elif read_ID_i == read_ID_current:
                index_i = tr_ID.index(tr_ID_i)
                Y_i[index_i] = 1
                left_read_i[index_i] = left_i
                frag_end_read_i[index_i] = frag_end_i
            
    return (Y_i, read_ID_current, left_read_i, frag_end_read_i, 1) #flag = 1 means end of file
    

def getReadsBias(inreads, filename, read_len, tr_ID, tr_num, intranscripts, bias_seq, bias_pos):
    i = 0
    read_ID_current = ''
    file_state = 0
    current_time = time.time()
    while file_state == 0:
        Y_i, read_ID_current, left_read_i, frag_end_read_i, file_state = pasRead(inreads, read_ID_current, tr_ID, tr_num)
        hit_num = len(left_read_i)
        mass_i = 1/float32(hit_num)
        bias_seq, bias_pos = updateBiasFG(left_read_i, frag_end_read_i, read_len, tr_ID, tr_num, intranscripts, mass_i, bias_seq, bias_pos)
        
        i = i + 1
        if i%10000 == 0:
            run_time = time.time() - current_time
            current_time = time.time()
            print '%s reads processed for counting reads bias of %s...' % (i, filename)
    
    inreads.seek(0)
    
    return (bias_seq, bias_pos)
    
def getTranscriptsBias(tr_ID, tr_num, intranscripts, bias_seq):
    current_time = time.time()
    for i in range(0, tr_num):
        tr_seq_i = str(intranscripts[tr_ID[i]]).upper()
        bias_seq.updateCountsBG(tr_seq_i)
    
        if (i+1)%1000 == 0:
            run_time = time.time() - current_time
            current_time = time.time()
            print '%s transcripts processed for bias counts...' % (i+1)
    
    return bias_seq


def main():
    usage = """usage: %prog [options] <-p pfile> <-m mfile> <-t tfile>  

Example: %prog -p pure.sam -m mix.sam -t transcripts.fasta -P 0.9 -l 75 -a type_a -b type_b -A 0 --bias-module 
"""
    parser = OptionParser(usage = usage, version = "1.01")
    
    parser.add_option("-p", "--pure-sample", dest="inreads_p", type="string", help="read alignment file of the pure sample in SAM format. REQUIRED")
    parser.add_option("-m", "--mixed-sample", dest="inreads_m", type="string", help="read alignment file of the mixed sample in SAM format. REQUIRED")
    parser.add_option("-t", "--transcripts", dest="intranscripts", type="string", help="reference transcripts file in fasta format. REQUIRED")
    parser.add_option("-a", "--type-a", dest="outreads_a", type="string", help="the name of the output file of cell type a, which is the only cell type of the pure sample. DEFAULT: \"type_a\"", default='type_a')
    parser.add_option("-b", "--type-b", dest="outreads_b", type="string", help="the name of the output file of cell type b, which is the second cell type within the mixed sample. DEFAULT: \"type_b\"", default='type_b')
    parser.add_option("-P", "--type-b-proportion", dest="b_prop", type="string", help="cell type b proportion. e.g. \"-P 0.9\". REQUIRED")
    parser.add_option("-l", "--read-length", dest="read_len", type="string", help="read length. REQUIRED")
    parser.add_option("-A", "--additional-rounds", dest="add_rounds", type="string", help="the number of addtional rounds of EM algorithm after the first online round. DEFAULT: 0", default='0')
    parser.add_option("--bias-module", dest="bias_flag", action="store_true", help="enable the positional and sequence specific bias module. DEFAULT: False", default=False)
    
    options, args = parser.parse_args()
    
    err_flag = False
    if options.inreads_p == None:
        print "Error: No read alignment file of pure sample!"
        err_flag = True
        
    if options.inreads_m == None:
        print "Error: No read alignment file of mixed sample!"
        err_flag = True

    if options.intranscripts == None:
        print "Error: No reference transcripts file!"
        err_flag = True
    
    if options.b_prop == None:
        print "Error: No cell type b proportion!"
        err_flag = True
    
    if options.b_prop == None:
        print "Error: No read length!"
        err_flag = True
    
    if err_flag == True:
        print ('\n')
        parser.print_help()
        parser.exit()
    
    inreads_p = open(options.inreads_p)
    filename_p = options.inreads_p.split('/')[len(options.inreads_p.split('/')) - 1]
    inreads_m = open(options.inreads_m)
    filename_m = options.inreads_m.split('/')[len(options.inreads_m.split('/')) - 1]
    intranscripts = seqdb.SequenceFileDB(options.intranscripts)
    outreads_a = open(options.outreads_a+'.temt', 'w')
    outreads_b = open(options.outreads_b+'.temt', 'w')
    read_len = float32(options.read_len)
    wa = 1-float(options.b_prop)
    wb = float(options.b_prop)
    add_rounds = int(options.add_rounds)
    bias_flag = options.bias_flag
    
    time_start = time.time()
    tr_ID = []
    tr_len = []
    tr_len_eff = []
    tr_avg_bias_a = []
    tr_avg_bias_x = []
    bias_seq_a = seqBias()
    bias_seq_x = seqBias()
    bias_pos_a = posBias()
    bias_pos_x = posBias()
    
    tr_ID = intranscripts.keys()
    tr_num = len(tr_ID)
    
####obtain reads bias counts#################

    if bias_flag == True:
        print 'obtain reads bias counts of %s...' % (filename_p)
        bias_seq_a, bias_pos_a = getReadsBias(inreads_p, filename_p, read_len, tr_ID, tr_num, intranscripts, bias_seq_a, bias_pos_a)
        print 'obtain reads bias counts of %s...' % (filename_m)
        bias_seq_x, bias_pos_x = getReadsBias(inreads_m, filename_m, read_len, tr_ID, tr_num, intranscripts, bias_seq_x, bias_pos_x)
    elif bias_flag == False:
        print 'skip counting reads bias step...'

####obtain transcripts bias counts#############

    if bias_flag == True:
        print 'obtain transcripts bias counts...'
        bias_seq_a = getTranscriptsBias(tr_ID, tr_num, intranscripts, bias_seq_a)
        bias_seq_x.copyCountsBG(bias_seq_a)
    elif bias_flag == False:
        print 'skip counting transcripts bias step...'
    
####calculating transcripts bias#############
    print 'calculating transcripts effective length...'
    current_time = time.time()
    for i in range(0, tr_num):        
        tr_seq_i = str(intranscripts[tr_ID[i]]).upper()
        tr_len_i = len(intranscripts[tr_ID[i]])
        tr_len.append(tr_len_i)
        tr_len_eff_i = getEffLen(tr_len_i)
        tr_len_eff.append(tr_len_eff_i)
        if bias_flag == True:
            tr_avg_bias_a_i, tr_len_bias_a_i = getAvgBias(tr_seq_i, bias_seq_a, bias_pos_a)
            tr_avg_bias_a.append(tr_avg_bias_a_i)
            tr_avg_bias_x_i, tr_len_bias_x_i = getAvgBias(tr_seq_i, bias_seq_x, bias_pos_x)
            tr_avg_bias_x.append(tr_avg_bias_x_i)
        
        if (i+1)%100 == 0:
            run_time = time.time() - current_time
            current_time = time.time()
            print '%s transcripts processed for effective length calculation...' % (i+1)
    
    tr_len = numpy.array(tr_len, dtype=float32)
    tr_len_eff = numpy.array(tr_len_eff, dtype=float32)
    tr_avg_bias_a = numpy.array(tr_avg_bias_a, dtype=float32)
    tr_avg_bias_x = numpy.array(tr_avg_bias_x, dtype=float32)
    
####Online EM step###########################

    alpha_a = numpy.array(numpy.ones(tr_num)/tr_num, dtype=float32) #initial probabilites of chosing 1 read from transcripts of normal cells
    alpha_b = numpy.array(numpy.ones(tr_num)/tr_num, dtype=float32) #initial probabilites of chosing 1 read from transcripts of tumor cells
    q_read_a = numpy.array(numpy.ones(tr_num)/tr_num, dtype=float32) #initial EM weight for each transcript of read_i from pure normal cells
    q_read_xa = numpy.array(numpy.ones(tr_num)/(2*tr_num), dtype=float32) #initial EM weight for each transcript of read_i from normal cells within mixture
    q_read_xb = numpy.array(numpy.ones(tr_num)/(2*tr_num), dtype=float32) #initial EM weight for each transcript of read_i from tumor cells within mixture
    tau_a = 0.5 #numpy.random.rand() #proportion of normal cells within mixture
    tau_b = 1 - tau_a #proportion of tumor cells within mixture
    i_a = 0
    i_x = 0

    print 'processing reads file...'
    for j in range(0, add_rounds+1):
        read_num_a = 0
        read_num_x = 0
        file_state_a = 0
        file_state_x = 0
        read_a_ID_current = ''
        read_x_ID_current = ''
        inreads_p.seek(0)
        inreads_m.seek(0)

        current_time = time.time()
        while file_state_a == 0 or file_state_x == 0: #at least one of the reads set do not end
            if file_state_a == 0: # inreads_p does not end
                #Online E-step
                Y_a_i, read_a_ID_current, left_read_a_i, frag_end_read_a_i, file_state_a = pasRead(inreads_p, read_a_ID_current, tr_ID, tr_num)
                
                if bias_flag == True:
                    w_a_i = getReadWeight(left_read_a_i, frag_end_read_a_i, read_len, tr_ID, tr_num, intranscripts, bias_seq_a, bias_pos_a)
                    likelihood_read_a_i = Y_a_i*alpha_a*w_a_i/(tr_len_eff*tr_avg_bias_a)
                elif bias_flag == False:
                    likelihood_read_a_i = Y_a_i*alpha_a/tr_len_eff
                
                q_read_a_i = likelihood_read_a_i/numpy.sum(likelihood_read_a_i)
                q_read_a = (1-1/pow(i_a+2, step_size))*q_read_a + (1/pow(i_a+2, step_size))*q_read_a_i
                
                #Online M-step
                alpha_a = (q_read_a + q_read_xa)/(1 + tau_a)
                
                i_a = i_a + 1
                read_num_a = read_num_a + 1
                
                if read_num_a%10000 == 0:
                    run_time = time.time() - current_time
                    current_time = time.time()
                    print 'Round %s\t%s reads in %s processed...' % (j+1, read_num_a, filename_p)
                    #print 'tau_a: %s\ttau_b: %s' % (tau_a, tau_b)
                    #print 'q_reads_xa_sum: %s\tq_reads_xb_sum: %s' %(numpy.sum(q_read_xa), numpy.sum(q_read_xb))
                
            if file_state_x == 0: # inreads_m does not end
                #Online E-step
                Y_x_i, read_x_ID_current, left_read_x_i, frag_end_read_x_i, file_state_x = pasRead(inreads_m, read_x_ID_current, tr_ID, tr_num)
                
                if bias_flag == True:
                    w_x_i = getReadWeight(left_read_x_i, frag_end_read_x_i, read_len, tr_ID, tr_num, intranscripts, bias_seq_x, bias_pos_x)
                    likelihood_read_xa_i = Y_x_i*alpha_a*w_x_i/(tr_len_eff*tr_avg_bias_x)
                    likelihood_read_xb_i = Y_x_i*alpha_b*w_x_i/(tr_len_eff*tr_avg_bias_x)
                elif bias_flag == False:
                    likelihood_read_xa_i = Y_x_i*alpha_a/tr_len_eff
                    likelihood_read_xb_i = Y_x_i*alpha_b/tr_len_eff
                    
                q_read_xa_i = likelihood_read_xa_i*tau_a/numpy.sum(likelihood_read_xa_i*tau_a + likelihood_read_xb_i*tau_b)
                q_read_xb_i = likelihood_read_xb_i*tau_b/numpy.sum(likelihood_read_xa_i*tau_a + likelihood_read_xb_i*tau_b)
                q_read_xa = (1-1/pow(i_x+2, step_size))*q_read_xa + (1/pow(i_x+2, step_size))*q_read_xa_i
                q_read_xb = (1-1/pow(i_x+2, step_size))*q_read_xb + (1/pow(i_x+2, step_size))*q_read_xb_i                

                #Online M-step
                tau_a = (numpy.sum(q_read_xa) + wa*prior_intens)/(1 + prior_intens)
                tau_b = (numpy.sum(q_read_xb) + wb*prior_intens)/(1 + prior_intens)
                alpha_a = (q_read_a + q_read_xa)/(1 + tau_a)
                alpha_b = q_read_xb/tau_b
            
                i_x = i_x + 1
                read_num_x = read_num_x + 1
                if read_num_x%10000 == 0:
                    run_time = time.time() - current_time
                    current_time = time.time()
                    print 'Round %s\t%s reads in %s processed...' % (j+1, read_num_x, filename_m)
                    #print 'tau_a: %s\ttau_b: %s' % (tau_a, tau_b)
                    #print 'q_reads_xa_sum: %s\tq_reads_xb_sum: %s' %(numpy.sum(q_read_xa), numpy.sum(q_read_xb))
            
############finalizing  Round####################

    read_num_a = 0
    read_num_x = 0
    file_state_a = 0
    file_state_x = 0
    read_a_ID_current = ''
    read_x_ID_current = ''
    inreads_p.seek(0)
    inreads_m.seek(0)
    est_counts_a = numpy.array(numpy.zeros(tr_num), dtype=float32)
    est_counts_xb = numpy.array(numpy.zeros(tr_num), dtype=float32)

    current_time = time.time()
    while file_state_a == 0:
        Y_a_i, read_a_ID_current, left_read_a_i, frag_end_read_a_i, file_state_a = pasRead(inreads_p, read_a_ID_current, tr_ID, tr_num)
        if bias_flag == True:
            w_a_i = getReadWeight(left_read_a_i, frag_end_read_a_i, read_len, tr_ID, tr_num, intranscripts, bias_seq_a, bias_pos_a)
            likelihood_read_a_i = Y_a_i*alpha_a*w_a_i/(tr_len_eff*tr_avg_bias_a)
        elif bias_flag == False:
            likelihood_read_a_i = Y_a_i*alpha_a/tr_len_eff
        q_read_a_i = likelihood_read_a_i/numpy.sum(likelihood_read_a_i)
        est_counts_a = est_counts_a + q_read_a_i #estimated counts for each transcripts of pure normal cells based on reads_a
        read_num_a = read_num_a + 1
        if read_num_a%10000 == 0:
            run_time = time.time() - current_time
            current_time = time.time()
            print 'Finalizing %s\t%s reads processed...' % (filename_p, read_num_a)
        
    while file_state_x == 0:
        Y_x_i, read_x_ID_current, left_read_x_i, frag_end_read_x_i, file_state_x = pasRead(inreads_m, read_x_ID_current, tr_ID, tr_num)
        if bias_flag == True:
            w_x_i = getReadWeight(left_read_x_i, frag_end_read_x_i, read_len, tr_ID, tr_num, intranscripts, bias_seq_x, bias_pos_x)
            likelihood_read_xa_i = Y_x_i*alpha_a*w_x_i/(tr_len_eff*tr_avg_bias_x)
            likelihood_read_xb_i = Y_x_i*alpha_b*w_x_i/(tr_len_eff*tr_avg_bias_x)
        elif bias_flag == False:
            likelihood_read_xa_i = Y_x_i*alpha_a/tr_len_eff
            likelihood_read_xb_i = Y_x_i*alpha_b/tr_len_eff
        q_read_xb_i = likelihood_read_xb_i*tau_b/numpy.sum(likelihood_read_xa_i*tau_a + likelihood_read_xb_i*tau_b)
        est_counts_xb = est_counts_xb + q_read_xb_i #esimated counts for each transcripts of tumor cells within the mixture based on reads_x
        read_num_x = read_num_x + 1
        if read_num_x%10000 == 0:
            run_time = time.time() - current_time
            current_time = time.time()
            print 'Finalizing %s\t%s reads processed...' % (filename_m, read_num_x)
        
    read_num_xb = numpy.sum(est_counts_xb)
    
###############Finishing##################

    time_end = time.time()
    print 'Time used: %ssec' % (time_end-time_start)

    outreads_a.write('transcript_ID\testimated_counts\tRPKM\n')   
    for i in range(0, tr_num):
        outreads_a.write('%s\t%s\t%s\n' % (tr_ID[i], est_counts_a[i], est_counts_a[i]*(10**9)/(read_num_a*tr_len[i])))
    
    outreads_b.write('transcript_ID\testimated_counts\tRPKM\n')  
    for i in range(0, tr_num):
        outreads_b.write('%s\t%s\t%s\n' % (tr_ID[i], est_counts_xb[i], est_counts_xb[i]*(10**9)/(read_num_xb*tr_len[i])))
    



    inreads_p.close()
    inreads_m.close()
    intranscripts.close()
    outreads_a.close()
    outreads_b.close()

if __name__ == "__main__":
    main()

