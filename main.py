"""
Methods and workflow for analyzing English-language scientific corpus concerning
Polistes wasps.

These methods are built with the iPython Notebook environment in mind. To use
externally, simply change all plt.show() calls to plt.savefig(), with the
appropriate arguments.

WARNING: Package :mod:`tethne` has undergone major revisions since this code was
written; should use tethne-0.4.2-alpha at the latest.
"""

import tethne.readers as rd
import tethne.writers as wr
import tethne.analyze as az
from tethne.data import DataCollection, GraphCollection
import networkx as nx
import os
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.cm as cm

from nltk.corpus import stopwords
sw = stopwords.words()

c = cm.get_cmap(name='Set1')



# Load JSTOR DfR datasets
unigrams = {}
bigrams = {}
trigrams = {}
datapath = "/Users/erickpeirson/Dropbox/Digital Hps/Files/data"
for directory in os.listdir(datapath):
    if directory == '.DS_Store':
        continue
    unigrams.update(rd.dfr.ngrams(datapath + "/" + directory, 'uni'))
    bigrams.update(rd.dfr.ngrams(datapath + "/" + directory, 'bi'))
    trigrams.update(rd.dfr.ngrams(datapath + "/" + directory, 'tri'))    

papers = rd.dfr.from_dir(datapath)    # Metadata for papers in this dataset.

# This makes it faster to search for tokens later on.
types = {1: {d:{k:v for k,v in vals} for d,vals in unigrams.iteritems()},
         2: {d:{k:v for k,v in vals} for d,vals in bigrams.iteritems()},
         3: {d:{k:v for k,v in vals} for d,vals in trigrams.iteritems()}}

# Define terms of interest.
explananda = [  #('polygyny',), 
                ('division of labor',), 
                ('hierarchy','hierarchies','hierarchical'),                               
                #('social dominance',),
                ('dominance',)]

explanantes = [ ('ovaries', 'ovary'), 
                ('hormones', 'hormone', 'hormonal'), 
                ('endocrine',), 
                ('ontogeny', 'ontogenesis', 'ontogenetic', 'developmental' ),
                ('relatedness',),                 
                ('fitness',), 
                ('cost','costs',), 
                ('benefit', 'benefits')]
#                ('embryonic development',)]

# For convenience, index terms by their tokens.
e_index = {}
for ex in explananda:
    e_index[ex[0]] = 'explananda'    
for es in explanantes:
    e_index[es[0]] = 'explanantes'

# Some globals.
N_s = len(explanantes)  # Number of explanantes.
N_m = len(explananda)   # Number of explananda.
N_d = len(D.papers())   # Number of documents in corpus.
T = range(1872, 2011)   # Can narrow this down later.

# Ordered vector of document counts over time for the corpus as a whole.
N = [ len(D.axes['date'][d]) for d in sorted(T) ]

# So that we use indices consistently across all matrices...
p_lookup = {}   # Paper (str[doi]) : index (int) hash.
t_lookup = {}   # Term (str) : index (int) hash.

em_indices = [ t_lookup[em[0]] for em in explananda ]

# Organize dataset by time (4-year sliding time-window)
D = DataCollection(papers, index_by='doi')
D.slice('date', method='time_window', window_size=1, step_size=1)

def build_occurrence_matrix(D, types):
    """
    Generates an occurrence matrix for all (em + es) terms across the corpus.
    
    Parameters
    ----------
    D : :class:`tethne.classes.DataCollection`
        Contains :class:`tethne.classes.Paper` for each N-gram vector.
    types : dict
        Contains N-indexed dictionaries of { doi : { gram : freq } }.
    
    
    Returns
    -------
    a : :class:`numpy.array`
        Rows are documents, columns are terms.
    """
    
    a = np.zeros(( N_d, N_s + N_m ))

    p_i = 0
    for p in D.papers():
        doi = p['doi']
        
        try:    # Not all Papers have N-gram vectors. Avoid those.
            types[1][doi]
        except KeyError:
            continue
            
        p_lookup[doi] = p_i
        p_lookup[p_i] = doi
        
        t_i = 0
        for s in xrange(N_s):   # Explanantes.
            es = explanantes[s]
            t_lookup[t_i] = es[0]
            t_lookup[es[0]] = t_i

            found = False   # To avoid redundant increments for multi-token
            for s_ in es:   #   hits.
                s_type = len(s_.split())
                if s_ in types[s_type][doi]:
                    found = True
            if found:
                a[p_i, t_i] += 1.

            t_i += 1
                
        for m in xrange(N_m):   # Explananda.
            em = explananda[m]
            t_lookup[t_i] = em[0]
            t_lookup[em[0]] = t_i
            
            found = False   # See above.
            for m_ in em:
                m_type = len(m_.split())
                if m_ in types[m_type][doi]:
                    found = True

            if found:
                a[p_i, t_i] += 1.   # Increment occurrence.

            t_i += 1
        p_i += 1

    return a

def build_cooccurrence_matrix(D, a):
    """
    Geneartes a time-variant co-occurrence matrix for all terms.
    
    Parameters
    ----------
    D : :class:`tethne.classes.DataCollection`
        Contains :class:`tethne.classes.Paper` for each N-gram vector.    
    a : :class:`numpy.array`
        Rows are documents, columns are terms.
    
    Returns
    -------
    co_f : numpy.array
        Time-variant term co-occurrence matrix. Shape: ( N_s+N_m, N_s+N_m, T ).
    co_f_N : numpy.array
        Normalization of `co_f` over the number of documents in the corpus.
        Shape: ( N_s+N_m, N_s+N_m, T ).
    """

    co_f = np.zeros(( N_s + N_m, N_s + N_m, len(T) ))

    for t in sorted(T): # One time-period at a time.
        dois = D.axes['date'][t]
        time_index = t - 1872   # Numpy arrays are 0-indexed.
        
        for doi in dois:
            try:    # Not all Papers have N-gram vectors. Avoid those.
                types[1][doi]
            except KeyError:
                continue        
                
            p_i = p_lookup[doi] # Get Paper's index.
            
            for s in xrange(N_s):               # Explanantes.
                t_i = t_lookup[explanantes[s][0]]

                for t in xrange(N_s):           # ...against explanantes.
                    y_i = t_lookup[explanantes[t][0]]
                    
                    if a[p_i, t_i] == 1. and a[p_i, y_i] == 1.:
                        co_f[t_i, y_i, time_index] += 1.
                    
                for t in xrange(N_m):           # ... against explananda.
                    y_i = t_lookup[explananda[t][0]]
                    
                    if a[p_i, t_i] == 1. and a[p_i, y_i] == 1.:
                        co_f[t_i, y_i, time_index] += 1.  
                        co_f[y_i, t_i, time_index] += 1.                      

            for s in xrange(N_m):               # Explananda.
                t_i = t_lookup[explananda[s][0]]   
                
                for t in xrange(N_m):           # ...against explananda.
                    y_i = t_lookup[explananda[t][0]]            
                    
                    if a[p_i, t_i] == 1. and a[p_i, y_i] == 1.:
                        co_f[t_i, y_i, time_index] += 1.    
                        
                for t in xrange(N_s):           # ...against explanantes.
                    y_i = t_lookup[explanantes[t][0]]
                    
                    if a[p_i, t_i] == 1. and a[p_i, y_i] == 1.:
                        co_f[t_i, y_i, time_index] += 1.                    

    # Normalize.
    co_f_N = np.zeros( ( N_m + N_s, N_m + N_s, len(T) ))
    for time in sorted(T):
        t = time - min(T)
        co_f_N[:,:,t] = co_f[:,:,t] / N_[t]

    return co_f, co_f_N

def calculate_f_EM(D, a):
    """
    Generates a vector over time for the number of documents containing one
    or more explanandum terms.
    
    Parameters
    ----------
    D : :class:`tethne.classes.DataCollection`
        Contains :class:`tethne.classes.Paper` for each N-gram vector.    
    a : :class:`numpy.array`
        Rows are documents, columns are terms.
        
    Returns
    -------
    f_EM : numpy.array
        A vector over time for the number of documents containing one or more 
        explanandum terms.
    """
    f_EM = np.zeros( ( len(T) ))
    for time in sorted(T):
        t = time - min(T)
        
        for doi in D.axes['date'][time]:
            try:
                p_i = p_lookup[doi]
            except KeyError:
                continue
                
            found = False
            for i in em_indices:
                if a[p_i, i] > 0.:
                    found = True
            if found:
                f_EM[t] += 1.
                
    return f_EM

def plot_f_EM(f_EM)
    """
    Generates a bar-chart showing the number of documents in each time-slice,
    and the number of EM-responding documents in each time-slice.
    """
    fig, ax1 = subplots(figsize=(20,14))
    ax1.bar(np.array(sorted(T))-0.5, N, alpha=0.2, width=0.5, color='g', 
                                        label='Papers in corpus')
    ax1.set_ylabel(r'$N$', rotation='horizontal', fontsize=24)
    ax1.tick_params(labelsize=16)
    ax1.bar(np.array(sorted(T)), f_EM, lw=3, 
                                       label='Papers containing explananda', 
                                       alpha=0.6)

    plt.legend(loc=2, fontsize=16)
    plt.xlim(1900, 2011)
    plt.show() 

# Distribution over journals
def plot_journal_distribution(D):
    """
    Plots the distribution of papers over journals for the top 20 journals.
    
    Parameters
    ----------
    D : :class:`tethne.classes.DataCollection`
        Contains :class:`tethne.classes.Paper` for each N-gram vector. 
    """
    journals = Counter()    # Number of papers in each journal.
    for p in D.papers():
        journals[p['jtitle']] += 1
    
    # Order by number of papers in each journal.
    jsorted = sorted(journals, key=journals.get, reverse=False)
    
    plt.figure(figsize=(7,10))
    plt.barh(arange(20), [ journals[j] for j in jsorted[-20:] ], color='green', 
                                                                 alpha=0.5, 
                                                                 lw=2)
    plt.yticks(arange(20)+0.3, [ '  '+j for j in jsorted[-20:]])
    plt.show()

def build_EM_subcorpus(D,a):
    """
    Generates a list of Paper identifiers (dois) for those containing one or
    more explanandum terms.
    
    Parameters
    ----------
    D : :class:`tethne.classes.DataCollection`
        Contains :class:`tethne.classes.Paper` for each N-gram vector.    
    a : :class:`numpy.array`
        Rows are documents, columns are terms.
        
    Returns
    -------
    C_em : list
        A list of Paper identifiers (dois) for those containing one or
        more explanandum terms.
    """

    C_em = []
    for p in D.papers():
        try:    # Not all papers in D have N-grams. Avoid those.
            a[ p_lookup[p['doi']], :]
        except KeyError:
            continue

        # If there is at least one explanandum hit in a paper, include it.
        if np.sum(a[ p_lookup[p['doi']], min(em_indices):max(em_indices) ]) > 0: 
            C_em.append(p_lookup[p['doi']])

    return C_em

def build_f_es_EM(D, a):
    """
    Generates per-document frequency vectors for explanantes in the explanandum
    subcorpus, `C_em`\, over time.
    
    Parameters
    ----------
    D : :class:`tethne.classes.DataCollection`
        Contains :class:`tethne.classes.Paper` for each N-gram vector.    
    a : :class:`numpy.array`
        Rows are documents, columns are terms.
        
    Returns
    -------
    f_es_EM : numpy.array
        Per-document frequency of each explanans term in the explanandum
        sub-corpus, over time. Shape: ( T, N_s ).
    N_em : numpy.array
        Number of explandum-containing papers over time. Shape: ( T )
    f_es_EM_ : numpy.array
        `f_es_EM` normalized on N_em. Shape: ( T, N_s ).
    """

    f_es_EM = np.zeros( ( len(T), N_s ) )
    N_em = np.zeros( (len(T)))
    f_es_EM = np.zeros( ( len(T), N_s ) )
    
    for t in sorted(T):
        dois = D.axes['date'][t]
        time_index = t - 1872
            
        for d in dois:
            try: p_lookup[d]    # Avoid papers with no data.
            except KeyError: continue        

            if d in C_em:   # Explanandum sub-corpus only.
                N_em[time_index] += 1.
                
                for s in xrange(N_s):
                    t_i = t_lookup[explanantes[s][0]] # Adds a[] just in case we
                                                      #  want to change how we
                                                      #  calculate frequencies.
                    f_es_EM[ time_index, t_i ] += a[ p_lookup[d], t_i ]

        # Normalize.
        for i in xrange(N_s):
            t_i = t_lookup[ explanantes[i][0] ]
            normed = f_es_EM [ time_index, t_i ]/N_em[time_index]
            f_es_EM_[ time_index, t_i ] = normed

    return f_es_EM, N_em, F_es_EM_
    
def plot_p_es_EM(f_es_EM_):
    """
    Plot the observed probability of each explanans term in the explanandum
    sub-corpus.
    
    Parameters
    ----------
    f_es_EM : numpy.array
        Normalized per-document frequency of es in the EM subcorpus.
    """

    fig = plt.figure(figsize=(5,20))

    for i in xrange(N_s):
        s = explanantes[i][0]
        v = f_es_EM_[:, i]
        ax = plt.subplot(N_s, 1, i+1)
        ax.plot(T, v, lw=3)
        ax.set_ylabel(s, fontsize=16)
        ax.set_ylim(0.,1.1)
        ax.set_xlim(1900, 2011)
        
        if i == 0:  # Header.
            plt.title(r'$p(es| EM) = \frac{f_{es}}{N_{EM}}$', fontsize=24, 
                                                              va='bottom')
    plt.show()

def plot_p_em(co_f):
    """
    Plot the observed probability of each explanandum term in the master corpus.
    
    Parameters
    ----------
    co_f : numpy.array
        A time-variant term co-occurrence matrix, shape ( N_s+N_m, N_s+N_m, T ).    
    """
    fig = plt.figure(figsize=(18,2.25))
    N_ = np.array([ len(D.axes['date'][v]) for v in T ])

    for i in xrange(N_m):
        em = explananda[i][0]
        ax = plt.subplot(1, N_m, i+1)
        values = np.array([ co_f[ t_lookup[em], t_lookup[em], t-min(T) ] 
                                                            for t in T ] ) / N_
        ax.plot(T, values, label=em, lw=3)#, color=c(t))
        ax.set_xlim(1900, 2011)
        ax.set_ylim(0., 0.6)
        ax.set_title(em, fontsize=16)
        if i == 0:
            ax.set_ylabel(r'$\frac{f_{em}}{N}$', rotation='horizontal',
                                                 fontsize=24)
    plt.show()

def calc_nPMI(t_i, y_i, t):
    """
    Calculates the normalized Pointwise Mutual Information for two terms
    (given by their indices, t_i and t_y) at a given time, t.
    
    Parameters
    ----------
    t_i : int
        Index of term 1.
    y_i : int
        Index of term 2.
    t : int
        Time index.
        
    Returns
    -------
    float
        Normalized Pointwise Mutual Information.
    """
    p_t = co_f_N[t_i, t_i, t]
    p_y = co_f_N[y_i, y_i, t]
    p_ty = co_f_N[t_i, y_i, t]
    
    if p_ty == 0.:
        return 0.
    
    return ( np.log( p_ty/(p_t*p_y) ) ) / ( -1* np.log(p_ty) )


def calc_osPMI(t_i, y_i, t):
    """
    Calculates the one-side PMI, which is just the MLE of p(y|t).
    
    Parameters
    ----------
    t_i : int
        Index of term 1.
    y_i : int
        Index of term 2.
    t : int
        Time index.
        
    Returns
    -------
    float
        One-sided Pointwise Mutual Information.    
    """
    emN = 0.
    esN = 0.
    for doi in D.axes['date'][t]:
        try:
            a[p_lookup[doi], y_i]
        except KeyError:
            continue
        if a[p_lookup[doi], y_i] > 0.:
            emN += 1.
            if a[p_lookup[doi], t_i] > 0.:
                esN += 1.
    if emN == 0.:
        return 0.
    return esN/emN

def calculate_PMI():
    """
    Generates nPMI and osPMI values for all term-pairs over time.
    
    Returns
    -------
    nPMI : numpy.array
        Normalized Pointwise Mutual Information for all term-pairs over time.
        Shape: ( N_m + N_s, N_m + N_s, T ).
    osPMI : numpy.array
        One-sided Pointwise Mutual Information for all term-pairs over time.        
        Shape: ( N_m + N_s, N_m + N_s, T ).
    """

    nPMI = np.zeros( ( N_m + N_s, N_m + N_s, len(T) ))
    osPMI = np.zeros( ( N_m + N_s, N_m + N_s, len(T) ))

    for time in sorted(T):
        t = time - min(T)
        for s in xrange(N_s):   # Explanantes.
            t_i = t_lookup[ explanantes[s][0] ]
            for r in xrange(N_s):   # ...against explanantes.
                y_i = t_lookup [ explanantes[r][0] ]
                nPMI[ t_i, y_i, t] = calc_nPMI( t_i, y_i, t )
                osPMI[ t_i, y_i, t] = calc_osPMI( t_i, y_i, time )
            for r in xrange(N_m):   # ...against explananda.
                y_i = t_lookup [ explananda[r][0] ]            
                nPMI[ t_i, y_i, t] = calc_nPMI( t_i, y_i, t )     
                osPMI[ t_i, y_i, t] = calc_osPMI( t_i, y_i, time )            
        for s in xrange(N_m):   # Explananda...
            t_i = t_lookup[ explananda[s][0] ]        
            for r in xrange(N_s):   # ...against explanantes.
                y_i = t_lookup [ explanantes[r][0] ]            
                nPMI[ t_i, y_i, t] = calc_nPMI( t_i, y_i, t )
                osPMI[ t_i, y_i, t] = calc_osPMI( t_i, y_i, time )            
            for r in xrange(N_m):   # ...against explananda.
                y_i = t_lookup [ explananda[r][0] ]            
                nPMI[ t_i, y_i, t] = calc_nPMI( t_i, y_i, t )
                osPMI[ t_i, y_i, t] = calc_osPMI( t_i, y_i, time )   

    return nPMI, osPMI

def plot_PMI(nPMI, osPMI):
    """
    Plots nPMI and osPMI for all term-pairs over time.
    
    Parameters
    ----------
    nPMI : numpy.array
        Normalized Pointwise Mutual Information for all term-pairs over time.
        Shape: ( N_m + N_s, N_m + N_s, T ).
    osPMI : numpy.array
        One-sided Pointwise Mutual Information for all term-pairs over time.        
        Shape: ( N_m + N_s, N_m + N_s, T ).    
    """
    fig = plt.figure(figsize=(15,20))
    
        osPMI_label = r'$p(es| em) = \frac{f_{es}}{N_{em}}$'
    
    for i_s in xrange(N_s):
        for i_m in xrange(N_m):
            s = t_lookup[ explanantes[i_s][0] ]
            m = t_lookup[ explananda[i_m][0] ]
            
            sp = i_m + 1 + i_s*N_m
            ax = plt.subplot(N_s, N_m, sp)
            ax.plot(T, nPMI[s, m, :], 'g--', label=r'$nPMI(es,em)$', alpha=0.8)  
            ax.plot(T, osPMI[m, s, :], label=osPMI_label, lw=3, alpha=0.6)
            ax.set_ylim(0., 1.1)
            ax.set_xlim(1900, 2011)

            if i_m == 0:
                ax.set_ylabel(explanantes[i_s][0], fontsize=16)
                
            if i_s == 0:
                ax.set_title(explananda[i_m][0], fontsize=16)
                
            if i_s == 0 and i_m == N_m - 1:
                ax.legend(bbox_to_anchor=(1.6,1))
    plt.show()

def plot_PMI_es(nPMI, osPMI):
    """
    Similar to :func:`.plot_PMI` except that only the top half of the matrix
    is plotted, and only explanantes terms are included.
    
   Parameters
    ----------
    nPMI : numpy.array
        Normalized Pointwise Mutual Information for all term-pairs over time.
        Shape: ( N_m + N_s, N_m + N_s, T ).
    osPMI : numpy.array
        One-sided Pointwise Mutual Information for all term-pairs over time.        
        Shape: ( N_m + N_s, N_m + N_s, T ).     
    """
    fig = plt.figure(figsize=(30,20))
    for i_s in xrange(N_s):
        for i_m in xrange(i_s + 1, N_s):
            if i_s == i_m:
                continue
            s = t_lookup[ explanantes[i_s][0] ]
            m = t_lookup[ explanantes[i_m][0] ]
            
            sp = i_m + 1 + i_s*N_s
            ax = plt.subplot(N_s, N_s, sp)
            ax.plot(T, osPMI[m, s, :], lw=3, alpha=0.6, c='b',
                                       label=r'$p(row|col)$')
            ax.plot(T, osPMI[s, m, :], lw=3, alpha=0.6, c='m',
                                       label=r'$p(col|row)$')
            ax.plot(T, nPMI[s, m, :], 'g--', label=r'$nPMI(es,em)$', lw=2)
            
            ax.set_ylim(0., 1.1)
            ax.set_xlim(1900, 2011)

            #if i_m == 0 or (i_s == 0 and i_m == 1):
            if i_m == i_s + 1:
                ax.set_ylabel(explanantes[i_s][0], fontsize=16)

            #if i_s == 0 or (i_m == 0 and i_s == 1):
            if i_s == 0:
                ax.set_title(explanantes[i_m][0], fontsize=16)
                
            if i_s == 1 and i_m == 2:
                ax.legend(bbox_to_anchor=(-0.4,0.9))
    plt.show()

def plot_PMI_em(nPMI, osPMI):
    """
    Similar to :func:`.plot_PMI` except that only the top half of the matrix
    is plotted, and only explananda terms are included.
    
   Parameters
    ----------
    nPMI : numpy.array
        Normalized Pointwise Mutual Information for all term-pairs over time.
        Shape: ( N_m + N_s, N_m + N_s, T ).
    osPMI : numpy.array
        One-sided Pointwise Mutual Information for all term-pairs over time.        
        Shape: ( N_m + N_s, N_m + N_s, T ).     
    """

    fig = plt.figure(figsize=(11.25,7.5))
    for i_s in xrange(N_m):
        for i_m in xrange(i_s + 1, N_m):
            s = t_lookup[ explananda[i_s][0] ]
            m = t_lookup[ explananda[i_m][0] ]
            
            sp = i_m + 1 + i_s*N_m
            ax = plt.subplot(N_m, N_m, sp)
            ax.plot(T, osPMI[m, s, :], lw=3, alpha=0.6, c='b', 
                                       label=r'$p(row|col)$')
            ax.plot(T, osPMI[s, m, :], lw=3, alpha=0.6, c='m',
                                       label=r'$p(col|row)$')    
            ax.plot(T, nPMI[s, m, :], 'g--', label=r'$nPMI(es,em)$', lw=2)
            ax.set_ylim(0., 1.1)
            ax.set_xlim(1900, 2011)

            if i_m == i_s + 1:
                ax.set_ylabel(explananda[i_s][0], fontsize=16)
                
            if i_s == 0:
                ax.set_title(explananda[i_m][0], fontsize=16)
                
            if i_s == 1 and i_m == 2:
                ax.legend(bbox_to_anchor=(-0.4,0.9))
    plt.show()


def ols_em(co_f_N):
    """
    Generates a correlation matrix for explananda terms. (1960 onward)
    
    Parameters
    ----------
    co_f_N : numpy.array
        Normalization of time-variant term co-occurrence matrix over the number
        of documents in the corpus. Shape: ( N_s+N_m, N_s+N_m, T ). Uses the
        diagonal to get normalized frequency values for each term over time.
    """
    # Colormap over time.
    c = plt.get_cmap('spectral', lut=len(t))

    q = 25  # First break-point.
    z = 40  # Second break-point.

    # <codecell>

    plt.figure(figsize=(22.5,15))
    for i_s in xrange(N_m):
        for i_m in xrange(i_s + 1, N_m):
            s = t_lookup[ explananda[i_s][0] ]
            m = t_lookup[ explananda[i_m][0] ]
            
            sp = i_m + 1 + i_s*N_m
            ax = plt.subplot(N_m, N_m, sp)  
            x = co_f_N[ s, s, T.index(1960): ]
            y = co_f_N[ m, m, T.index(1960): ]

            # Fit OLS linear regression to first break-point.
            regress = stats.linregress(x[0:q],y[0:q])
            slope, intercept, r_value, p_value, std_err = regress        
            fit_fn = [ ( slope * xi ) + intercept for xi in x[0:q] ]   
            label = r'$r^2={0}$, $p={1}$'.format(round(r_value**2, 4)     
            
            # Fit OLS linear regression to second break-point.
            regress2 = stats.linregress(x[q:z],y[q:z])        
            slope2, intercept2, r_value2, p_value2, std_err2 = regress2
            fit_fn2 = [ ( slope2 * xi ) + intercept2 for xi in x[q:z] ]  
            label2 = r'$r^2={0}$, $p={1}$'.format(round(r_value2**2, 4)
            
            # Fit OLS linear regression after second break-point.
            regress3 = stats.linregress(x[z:],y[z:])        
            slope3, intercept3, r_value3, p_value3, std_err3 = regress3
            fit_fn3 = [ ( slope3 * xi ) + intercept3 for xi in x[z:] ]          
            label3 = r'$r^2={0}$, $p={1}$'.format(round(r_value3**2, 4)
            
            ax.scatter(x,y,c=t, cmap=c)
            plt.plot(x[0:q], fit_fn, label=label, round(p_value, 4)))
            plt.plot(x[q:z], fit_fn2, label=label2, round(p_value2, 4)))
            plt.plot(x[z:], fit_fn3, label=label3, round(p_value3, 4)))
            plt.legend(loc=2, fontsize=10)       

            if i_m == i_s + 1:
                ax.set_ylabel(explananda[i_s][0], fontsize=16)
                
            if i_s == 0:
                ax.set_title(explananda[i_m][0], fontsize=16)        


#correlations = { k:{ l:{} for l in xrange(N_s) } for k in xrange(N_s) }


def mc_test(a,b,lower,upper):
    """
    Performs a Monto Carlo simulation for cross-correlation of time-series
    `a` and `b`\, and generates upper and lower percentiles.
    
    Parameters
    ----------
    a : numpy.array
        Time-series data. Must have same shape as `b`\.
    b : numpy.array
        Time-series data. Must have same shape as `a`\.  
    lower : float
        Lower percentile bound for MC test. (e.g. 2.5)
    upper : float
        Upper percentile bound for MC test. (e.g. 97.5)
        
    Returns
    -------
    low : numpy.array
        Time-series threshold for lower percentile.
    high : numpy.array
        Time-series threshold for higher percentile.        
    """
    s_ = [ i for i in a ]
    corrs = []
    for i in xrange(1000):
        np.random.shuffle(s_)
        r = np.correlate(s_, b, 'full')
        corrs.append(r)
        
    low = np.percentile(array(corrs), lower, axis=0)
    high = np.percentile(array(corrs), upper, axis=0)
    return low,high

def perform_crosscorrelation(f_es_EM_):
    """
    Performs cross-correlation analysis.
    
    1. Calculates cross-correlation for each term-pair (above the diagonal).
    2. Performs a Monte Carlo simulation for each term-pair.
    3. Plots cross-correlation, upper and lower percentiles, and marks
       maxima/minima outside the percentile thresholds.
       
    Parameters
    ----------
    f_es_EM_ : numpy.array
        Per-document frequency of each explanans term in the explanandum
        sub-corpus, over time, normalized on the number of 
        explanandum-containing papers over time. Shape: ( T, N_s ).   
    """

    plt.figure(figsize(20,20))
    for i_s in xrange(N_s):
        for i_m in xrange(i_s + 1, N_s):
            s = t_lookup[ explanantes[i_s][0] ]
            m = t_lookup[ explanantes[i_m][0] ]
            
            sp = i_m + 1 + i_s*N_s
            ax = plt.subplot(N_s, N_s, sp) 
            
    #        x = co_f_N[ s, s, T.index(1960):2001 ]
    #        y = co_f_N[ m, m, T.index(1960):2001 ]     
            x = f_es_EM_[ T.index(1960):, s ]
            y = f_es_EM_[ T.index(1960):, m ] 
            t = T[T.index(1960):]
            
            c = np.correlate(x, y, 'full')
            low,high = mc_test(x, y, 0.5, 99.5)
            low_, high_ = mc_test(x, y, 2.5, 97.5)
            
    #        plt.plot(t,x)
    #        plt.plot(t,y)

            lagX = arange( -x.size+1, x.size )
            #plt.plot(lagX, low, label='low', alpha=0.5)
            #plt.plot(lagX, high, label='high', alpha=0.5)
            plt.fill_between(lagX, low, high, alpha=0.2, facecolor='yellow')
            plt.fill_between(lagX, low_, high_, alpha=0.3, facecolor='yellow')
            plt.plot(lagX, c, alpha=0.8, color='red')
            plt.yticks([])

     
            diff_low = c - low
            diff_high = high - c
            ax2 = ax.twinx()
            ax2.plot(lagX, diff_low, lw=2)
            ax2.plot(lagX, diff_high, lw=2)
            ax2.plot(lagX, np.array([0]*lagX.size), color='black')
            ax2.plot([0,0], [-2,4], color='black')        
            
            l = list(lagX).index(-9)
            h = list(lagX).index(9)
            if np.min(diff_low[l:h]) < 0.:
                minimum = np.min(diff_low[l:h])
                peak = lagX[l:h][list(diff_low[l:h]).index(minimum)]
                ax2.plot([peak,peak], [-2, minimum], color='red',
                                                     alpha=0.8, 
                                                     lw=3)
            if np.min(diff_high[l:h]) < 0.:
                minimum = np.min(diff_high[l:h])
                peak = lagX[l:h][list(diff_high[l:h]).index(minimum)]
                ax2.plot([peak,peak], [-2, minimum], color='red', 
                                                     alpha=0.8,
                                                     lw=3)            
            
            ax.set_ylabel(explanantes[i_m][0], fontsize=16)
                
            ax.set_xlabel(explanantes[i_s][0], fontsize=16)          
            
            plt.xlim(-9, 9)
            ax2.set_ylim(-2.0, 4.0)
            
    plt.tight_layout()
    plt.show()
