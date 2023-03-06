#---------------#

import sys
import math
import time
import psutil
#Plus simple
import tracemalloc
from time import perf_counter

class Timer:
    def __init__(self):
        self.debut = 0
        self.fin = 0
        self.timer = time.time()
        self.finish = False

    def start(self):
        self.timer = time.time()

    def end(self, form='s'):
        self.fin = time.time()
        self.finish = True
        temps = self.fin - self.debut
        temps /= 100000000000
        #print(f"Between {self.debut} and {self.fin}")
        if form == 'm':
            print(f"Temps écoulé = {temps/60} minutes")
        if form == 'h':
            print(f"Temps écoulé = {temps/3600} heures")
        if form == 'j':
            print(f"Temps écoulé = {temps/86400} jours")
        else:
            print(f"Temps écoulé = {temps} secondes")

    def profondeur(self,n,m):
        return int(math.log(n, m))

    def nb_branches(self,n,m):
        sum = 0
        for i in range(0,n*m):
            sum += i*(n*m-1)
        return sum

    #Use considering One set of possibilities
    # {root} -> {p1,p2}
    def arrangement(self, n, m):
        return math.comb(m, n)*math.factorial(n*m)

    #Use considering all set 
    # {root} -> either {p1,{p2,p3}} or {p2,{p1,p3}}
    def permutations(self, n, m):
        somme = 0
        for i in range(0,n*m):
            somme += self.arrangement(1,n*m-i+1)
        return somme

    #Calculate the number of exploration from the number of branches, the possibilities of explorations
    #set possibilities to True for considering all possibles organisation of data set
    def exploration(self,n,m,possibilities=False):
        if possibilities is True:
            return self.permutations(n,m)*self.nb_branches(n,m)
        else :
            return self.arrangement(n,m)*self.nb_branches(n,m)
    
    #Return the number of byte from an object, careful for an class without initializing it since the value of a variable will takes space more resulting in incorret stance
    def espace(self, obj):
        return sys.getsizeof(obj)

    def complexite(self, obj, n=3, m=3, form='s'):
        men = self.espace(obj) * self.permutations(n,m)
        if self.finish == True :
            t = self.fin - self.debut
            t /= 100000000000
            print(f"Temps = {t} secondes")
            return t
        else :
            self.end(form)
            print(f"Memoire Total = {men} bits")
            return mem
    
    def analyse_function(self,func):
        self.start()
        analyse = func()
        self.end()
        mem = self.espace(analyse)
        print(f"Memoire Total de la fonction = {mem} bits")
        return analyse

    def espace_used(self, obj):
        return psutil.Process().memory_info().rss

    def compare(self, obj):
        used = self.espace_used(obj)
        reste = psutil.virtual_memory().available
        bg = reste + used
        ed = reste - used
        print(f"Memoire Total utilisée {used} sur {bg} soit {ed} bits")

    def get_lambda(self,stvar,steval):
        return eval("lambda "+stvar+": " + steval)

        
    #simule un acces memoire à une classe inconnue
    #lambda qui défnie la classe et l'accès
    #1accès et on estime le temps
    def acces_1_mem(self, st1, st2) :
        c = ["banana"]
        d = ["split"]
        dummy = [self.get_lambda(st1,st2)(d,c)]
        t0 = time.time()
        used = self.espace_used(dummy)
        reste = psutil.virtual_memory().available
        t1 = time.time()
        return t1-t0

    def acces_memmoire(self, st1, st2, n , m) :
        acess = self.acces_1_mem(st1,st2)
        mem = acess
        for i in range(0,self.permutations(n,m)-1) :
            mem += acess
        mem /= 1000000
        print(f"Time for memory acces = {mem}")
        return mem

    def data_arbre(self,n,m):
        tacess = self.acces_memmoire("x,y","x.append(y)",n,m)

        class tree:
            def __init__(self):
                self.root = 0
                self.child = [0] * n*main
            def getchild(self,i) :
                return self.child[i]

        estimated_space = self.espace_used(tree)
        number = self.exploration(n,m)
        print(f"Estimated Time = {estimated_space*number*tacess} secondes\n")
        print(f"Estimates Space = {estimated_space} bytes\n")

    def time_and_memory(self, f):
        def wrapper(*args, **kwargs):
            # Mesure de la mémoire utilisée
            tracemalloc.start()
            result = f(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Affichage des résultats
            print(f"Temps d'exécution : {perf_counter()/1000000:.6f} secondes")
            print(f"Utilisation de la mémoire : {current / 10**6:.6f} Mo (pic de {peak / 10**6:.6f} Mo)")

            return result
        return wrapper


#timer = Timer()
#timer.start()
#for i in range(0, 10):
  #i = i**2
#timer.end('s')

#timer2 = Timer()
#timer2.acces_memmoire("x","x + 2",3,3)
#c = []
#timer2.acces_memmoire("x,y","x.append(y)",3,3)

#class c:
  #def __init__(self):
    #self.x = 0
    
  #def up(n):
    #for i in range(0,n) : i=i

#C = c()
#timer2.start()
#c.up(1000000)
#timer2.end()
#timer.complexité(C)

#print('------------')
#timer3 = Timer()
#timer3.data_arbre(3,3)
#timer3.analyse_function(main)
#---------------#

data = Timer()