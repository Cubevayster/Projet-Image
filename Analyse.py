import time
import resource
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def measure(f, *args):
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    result = f(*args)
    
    end_time = time.time()
    end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem
    
    
    return result, end_time - start_time, end_mem / 1024, peak_mem / 1024

####def f(x,y):
        #####
#result, exec_time, mem_used, peak_mem = measure(f, 3, 5)
#print("Résultat : ", result)
#print("Temps d'exécution : ", exec_time, "secondes")
#print("Mémoire utilisée : ", mem_used, "Mo")
#print("Peak de mémoire utilisée : ", peak_mem, "Mo")

def write_result(filename, effacement, threshold, vp, vn, fn, fp, exec_time, mem_used, peak_mem):
    if effacement and os.path.isfile(filename):
        os.remove(filename)
    
    result_dict = {
        "Threshold": threshold,
        "VP": vp,
        "VN": vn,
        "FN": fn,
        "FP": fp,
        "Tp": exec_time,
        "Mem": mem_used,
        "Pk": peak_mem
    }

    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            json.dump([result_dict], f)
        return

    with open(filename, 'r') as f:
        data = json.load(f)
        
    index = len(data)
    for i in range(len(data)):
        if data[i]['Threshold'] > threshold:
            index = i
            break
    
    data.insert(index, result_dict)
    
    with open(filename, 'w') as f:
        json.dump(data, f)

def plot_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    thresholds = []
    precisions = []
    recalls = []
    accuracies = []
    f1_scores = []
    times = []
    memories = []
    peeks = []

    for row in data:
        thresholds.append(row['Threshold'])
        vp = row['VP']
        vn = row['VN']
        fp = row['FP']
        fn = row['FN']
        #print(vp,vn,fp,fn)
        precision = vp / (vp + fp) if vp + fp != 0 else 0
        rappel = vp / (vp + fn) if vp + fn != 0 else 0
        accuracy = (vp + vn) / (vp + vn + fp + fn) if vp + vn + fp + fn != 0 else 0
        f1_score = 2 * precision * rappel / (precision + rappel) if precision + rappel != 0 else 0
        #print(precision,rappel,accuracy,f1_score)
        precisions.append(precision)
        recalls.append(rappel)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        times.append(row['Tp'])
        memories.append(row['Mem'])
        peeks.append(row['Pk'])

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs[0, 0].plot(thresholds, precisions)
    axs[0, 0].set_xlabel('Threshold')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 1].plot(thresholds, recalls)
    axs[0, 1].set_xlabel('Threshold')
    axs[0, 1].set_ylabel('Recall')
    axs[1, 0].plot(thresholds, accuracies)
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 1].plot(thresholds, f1_scores)
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('F1 Score')
    axs[2, 0].plot(thresholds, times)
    axs[2, 0].set_xlabel('Threshold')
    axs[2, 0].set_ylabel('Time (s)')
    axs[2, 1].plot(thresholds, memories)
    axs[2, 1].set_xlabel('Threshold')
    axs[2, 1].set_ylabel('Memory (Mo)')
    plt.show()

def plot_complexite(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    thresholds = []
    temps = []
    memoires = []
    peaks = []
    for d in data:
        thresholds.append(d['Threshold'])
        temps.append(d['Tp'])
        memoires.append(d['Mem'])
        peaks.append(d['Pk'])

    # Tracer le graphique
    fig, axs = plt.subplots(3, figsize=(10, 10))
    fig.suptitle('Complexité en fonction du threshold')
    axs[0].plot(thresholds, temps)
    axs[0].set_ylabel('Temps (s)')
    axs[1].plot(thresholds, memoires)
    axs[1].set_ylabel('Mémoire utilisée (Mo)')
    axs[2].plot(thresholds, peaks)
    axs[2].set_ylabel('Peek (Mo)')
    axs[2].set_xlabel('Threshold')

    plt.show()


def plot_resultats(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    thresholds = []
    precisions = []
    rappels = []
    accuracies = []
    f1_scores = []
    for d in data:
        thresholds.append(d['Threshold'])
        vp = d['VP']
        vn = d['VN']
        fp = d['FP']
        fn = d['FN']
        #print(vp,vn,fp,fn)
        precision = vp / (vp + fp) if vp + fp != 0 else 0
        rappel = vp / (vp + fn) if vp + fn != 0 else 0
        accuracy = (vp + vn) / (vp + vn + fp + fn) if vp + vn + fp + fn != 0 else 0
        f1_score = 2 * precision * rappel / (precision + rappel) if precision + rappel != 0 else 0
        #print(precision,rappel,accuracy,f1_score)
        precisions.append(precision)
        rappels.append(rappel)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)

    # Tracer le graphique
    fig, axs = plt.subplots(4, figsize=(10, 10))
    fig.suptitle('Performances en fonction du threshold')
    axs[0].plot(thresholds, precisions)
    axs[0].set_ylabel('Précision')
    axs[1].plot(thresholds, rappels)
    axs[1].set_ylabel('Rappel')
    axs[2].plot(thresholds, accuracies)
    axs[2].set_ylabel('Accuracy')
    axs[3].plot(thresholds, f1_scores)
    axs[3].set_ylabel('F1 score')
    axs[3].set_xlabel('Threshold')

    plt.show()