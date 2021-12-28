import numpy as np
import subprocess
import os
import glob
datfold = './xray_imgs/xmm4_all/'
def getxmmdata(obsnumb, m1=True, m2=True, pn=True):
    '''
    Interfacing with the online server for XMM-Newton
    '''
    fold = obsnumb+'/'
    #setting up the unix commands.
    if m1:
        m1comm =  'curl -o' +datfold+'m1/'+obsnumb+'m1.tar "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno='+obsnumb+'&instname=M1&extension=FTZ&name=IMAGE_&level=PPS" '
        pm1 = subprocess.Popen(m1comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outm1, errm1 = pm1.communicate()
        print('downloading m1')
        print(outm1)
    if m2:        
        m2comm =  'curl -o' +datfold+'m2/'+obsnumb+'m2.tar "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno='+obsnumb+'&instname=M2&extension=FTZ&name=IMAGE_&level=PPS" '
        pm2 = subprocess.Popen(m2comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outm2, errm2 = pm2.communicate()
        print('downloading m2')
        print(outm2)

    if pn:
        pncomm =  'curl -o' +datfold+'pn/'+obsnumb+'pn.tar "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno='+obsnumb+'&instname=PN&extension=FTZ&name=IMAGE_&level=PPS" '
        ppn = subprocess.Popen(pncomm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outpn, errpn = ppn.communicate()
        print('downloading pn')
        print(outpn)

    #creating the command
    #sending the command for download
#
#goodobs = np.loadtxt('./catalogs/goodobstimes.txt',dtype='U32')
#goodobsalltimes is all the XMM observations between log(t_exp) 4.1, 4.5
goodobsalltimes = np.loadtxt('./catalogs/goodobsalltimes.txt', dtype='U32')
goodobsalltimesx4 = np.load('./catalogs/x4_obsids_t4.1_4.5.npy', allow_pickle=True)
alltimesx4 = np.load('./catalogs/x4_obsids.npy', allow_pickle=True)


def runit(lstfiles=alltimesx4, m1=False, m2=False, pn=False):
    '''
    Running the download for each observation.
    '''
    for i, obs in enumerate(lstfiles):
        if len(obs)<10:
            obs = '0'*(10-len(obs))+obs
        getxmmdata(str(obs), pn=pn, m1=m1, m2=m2)
        print(obs)
        print(i)

def extractdata(fold):
    '''
    For extracting tars in a given folder
    '''
    fils = glob.glob(fold+'/*.tar')
    for fil in fils:
        comm = 'tar xopf '+fil
        print(fil)
        p = subprocess.Popen(comm, shell=True, stdout=subprocess.PIPE, stderr = subprocess.PIPE)
        out = p.communicate()

def checksuccess(fold, typ):
    fils = glob.glob(fold+'/*.tar')
    folds = glob.glob('*/')
    folds = [folds[i].split('/')[0] for i in range(len(folds))]
    missed = []
    for fil in fils:
        splt = fil.split('/')[1].split('.')[0].split(typ)[0]
        if splt not in folds:
            comm = 'tar xvf ' +fil
            print(fil)
            #p = subprocess.Popen(comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #out = p.communicate()
            missed.append(fil)
    return missed

def move_extracted(fold, typ):
    fils = glob.glob(fold+'/*.tar')
    folds = glob.glob('*/')
    folds = [folds[i].split('/')[0] for i in range(len(folds))]
    for fil in fils:
        splt = fil.split('/')[1].split('.')[0].split(typ)[0]
        if splt in folds:
            comm = 'mv ' +fil +' extracted/'
            print(fil)
            p = subprocess.Popen(comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = p.communicate()
            #missed.append(fil)