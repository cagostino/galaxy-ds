import numpy as np
import subprocess
import os
import glob
datfold = './xray_imgs/xmm3/'
def getxmmdata(obsnumb):
    '''
    Interfacing with the online server for XMM-Newton
    '''
    fold = obsnumb+'/'
    #setting up the unix commands
    m1comm =  'curl -o' +datfold+'m1/'+obsnumb+'m1.tar "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno='+obsnumb+'&instname=M1&extension=FTZ&name=IMAGE_&level=PPS" '
    m2comm =  'curl -o' +datfold+'m2/'+obsnumb+'m2.tar "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno='+obsnumb+'&instname=M2&extension=FTZ&name=IMAGE_&level=PPS" '
    pncomm =  'curl -o' +datfold+'pn/'+obsnumb+'pn.tar "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno='+obsnumb+'&instname=PN&extension=FTZ&name=IMAGE_&level=PPS" '
    #creating the command
    pm1 = subprocess.Popen(m1comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pm2 = subprocess.Popen(m2comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ppn = subprocess.Popen(pncomm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #sending the command for download
    outm1, errm1 = pm1.communicate()
    outm2, errm2 = pm2.communicate()
    outpn, errpn = ppn.communicate()
#
#goodobs = np.loadtxt('./catalogs/goodobstimes.txt',dtype='U32')
#goodobsalltimes is all the XMM observations between log(t_exp) 4.1, 4.5
goodobsalltimes = np.loadtxt('./catalogs/goodobsalltimes.txt', dtype='U32')
def runit():
    '''
    Running the download for each observation.
    '''
    for i, obs in enumerate(goodobsalltimes):
        getxmmdata(obs)
        print(i)

def extractdata(fold):
    '''
    For extracting tars in a given folder
    '''
    fils = glob.glob(fold+'/*.tar')
    for fil in fils:
        comm = 'tar xopf '+fil
        p = subprocess.Popen(comm, shell=True, stdout=subprocess.PIPE, stderr = subprocess.PIPE)
        out = p.communicate()


