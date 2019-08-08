import numpy as np
def ra_hr_arr(ra):
    racop = np.empty_like(ra,dtype='U32')
    for i in range(len(ra)):
        rahr = str(ra[i] // 15)
        ram = str((ra[i]%15 *60)//15)
        ras = str((np.round((ra[i]*60)%60)))
        rastr =rahr+' '+ram+' '+ras+' '
        racop[i]= rastr
    return racop
def ra_hr(ra):
    rahr = str(ra // 15)
    ram = str((ra%15 *60)//15)
    ras = str((np.round((ra*60)%60)))
    rastr =rahr+' '+ram+' '+ras+' '
    return rastr