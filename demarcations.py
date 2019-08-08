import numpy as np
def y1_kauffman(xvals):
    yline1_kauffman = 10**(0.61 / (xvals - 0.05) + 1.3) 
    return yline1_kauffman
def y1_kewley(xvals):
    yline1_kewley = 10**(0.61 / (xvals - 0.47) + 1.19) 
    return yline1_kewley
def y2_agn(xvals):
    return 10**(0.72 / (xvals - 0.32) + 1.30)
def y2_linersy2(xvals):
    return 10**(1.89*xvals + 0.76 )
def y3_agn(xvals):
    return 10**(0.73/(xvals+0.59)+1.33)
def y3_linersy2(xvals):
    return 10**(1.18*xvals+1.30)
def mex_main(xvals):
    return 0.375 / (xvals-10.5) + 1.14
def y_stasinska(xvals):
    return 10**( (-30.787+1.1358*xvals+0.27297*xvals**2)*np.tanh(5.7409*xvals)-31.093)

def mex_upper(xvals):
    if type(xvals) == int or type(xvals) == float:
        return 410.24 -109.333*xvals +9.71731*xvals**2 -0.288244*xvals**3
    else:
        yvals = []
        for x in xvals:
            if x < 10:
                yvals.append(mex_main(x))
            else:
                yvals.append(410.24 -109.333*x +9.71731*x**2 -0.288244*x**3)
        return np.array(yvals) 
def mex_lower(xvals):
    if type(xvals) == int or type(xvals) == float:
        return 352.066 - 93.8249*xvals+ 8.32651*xvals**2 -0.246416*xvals**3
    else:
        yvals = []
        for x in xvals:
            if x < 10:
                yvals.append(mex_main(x))
            else:
                yvals.append(352.066 - 93.8249*x+ 8.32651*x**2 -0.246416*x**3)
        return np.array(yvals) 

xline1_kewley =np.log10(np.logspace(-2.5,1,num=100))
xline1_kauffman =np.log10(np.logspace(np.log10(0.007),0,num=100))

yline1_kewley= y1_kewley(xline1_kewley)
yline1_kauffman=y1_kauffman(xline1_kauffman)
yline_stasinska = y_stasinska(xline1_kauffman)

xline1_kewley = 10**(xline1_kewley)
xline1_kauffman = 10**(xline1_kauffman)

xline_mex = np.linspace(8,12,100)
ylineup_mex= mex_upper(xline_mex)
ylinedown_mex = mex_lower(xline_mex)
#agn
xline2_agn= np.log10(np.logspace(-1.5,0.1,num=100))
yline2_agn= y2_agn(xline2_agn)
#liner/sy2
xline2_linersy2= np.log10(np.logspace(-0.3,0.5,num=100))
yline2_linersy2 = y2_linersy2(xline2_linersy2)
xline2_agn = 10**(xline2_agn)
xline2_linersy2 = 10**(xline2_linersy2)

xline3_agn= np.log10(np.logspace(-2.5,-.8,num=100))
yline3_agn= y3_agn(xline3_agn) 
xline3_linersy2= np.log10(np.logspace(-1.1,0,num=100)) 
yline3_linersy2 = y3_linersy2(xline3_linersy2)
xline3_agn = 10**(xline3_agn)
xline3_linersy2 = 10**(xline3_linersy2)


nii_bound=-0.4
