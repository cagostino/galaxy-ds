#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class MPAJHU_Spec:
    def __init__(self, GSW_Cat, sdssobj, find=True, sedtyp=0, gsw=False):
        self.GSW_Cat = GSW_Cat
        self.sdssobj = sdssobj
        self.sedtyp= sedtyp
        if find:
            self.spec_inds_prac, self.spec_plates_prac, self.spec_fibers_prac, self.spec_mass_prac, self.make_prac, self.miss_prac, self.ids_prac = self.matchspec_prac(self.GSW_Cat, self.sdssobj, gsw = gsw, sedtyp = sedtyp)
            self.spec_inds_full, self.spec_plates_full, self.spec_fibers_full, self.spec_mass_full, self.make_full, self.miss_full, self.ids_full = self.matchspec_full(self.GSW_Cat, self.sdssobj, gsw = gsw, sedtyp = sedtyp)
            
    def matchspec_prac(self,GSW_Cat,sdssobj, gsw=False, sedtyp=0):
        '''
        For practical use, essentially take the output and use it to index
        '''
        if gsw:
            plate_id = GSW_Cat.allplate
            fiber_id = GSW_Cat.allfiber
            mjds = GSW_Cat.allmjds
            sedflag = GSW_Cat.sedflags
            ids = GSW_Cat.allids            
        else:
            plate_id = GSW_Cat.matchplate
            fiber_id = GSW_Cat.matchfiber
            mjds = GSW_Cat.matchmjd
            sedflag = GSW_Cat.matchsedflags
            ids = GSW_Cat.ids
        obj_ids = []
        inds = []
        plates =[]
        fibers = []
        masses = []
        misses = []
        makes = []
        for i in range(len(plate_id)):
            if gsw:
                if i%1000==0:
                    print(i)
            if sedflag[i] == sedtyp:
                valid_id = np.array([np.where((sdssobj.allplateids == plate_id[i] ) &(sdssobj.allfiberids==fiber_id[i]) &(sdssobj.allmjds == mjds[i]))  ] )[0]
                if valid_id.size !=0:
                    
                    inds.append(valid_id)
                    plates.append(sdssobj.allplateids[valid_id])
                    fibers.append(sdssobj.allfiberids[valid_id])
                    masses.append(sdssobj.all_sdss_avgmasses[valid_id])
                    obj_ids.append(ids[i])
    
                    makes.append(i)
                else:
                    misses.append(i)
            else:
                misses.append(i)
        return np.array(inds), np.array(plates), np.array(fibers), np.array(masses), np.array(makes), np.array(misses), np.array(ids)
    def matchspec_full(self, GSW_Cat, sdssobj, gsw = False, sedtyp=0):
        '''
        This function keeps track of which ones match and which do not match.
        '''
        if gsw:
            plate_id = GSW_Cat.allplate
            fiber_id = GSW_Cat.allfiber
            mjds = GSW_Cat.allmjds
            sedflag = GSW_Cat.sedflags
            ids = GSW_Cat.allids            
        else:
            plate_id = GSW_Cat.matchplate
            fiber_id = GSW_Cat.matchfiber
            mjds = GSW_Cat.matchmjd
            sedflag = GSW_Cat.matchsedflags
            ids = GSW_Cat.ids
        obj_ids = []
        inds = []
        plates = []
        fibers = []
        makes = []
        misses = []
        masses = []
        for i in range(len(plate_id)):
            if gsw:
                if i%1000==0:
                    print(i)
            if sedflag[i] == sedtyp:
                valid_id = np.array([np.where((sdssobj.allplateids == plate_id[i] ) &(sdssobj.allfiberids==fiber_id[i]) &(sdssobj.allmjds == mjds[i]))  ] )[0]
                if valid_id.size != 0:
                    inds.append(valid_id)
                    plates.append(sdssobj.allplateids[valid_id])
                    fibers.append(sdssobj.allfiberids[valid_id])
                    makes.append(i)
                    masses.append(sdssobj.all_sdss_avgmasses[valid_id])
                    obj_ids.append(ids[i])
                else:
                    misses.append(i)
            else:
                inds.append([])
                plates.append([])
                fibers.append([])
                misses.append(i)
    
        return np.array(inds), np.array(plates), np.array(fibers), np.array(masses), np.array(makes), np.array(misses), np.array(obj_ids)
    

