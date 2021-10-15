#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:59:02 2020

@author: caug
"""

#lupton
def gr_to_V_lupton(g,r):
    return g-0.5784*(g-r)-0.0038
def gr_to_R_lupton(g,r):
    return r-0.1837*(g-r)-0.0971
def gr_to_V_lester(g,r):
    return g-0.59*(g-r)-0.01


tts8_V_lup = gr_to_V_lupton(16.34, 16.12)
tts8_R_lup = gr_to_R_lupton(16.34, 16.12)
tts8_V_les = gr_to_V_lester(16.34, 16.12)

tts4_V_lup = gr_to_V_lupton(16.04, 13.71)
tts4_R_lup = gr_to_R_lupton(16.04, 13.71)
tts4_V_les = gr_to_V_lester(16.04, 13.71)

tts31_V_lup = gr_to_V_lupton(18.08, 16.9)
tts31_R_lup = gr_to_R_lupton(18.08, 16.9)
tts31_V_les = gr_to_V_lester(18.08, 16.9)

tts59_V_lup = gr_to_V_lupton(14.54, 14.08)
tts59_R_lup = gr_to_R_lupton(14.54, 14.08)
tts59_V_les = gr_to_V_lester(14.54, 14.08)

tts22_V_lup = gr_to_V_lupton(15.72, 12.79)
tts22_R_lup = gr_to_R_lupton(15.72, 12.79)
tts22_V_les = gr_to_V_lester(15.72, 12.79)

tts23_V_lup = gr_to_V_lupton(16.79, 16.18)
tts23_R_lup = gr_to_R_lupton(16.79, 16.18)
tts23_V_les = gr_to_V_lester(16.79, 16.18)

tts57_V_lup = gr_to_V_lupton(17.27, 16.22)
tts57_R_lup = gr_to_R_lupton(17.27, 16.22)
tts57_V_les = gr_to_V_lester(17.27, 16.22)


tts104_V_lup = gr_to_V_lupton(14.92, 13.94)
tts104_R_lup = gr_to_R_lupton(14.92, 13.94)
tts104_V_les = gr_to_V_lester(14.92, 13.94)

tts210_V_lup = gr_to_V_lupton(17.19, 16.14)
tts210_R_lup = gr_to_R_lupton(17.19, 16.14)
tts210_V_les = gr_to_V_lester(17.19, 16.14)
