# -*- coding: utf-8 -*-


from EMAN2 import *


volume = EMData("/home/lqhuang/Git/orientation-python/particle/EMD-6044.map")

display(volume)

sym = Symmetries.get("c1")
orients = sym.gen_orientations("eman", {"delta": 30})

# euler_display(orients)

proj = [volume.project("standard", t) for t in orients]

# display(proj)

boxsize = 120
pad = 125

# p contains a list of projection objects with the "xform.projection" attribute
recon=Reconstructors.get("fourier", {"sym":"c1","size":(pad,pad,pad),"mode":"gauss_2","verbose":True})

# In typical usage this sequence would be repeated multiple times, with information about quality
# and normalization improving on each cycle. 2-3 cycles is generally sufficient. This has 1 1/2 iterations
# hardcoded. ie - insert slices once, check qualities, then reconstruct one more time.
recon.setup()
scores=[]

# First pass to assess qualities and normalizations
for i,p in enumerate(proj):
  p2=p.get_clip(Region(-(pad-boxsize)/2,-(pad-boxsize)/2, pad, pad))
  p2=recon.preprocess_slice(p2,p["xform.projection"])
  recon.insert_slice(p2,p["xform.projection"],1.0)

# after building the model once we can assess how well everything agrees
for p in proj:
  p2=p.get_clip(Region(-(pad-boxsize)/2,-(pad-boxsize)/2,pad,pad))
  p2=recon.preprocess_slice(p2,p["xform.projection"])
  recon.determine_slice_agreement(p2,p["xform.projection"],1.0,True)
  scores.append((p2["reconstruct_absqual"],p2["reconstruct_norm"]))

# setup for the second run
recon.setup()

thr=0.7*(scores[len(scores)/2][0]+scores[len(scores)/2-1][0]+scores[len(scores)/2+1][0])/3       # this is rather arbitrary
for i,p in enumerate(proj):
  if scores[i][0]<thr : continue

  p2=p.get_clip(Region(-(pad-boxsize)/2,-(pad-boxsize)/2,pad,pad))
  p2=recon.preprocess_slice(p2,p["xform.projection"])
  p2.mult(scores[i][1])
  recon.insert_slice(p2,p["xform.projection"],1.0)

ret=recon.finish(True)
ret=ret.get_clip(Region((pad-boxsize)/2,(pad-boxsize)/2,(pad-boxsize)/2,boxsize,boxsize,boxsize))

display(ret)