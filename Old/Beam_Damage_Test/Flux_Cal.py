

Pind_Current = 32.7e-6*5.10                
Photon_Energy = 12.4e3                      # Photon energy in eV
Spec_Res = 0.1882
loss_mirror = 0.94*0.94                  # Loss from mirror reflectivity
loss_diamond = 0.735                     # Loss from BPM and CVD diamond window at 8-ID-E
norm_ring_current = 1.0                  
loss_factor = loss_mirror*loss_diamond*norm_ring_current

# Constants for flux calculation
ev2J = 1.6e-19                           # Conversion from eV to Joule  
E_0p1bw = 9.89e-05/1e-3                  # From Xianbo's calculation on 07/22/2025  

# Calculation of raw and corrected flux
Beam_Power = Pind_Current/Spec_Res
Raw_Flux = Beam_Power/(Photon_Energy*ev2J)       # Flux in photons per second
Corrected_Flux = Raw_Flux/loss_factor


print(f"\n The measured raw flux of the 8-ID-I beamline at {Photon_Energy/1e3:.1f} keV is: {Raw_Flux:.2e} \
photons/sec")
