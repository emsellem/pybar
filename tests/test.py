infolder = "/home/soft/python/pybar-master/"
%run pybar.py

X, Y = np.meshgrid(np.linspace(-10,10,21), np.linspace(-10,10,21))
F = exp(-(X**2 + Y**2 / 0.8**2) / (2. * 3.0**2))
V = X * F

bar = mybar(Flux=F, Velocity=V, Xin=X, Yin=Y, alphaNorth=20.0, PAbar=50.0)

ex, Xn, Yn, Zn = visualise_data(bar.X_NE, bar.Y_NE, F)
ex, Xn, Yn, Zn = visualise_data(bar.X_bardep, bar.Y_bardep, F)

## example from Witold
## delta=30, i=30, alpha=26.565.
## delta = PA(LON) + phi + 90deg 
## alpha_sky: counterclockwise angle in the sky from the LON to the bar major axis = BAR - LON
## alpha:     counterclockwise angle from the LON to the bar major axis in the disc plane after deprojection
# alpha = atan(tan(alpha_sky)/cos(i))
## phi: counterclockwise angle from the top of the frame to the North direction (after mirroring)
X2, Y2, Vr, Vt = np.loadtxt('data/example_rt.dat').T

delta = 30.0
inclin = 30.0
alphasky = 26.56505117707799
#alphasky = rad2deg(arctan(tan(deg2rad(alpha)) * cos(deg2rad(inclin))))
PAnodes = -60.0
PAbar = alphasky + PAnodes
X, Y, F, V = np.loadtxt('example.dat').T
#step = 5.2040976847096116
step = 5.
minX = step * 100.0
Xn, Yn = np.meshgrid(linspace(-minX, minX, 201), linspace(-minX, minX, 201))
bar = mybar(Flux=F, Velocity=V, Xin=Xn, Yin=Yn, alphaNorth=0.0, PAbar=PAbar, PAnodes=PAnodes, inclin=inclin)

ex, nX, nY, nVr = resample_data(bar.X_bardep, bar.Y_bardep, bar.Vr, newextent=[-500,500,-500,500], newstep=5.0)

bar.get_PatternSpeed_2()

####
alphasky=29.62
inclin=10
delta=0
PAnodes = -90.0
PAbar = alphasky + PAnodes
X, Y, F, V = np.loadtxt('an2_30a10b.dat').T
bar = mybar(Flux=F, Velocity=V, Xin=X, Yin=Y, alphaNorth=0.0, PAbar=PAbar, PAnodes=PAnodes, inclin=inclin)
bar.get_PatternSpeed()


####
# In an22_55a40b.dat i=40, alphasky=47.57. In an22_45a45b.dat i=45, alphasky=35.26. In both delta=0.
alphasky=47.57
inclin=40
delta=0
PAnodes = -90.0
PAbar = alphasky + PAnodes
X, Y, F, V = np.loadtxt('an22_55a40b.dat').T
bar = mybar(Flux=F, Velocity=V, Xin=X, Yin=Y, alphaNorth=0.0, PAbar=PAbar, PAnodes=PAnodes, inclin=inclin)
bar.get_PatternSpeed()

alphasky=35.26
inclin=45
delta=0
PAnodes = -90.0
PAbar = alphasky + PAnodes
X, Y, F, V = np.loadtxt('an22_45a45b.dat').T
bar = mybar(Flux=F, Velocity=V, Xin=X, Yin=Y, alphaNorth=0.0, PAbar=PAbar, PAnodes=PAnodes, inclin=inclin)
bar.get_PatternSpeed()

## NGC 936
import astropy.io.fits as pyfits
v936 = pyfits.getdata("data/VPXF_VS_bin_MS_NGC936_r9.fits")
s936 = pyfits.getdata("data/SPXF_VS_bin_MS_NGC936_r9.fits")
i936 = pyfits.getdata("data/Ima_bin_MS_NGC936_r9.fits")

