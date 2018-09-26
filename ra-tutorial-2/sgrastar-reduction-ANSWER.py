inp importuvfits
fitsfile = 'p1conc.uvf'
vis = 'p1conc.MS'
go importuvfits

inp vishead
go vishead

inp listobs
go listobs

inp plotants
go plotants

inp plotms
field = '0'
go plotms


inp tclean
field = '0'
niter = 0
datacolumn = 'data'
imsize = [512, 512]

freq_hz = 1612e6
lambda_over_d = 299792458(freq_hz*8685)
res_arcsec = lambda_over_d*180*3600/pi

cell = '%.1f arcsec' % (res_arcsec/3.0,)
imagename = 'calibrator-raw'
interactive=True
go tclean


inp viewer
infile = imagename+'.image'
go viewer

infile = imagename+'.psf'
go viewer

infile = imagename+'.residual'
go viewer

infile = imagename+'.model'
go viewer

inp gaincal
caltable = '1748-253.cal'
smodel = ['1.17 Jy', '0 Jy', '0 Jy', '0 Jy'] # I, Q, U, V
solint = '300 s'
calmode = 'ap'
refant = '19'
go gaincal

inp plotcal
xaxis = 'time'
yaxis = 'phase'
iteration = 'antenna'
subplot = 931
plotrange = [None, None, -180, +180]
go plotcal

inp plotcal
yaxis = 'amp'
plotrange = [None, None, 0.0, 0.25]
go plotcal

inp plotcal
yaxis = 'snr'
plotrange = [None, None, 0.0, 100]
go plotcal


inp applycal
gaintable = [caltable]
field = ''
go applycal

inp plotms
field='0'
plotrange = []
xaxis='uvdist'
yaxis='amp'
go plotms

inp tclean
niter=200
imagename = 'calibrator'
datacolumn = 'corrected'
interactive = True
go tclean

inp viewer
infile= imagename+'.image'
go


inp plotms
field = '1'
xaxis = 'uvdist'
yaxis = 'amp'
go plotms

inp tclean
field = '1'
niter = 10000
interactive = True
imagename = 'Sgr-A-B-array'
go




inp tclean
field = '1'
niter = 20000
interactive='True'
weighting='uniform'
deconvolver='multiscale'
scales=[0,1,2,4,8,16,32,64,128]
usemask='auto-multithresh'
go tclean



inp tclean
field = '1'
niter = 25000
interactive='True'
weighting='uniform'
deconvolver='multiscale'
scales=[0,1,2,4,8,16,32,64,128]
usemask='auto-multithresh'
gridder='wproject'
wprojplanes=-1
go tclean








