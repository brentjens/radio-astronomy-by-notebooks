def pause():
    raw_input('Press <enter> to continue...')

fitsfile = 'p1conc.uvf'
vis = 'p1conc.MS'
cal_field = '0'
sgr_field = '1'

# This is how one would write one's own scripts:
importuvfits(
    fitsfile=fitsfile,
    vis=vis)

vishead()

listobs()

plotants()

pause()

plotms(
    vis=vis,
    field=cal_field)
pause()


freq_hz = 1612e6
lambda_over_d = 299792458/(freq_hz*8685)
res_arcsec = lambda_over_d*180*3600/pi
cell = '%.1f arcsec' % (res_arcsec/3.0,)

imagename = 'images/calibrator-raw'

tclean(
    vis=vis,
    field=cal_field,
    niter=0,
    datacolumn='data',
    imsize=[512, 512],
    cell=cell,
    imagename = imagename,
    interactive=True)
pause()

viewer(infile = imagename+'.image')
pause()

viewer(infile = imagename+'.psf')
pause()

viewer(infile = imagename+'.model')
pause()

viewer(infile = imagename+'.residual')
pause()


caltable = '1748-253.cal'
gaincal(
    vis=vis,
    field=cal_field,
    caltable=caltable,
    smodel=['1.17 Jy', '0 Jy', '0 Jy', '0 Jy'], # I, Q, U, V
    solint='300 s',
    calmode='ap',
    refant='19')
pause()


plotcal(
    caltable=caltable,
    xaxis = 'time',
    yaxis = 'phase',
    iteration = 'antenna',
    subplot = 931,
    plotrange = [None, None, -180, +180])
pause()

plotcal(
    caltable=caltable,
    xaxis = 'time',
    yaxis = 'amp',
    iteration = 'antenna',
    subplot = 931,
    plotrange = [None, None, 0.0, 0.25])
pause()


plotcal(
    caltable=caltable,
    field=cal_field,
    xaxis = 'time',    
    yaxis = 'snr',
    iteration = 'antenna',
    subplot = 931,
    plotrange = [None, None, 0.0, 100])
pause()


applycal(
    vis=vis,
    gaintable=[caltable],
    field='')
pause()



plotms(
    vis=vis,
    field=cal_field,
    plotrange = [],
    xaxis='uvdist',
    yaxis='amp')
pause()

imagename = 'images/calibrator'

tclean(
    vis=vis,
    field=cal_field,
    datacolumn='corrected',
    niter=200,
    imsize=[512, 512],
    cell=cell,
    imagename=imagename,
    interactive=True)
pause()


viewer(infile=imagename+'.image')
pause()

viewer(infile=imagename+'.residual')
pause()


plotms(
    vis=vis,
    field=sgr_field,
    xaxis='uvdist',
    yaxis='amp')
pause()


imagename = 'images/Sgr-A-B-array'
tclean(vis=vis,
       field=sgr_field,
       datacolumn='corrected',
       cell=cell,
       imsize=[1024, 1024],
       niter=10000,
       interactive=True,
       imagename=imagename)
pause()


imagename = 'images/Sgr-A-B-array-uvcut'
tclean(vis=vis,
       field=sgr_field,
       datacolumn='corrected',
       cell=cell,
       imsize=[1024, 1024],
       niter=1000,
       uvrange='>10klambda',
       interactive=True,
       imagename=imagename)
pause()

imagename = 'images/Sgr-A-B-array-msclean'
tclean(vis=vis,
       field=sgr_field,
       datacolumn='corrected',
       cell=cell,
       imsize=[1024, 1024],
       weighting='uniform',
       deconvolver='multiscale',
       scales=[0,1,2,4,8,16,32,64,128],
       usemask='auto-multithresh',
       niter=20000,
       interactive=True,
       imagename=imagename)
pause()





# inp(tclean)
# field = '1'
# niter = 30000
# interactive='True'
# weighting='briggs'
# deconvolver='multiscale'
# scales=[0,1,2,4,8,16,32,64,128]
# usemask='auto-multithresh'
# gridder='wproject'
# wprojplanes=-1
# imagename = 'images/Sgr-A-B-array-wproj'
# go(tclean)

