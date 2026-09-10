vis_75_cal = 'bllac.4cm.cal.MS'

listobs(vis=vis_75_cal)
plotms(vis=vis_75_cal)

bl_max_lambda = 2.5e8
cell = '%.5farcsec' % (1/bl_max_lambda*(180*3600/pi)/3,)
imsize=[256,256]

for weighting in ['natural', 'uniform', 'briggs']:
    imagename = '7.5GHz-%s' % weighting
    tclean(vis=vis_75_cal,
           imagename=imagename,
           cell=cell,
           imsize=imsize,
           weighting=weighting,
           niter=1000,
           deconvolver='multiscale',
           scales=[0,1,2,4,8,16],
           usemask='auto-multithresh')


weighting='briggs'
imagename = '7.5GHz-%s-taper' % weighting
tclean(vis=vis_75_cal,
       imagename=imagename,
       cell=cell,
       imsize=imsize,
       uvtaper=['0.0002arcsec'],
       weighting=weighting,
       niter=1000,
       deconvolver='multiscale',
       scales=[0,1,2,4,8,16],
       usemask='auto-multithresh')


imagename = '7.5GHz-mod-%s' % weighting
vis_75_mod='bllac.4cm.mod.MS'
tclean(vis=vis_75_mod,
       imagename=imagename,
       cell=cell,
       imsize=imsize,
       weighting=weighting,
       niter=1000,
       deconvolver='multiscale',
       scales=[0,1,2,4,8,16],
       usemask='auto-multithresh')


## Selfcal!

weighting='briggs'
for sc_iter, (niter, calmode) in enumerate(zip([1, 10, 100, 1000, 2000, 4000],
                                               ['p', 'p', 'p', 'p', 'ap', 'ap'])):
    imagename='sc-%02d' % (sc_iter,)
    tclean(vis=vis_75_mod,
           imagename=imagename,
           cell=cell,
           imsize=imsize,
           weighting=weighting,
           niter=niter,
           deconvolver='multiscale',
           scales=[0,1,2,4,8,16],
           usemask='auto-multithresh',
           savemodel='modelcolumn')
    
    caltable=imagename+'-'+calmode+'.cal'

    gaincal(vis=vis_75_mod,
            caltable=caltable,
            calmode=calmode,
            refant='ANT5',
            solint='300s')

    plotcal(caltable=caltable,
    )

    applycal(vis=vis_75_mod,
             gaintable=[caltable],
             gainfield=['0'])
