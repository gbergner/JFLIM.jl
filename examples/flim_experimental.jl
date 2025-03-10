# using DelimitedFiles
using JFLIM
using View5D
using TestImages
using NDTools
using Noise

function simulate_flim_data(img1, img2, ntimes, tau1, tau2)
    tau1 = Float32(tau1)
    tau2 = Float32(tau2)
    times = reorient(0:ntimes-1, Val(4))
    dat = img1 .* exp.(.-times./tau1) .+ img2 .* exp.(.-times./tau2)
    return dat
end

function main()
    ISM = nothing, img1 = nothing, img2 = nothing, to_fit=nothing
    if (false)
        ISM = readdlm("ISMdata.txt")
        ISM = reshape(ISM, (512, 512, 256))
        ISM = ISM[:,:, 50:120]

        to_fit = max.(0, reshape(ISM,(size(ISM,1),size(ISM,2),1,size(ISM,3),1)))[:,:,:,:,1];
    else
        timg = Float32.(testimage("resolution_test_512.tif"))
        img1 = timg[1:256,1:256] .* 100
        img2 = timg[257:512,257:512] .* 100
        to_fit = simulate_flim_data(img1, img2, 50, 10.0, 20.0)
        to_fit = Float32.(poisson(Float64.(to_fit)))
    end

    # mytimes = (1:size(to_fit, 4)) *20f0


    use_cuda = true
    amp_positive = true
    # first fit with fixed taus and gaussian noise
    res, fwd = flim_fit(to_fit, ScaleMaximum(); use_cuda = use_cuda, verbose=true, stat=loss_gaussian, iterations=30, num_exponents=2,
                        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
                        # tau_start=reorient([10f0, 20f0], Val(5)),
                        off_start=0.0001f0, bgnoise=1f0);
    res[:τs] 

    # ... and then improve the fit with free global taus and poisson or anscombe noise. reuse the previous result as starting point
    res2, fwd2 = flim_fit(to_fit, ScaleMaximum(); use_cuda = use_cuda, verbose=true, stat=loss_anscombe_pos, iterations=80, num_exponents=2,
                        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
                        all_start=res, bgnoise=1f0);

    amps = res2[:amps]
    τs = res2[:τs] 
    res2.offset

    @vt to_fit fwd2
    set_elements_linked(true)

    @vt res[:amps] 
    @vt amps
end
