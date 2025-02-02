export flim_fit, get_tau

"""
    get_time_max(data)

Get the time of the maximum intensity in the FLIM data `data`.

# Arguments
- `data::Array{Float32, 4}`: The FLIM data.

"""
function get_time_max(data)
    tmp = argmax(data, dims=4)[:, :, :, 1, 1]   # Find the time of the maximum intensity
    t0_max = ((c)->Tuple(c)[4]).(tmp)  # to convert the CartesianIndex to a tuple and extract the 4th element
    return t0_max
end


"""
    get_tau(data)

Get the pixelwise lifetime of the FLIM data `data` in a fast way by ratioing the intensities of the decay curve in two windows.
The "rapid lifetime determination (RLD)" method is used.
See https://www.researchgate.net/publication/265171406_Precision_of_mono-exponential_decay_estimates_from_rapid_lifetime_determination_in_the_presence_of_signal_photon_and_background_noise

# Arguments
- `data::Array{Float32, 4}`: The FLIM data.
- `sigma::Float32=0`: The standard deviation of the Gaussian filter to apply to the data.
- `irfsigma::Float32=4.0`: The standard deviation of the instrument response function (irf), delaying the start of estimation from the maximum.

"""
function get_tau(data; sigma = 0, irfsigma=4.0, verbose=false)
    mean_decay = mean(data, dims=(1, 2, 3, 5))
    mean_decay = max.(mean_decay, 0)
    ti = round.(Int, get_time_max(mean_decay)[1] .+ irfsigma)
    tend = size(data, 4)
    # find the first index where the intensity is below half of the maximum intensity
    t_half = findfirst(x-> x < mean_decay[:][ti]/2, mean_decay[:][ti:tend])
    tau_est = t_half  / log(2)
    tmid = ti + t_half   # Point time between ti and tend
    tend = ti + t_half*4 # to yield approximately equal signals for both ranges
    # if (verbose)
    #     println("range is $(ti) $(tmid) $(tend)")
    # end

    # Standard Rapid Lifetime Determination (SRLD) using trapezoidal integration
    # a1 = (sum((@view data[:, :, :, ti:tmid-1, :]), dims=4) + sum((@view data[:, :, :, ti+1:tmid, :]), dims=4) )./ 2  # trapez
    # a2 = (sum((@view data[:, :, :, tmid+1:tend-1, :]), dims=4) + sum((@view data[:, :, :, tmid+2:tend, :]), dims=4) )./ 2
    a1 = Float32.(sum((@view data[:, :, :, ti:tmid-1, :]), dims=4))
    a2 = Float32.(sum((@view data[:, :, :, tmid+1:tend, :]), dims=4))
    # a1 = Float32.(sum((@view data[:, :, :, ti:2:tmid-1, :]), dims=4))
    # a2 = Float32.(sum((@view data[:, :, :, ti+1:2:tmid, :]), dims=4))
    if (sigma > 0)
        a1 = filter_gaussian(a1, sigma=sigma);
        a2 = filter_gaussian(a2, sigma=sigma);
    end
    a2 .= max.(a2, 1e-3);
    a1 .= max.(a1, a2 .+ 1e-3, 1e-3);
    Δt = tmid-ti
    factor = ((1-exp(-(tend-(tmid+1))/tau_est))/(1-exp(-((tmid-1)-ti)/tau_est)))
    R = factor * a1 ./ a2  # the factor compensates for the different integration times of the two windows
    @show minimum(R)
    R = max.(R, 1.0001)
    tau = Δt ./ log.(R)
    # tau = min.(max.(tau, 0.1),100.0)
    if (verbose)
        println("mean lifetime is $(mean(tau))")
    end
    return tau
end 


##### See if we can get an individual lifetime-curve fit to each pixel

"""
    multi_exp(params)

Create a single exponential decay function in parallel for the whole 2D or 3D image.
The parameters are individual for each pixel and supplied as a 2D or 3D array.

    It is important that this function has minimal memory allocation
        5th dimension iterates over the mulitexponential components of the decay curve
"""
function multi_exp(params, time_data)
    # tofit should contain: 
    # t0 = params(:t0)
    offset = params(:offset)
    amps = params(:amps)
    τs = params(:τs)

    # the line below works, but is quite memory intensive
    # return sum(offset .+ amps .* MicroscopyTools.soft_theta_pw.(time_data, 0.002f0) .* exp.(.-time_data ./ τs), dims=5);

    return sum(offset .+ amps .* exp.(.- time_data ./ τs), dims=5)

    # # better version but there are some problems in the pullback ?
end

function multi_exp_irf(params, time_data)
    t0 = params(:t0)
    offset = params(:offset)
    amps = params(:amps)
    τs = params(:τs) 

    # the line below works, but is quite memory intensive
    # return sum(offset .+ amps .* MicroscopyTools.soft_theta_pw.(time_data, 0.002f0) .* exp.(.-time_data ./ τs), dims=5);

    # forward = (vec) -> DeconvOptim.conv_aux(conv, multi_exp(time_data, merge(NamedTuple(vec),fixed_val)), otf)
# otf, conv = plan_conv_r(irf_n, measured, 1);

    res = @view sum(offset .+ amps .* MicroscopyTools.soft_theta_pw.(time_data .- t0, 1.0f0) .* exp.(.- (time_data.-t0)./ τs), dims=5)[:,:,:,:,1]
    return res
end

# function multi_exp(t, tofit)
#     # tofit should contain: 
#     @unpack t0, offsets, crosstalk, τ_rot, r0, G, amps, τs = tofit
#     I_iso =  multi_exp_decay(t .- t0, amps, τs)
#     r =  multi_exp_decay(t .- t0, r0, τ_rot)
#     I_par = crosstalk[1].*soft_delta.(t .- t0)  .+ offsets[1] .+ (1 .+ 2 .* r) .* I_iso # .+ ref .* reflection.(t .- t0)
#     I_perp = crosstalk[2].*soft_delta.(t .- t0) .+ offsets[2] .+ G .* (1 .- r) .* I_iso
#     return cat(I_par, I_perp, dims=2)
# end
# forward = (vec) -> DeconvOptim.conv_aux(conv, multi_exp(time_data, merge(NamedTuple(vec),fixed_val)), otf)
# otf, conv = plan_conv_r(irf_n, measured, 1);


"""
    do_fit(to_fit, amp_start, tau_start; times=0:size(to_fit,4)-1)

performs the fit of `to_fit` with the starting amplitudes `amp_start` and the lifetimes `tau_start`.
"""
function do_fit(to_fit, off_start, amp_start, tau_start, t0_start=1; times=0:size(to_fit,4)-1, iterations=20, stat=loss_gaussian, bgnoise=2f0, verbose=true, irf=nothing)
    if isnothing(irf)
        all_start = (offset=Positive(Float32.(off_start)), amps=Positive(Float32.(amp_start)), τs=Positive(Float32.(tau_start)))
    else
        all_start = (offset=Positive(Float32.(off_start)), amps=Positive(Float32.(amp_start)), τs=Positive(Float32.(tau_start)), t0=Float32.(t0_start))
    end

    to_fit = Float32.(to_fit)
    times = Float32.(reorient(times[:], Val(4)))

    fwd = let 
        if isnothing(irf)
            (vec) -> multi_exp(vec, times);
        else
            all_start = (offset=Positive(Float32.(off_start)), amps=Positive(Float32.(amp_start)), τs=Positive(Float32.(tau_start)), t0=Float32.(t0_start))
            irf = eltype(to_fit).(irf)
            otf, pconv = plan_conv_psf(to_fit, irf, 4);
            # @noinline function doconv(vec) # somehow inlining this function is causins an internal Zygote error
            #     @noinline tmp = multi_exp_irf(vec, times)
            #     @noinline return DeconvOptim.conv_aux(pconv, tmp, otf);
            # end
            # (vec) -> multi_exp_irf(vec, times)
            
            (vec) -> conv_aux(pconv, multi_exp_irf(vec, times), otf); # somehow inlining this function is causins an internal Zygote error
            # doconv
        end
    end
    # afer_irf = DeconvOptim.conv_aux(conv, exp.(.- MicroscopyTools.soft_theta_pw.(time_data, 0.2f0) ./ τs)

    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, all_start);
    @time start_simulation = forward(start_vals);

    myloss = loss(to_fit, forward, stat, bgnoise);
    # myloss = loss(to_fit, forward, loss_anscombe_pos);
    if (verbose)
        println("Starting loss: ", myloss(start_vals))
    end

    # optim_res = InverseModeling.optimize(loss(to_fit, forward), start_vals, iterations=0);
    @time optim_res1 = optimize_model(myloss, start_vals, iterations=iterations, show_trace=verbose);

    if (verbose)
        # @show optim_res
    end
    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    bare, res = get_fit_results(optim_res1)
    fit = forward(bare);

    return bare, res, fit
end

global junc

has_nan(x) = any(isnan, x)

is_pos(x) = all(x .>= 0)

"""
    FLIM_fit(to_fit; use_cuda=false, verbose=true)

Fit the FLIM data `measured` with a (multi-)exponential decay model.
The function returns the fit parameters and the fit itself.
All results are in time bins as units.

# Arguments
- `to_fit::Array{Float32, 4}`: The measured data to fit.
- `use_cuda::Bool=false`: Use CUDA for the fitting.
- `verbose::Bool=true`: Print the fitting progress.
- `stat::Function=loss_poisson`: The loss function to use for the fitting.
- `iterations::Int=10`: The number of iterations to perform.

"""
function flim_fit(to_fit; use_cuda=false, verbose=true, stat=loss_poisson, iterations=10, irf=nothing, num_exponents=1)
    any(isnan, to_fit) && error("NaN in data");

    sigma = 0f0
    if isa(irf, Number) # generate a Gaussian IRF
        sigma = Float32(irf)
        irf = JFLIM.normal(Float32, (1,1,1,size(to_fit,4)), sigma=sigma)
        irf = irf ./ sum(irf) # changes datatype
    elseif isa(irf, Array)
        sigma = 4f0
        irf ./= sum(irf)
    end
    flim_curve = mean(to_fit, dims=(1, 2, 3, 5))
    img_size = (size(to_fit)[1:3]...,1,1)
    numtimes = size(to_fit, 4)
    off_start = mean(flim_curve[end-(numtimes÷10+1)]) .* ones(Float32, img_size)
    to_fit_no_offset = to_fit .- off_start    # may contain negative values
    tau_start = max.(1, get_tau(to_fit_no_offset, irfsigma=sigma, verbose=verbose)) # 25f0 .* ones(Float32, img_size)
    amp_start = max.(1e-3, sum(to_fit_no_offset, dims=(3,4,5))./tau_start) # due to the integral of the exponential decay

    if num_exponents > 1 # split the estimations into different exponents
        amp_start =  ones(Float32, (1,1,1,1,num_exponents)) .* amp_start./num_exponents # due to the integral of the exponential decay
        tau_start =  ones(Float32, (1,1,1,1,num_exponents)) .* tau_start # due to the integral of the exponential decay
        # 5th dimension in tau_start contains the different exponents
        for n in 1:num_exponents
            tau_start[:,:,:,:,n] *= (n-0.5f0) # /num_exponents
        end
    end
    to_fit_no_offset = nothing
    # amp_start = maximum(to_fit, dims=(3,4,5)) # ones(Float32, (size(to_fit)[1:3]...,1,1))
    times = axes(to_fit, 4).-1
 
    t0_start = 1
    if !isnothing(irf)
        t0_start = get_time_max(to_fit)
        t0_start = mean(t0_start .- 2*sigma) # The 2*sigma corrects for the apparant shift caused by the convolution of a single-sided peak with a Gaussian
    end

    if use_cuda
        to_fit = CuArray(to_fit)
        off_start = CuArray(off_start)
        tau_start = CuArray(tau_start)
        amp_start = CuArray(amp_start)
        t0_start = CuArray([t0_start;;;;;])
        times = CuArray(times)
        if !isnothing(irf)
            irf = CuArray(irf)
        end
    end
    bare, res, fit = do_fit(to_fit, off_start, amp_start, tau_start, t0_start; times=times[:], iterations=iterations, stat=stat, verbose=verbose, irf=irf);

    if use_cuda
        fit = Array(fit)
        if isnothing(irf)
            res = (amps=Array(res.amps), τs=Array(res.τs), offset=Array(res.offset))
        else
            res = (amps=Array(res.amps), τs=Array(res.τs), offset=Array(res.offset), t0=Array(res.t0))
        end
    end

    any(isnan, fit) && error("NaN in fit");

    return res, fit
end
