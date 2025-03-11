export flim_fit, get_tau
export ScaleStdDev, ScaleMaximum, ScaleMean, ScaleIQR, ScaleNorm, ScaleQ10

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
function multi_exp(params, time_data) # , tmp_decays, result)
    # tofit should contain: 
    # t0 = params(:t0)
    offset = params(:offset)
    amps = params(:amps)
    τs = params(:τs)

    # the line below works, but is quite memory intensive
    # return sum(offset .+ amps .* MicroscopyTools.soft_theta_pw.(time_data, 0.002f0) .* exp.(.-time_data ./ τs), dims=5);
    tmp_decays = exp.(.- time_data ./ τs)
    result = sum(offset .+ amps .* tmp_decays, dims=5)
    # slice(result, 5, Val(1)) .= offset .+ amps .* tmp_decays
    # for d in 2:lastindex(res, 5)
    #     slice(result, 5, Val(d)) .+= offset .+ slice(amps, 5, Val(d) .* tmp_decays
    # end
    # return sum(offset .+ amps .* tmp_decays, dims=5)
    # # better version but there are some problems in the pullback ?
    return result
end

function multi_exp_irf(params, time_data) #, tmp_decays, result)
    t0 = params(:t0)
    offset = params(:offset)
    amps = params(:amps)
    τs = params(:τs) 

    # the line below works, but is quite memory intensive
    # return sum(offset .+ amps .* MicroscopyTools.soft_theta_pw.(time_data, 0.002f0) .* exp.(.-time_data ./ τs), dims=5);

    # forward = (vec) -> DeconvOptim.conv_aux(conv, multi_exp(time_data, merge(NamedTuple(vec),fixed_val)), otf)
# otf, conv = plan_conv_r(irf_n, measured, 1);
    tmp_decays = MicroscopyTools.soft_theta_pw.(time_data .- t0, 1.0f0) .* exp.(.- (time_data.-t0)./ τs)

    result = @view sum(offset .+ amps .* tmp_decays, dims=5)[:,:,:,:,1]
    # @show size(tmp_decays)
    # @show size(time_data)
    # @show size(result)
    return result
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

function get_start_vals(tau_start, off_start, amp_start, t0_start=nothing; fixed_tau, fixed_offset=true, amp_positive=true)
    tau_start = (fixed_tau) ? Fixed(Float32.(tau_start)) : Positive(Float32.(tau_start))
    off_start = (fixed_offset) ? Fixed(Float32.(off_start)) : Positive(Float32.(off_start))

    amps_start = (amp_positive) ? Positive(Float32.(amp_start)) : Float32.(amp_start)
    if isnothing(t0_start)
        all_start = (offset=off_start, amps=amps_start, τs=tau_start)
    else
        all_start = (offset=off_start, amps=amps_start, τs=tau_start, t0=Float32.(t0_start))
    end
    return all_start
end

function get_fwd(to_fit, irf, mytimes) # , tmp_decays, result)
    if isnothing(irf)
        return (vec) -> multi_exp(vec, mytimes) #, tmp_decays, result);
    else
        irf = eltype(to_fit).(irf)
        # all_start = (offset=Positive(Float32.(off_start)), amps=Positive(Float32.(amp_start)), τs=tau_start, t0=Float32.(t0_start))
        # @noinline function doconv(vec) # somehow inlining this function is causins an internal Zygote error
        #     @noinline tmp = multi_exp_irf(vec, mytimes)
        #     @noinline return DeconvOptim.conv_aux(pconv, tmp, otf);
        # end
        # (vec) -> multi_exp_irf(vec, mytimes)
        # @show size(to_fit)
        # @show size(irf)
        otf, pconv = plan_conv_psf(to_fit, irf, 4); # allocates an array internally        
        return (vec) -> conv_aux(pconv, multi_exp_irf(vec, mytimes), otf); # somehow inlining this function is causins an internal Zygote error
        # doconv
    end
end

function get_fwd_val(to_fit, all_start, irf, mytimes; stat=loss_gaussian, bgnoise=2f0) # , tmp_decays, result
    fwd =  get_fwd(to_fit, irf, mytimes) # , tmp_decays, result
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, all_start);
    myloss = loss(to_fit, forward, stat, bgnoise);
    return myloss(start_vals)
end

function alloc_memory(to_fit, all_start)
    decay_sz = (size(all_start.τs)[1:3]..., size(to_fit)[4], size(all_start.τs, 5));
    tmp_decays = similar(to_fit, decay_sz)
    result = similar(to_fit)
    return tmp_decays, result
end

# function get_fwd_val(to_fit, all_start, irf, mytimes; stat=loss_gaussian, bgnoise=2f0)
#     # tmp_decays, result = alloc_memory(to_fit, all_start)
#     return get_fwd_val(to_fit, all_start, irf, mytimes; stat=stat, bgnoise=bgnoise)
# end

"""
    do_fit(to_fit, amp_start, tau_start; mytimes=0:size(to_fit,4)-1)

performs the fit of `to_fit` with the starting amplitudes `amp_start` and the lifetimes `tau_start`.
"""
function do_fit(to_fit, all_start; mytimes=0:size(to_fit,4)-1, iterations=20, stat=loss_gaussian, bgnoise=2f0, verbose=true, irf=nothing, fixed_tau=true)
    # tmp_decays, result = alloc_memory(to_fit, all_start);
    fwd = get_fwd(to_fit, irf, mytimes); #, tmp_decays, result
    # fwd = let 
    #     if isnothing(irf)
    #         (vec) -> multi_exp(vec, mytimes);
    #     else
    #         # all_start = (offset=Positive(Float32.(off_start)), amps=Positive(Float32.(amp_start)), τs=tau_start, t0=Float32.(t0_start))
    #         irf = eltype(to_fit).(irf)
    #         otf, pconv = plan_conv_psf(to_fit, irf, 4);
    #         # @noinline function doconv(vec) # somehow inlining this function is causins an internal Zygote error
    #         #     @noinline tmp = multi_exp_irf(vec, mytimes)
    #         #     @noinline return DeconvOptim.conv_aux(pconv, tmp, otf);
    #         # end
    #         # (vec) -> multi_exp_irf(vec, mytimes)
            
    #         (vec) -> conv_aux(pconv, multi_exp_irf(vec, mytimes), otf); # somehow inlining this function is causins an internal Zygote error
    #         # doconv
    #     end
    # end
    # # afer_irf = DeconvOptim.conv_aux(conv, exp.(.- MicroscopyTools.soft_theta_pw.(time_data, 0.2f0) ./ τs)

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
    FLIM_fit(to_fit; scale_factor=nothing, use_cuda=false, verbose=true, stat=loss_poisson,
                    iterations=10, irf=nothing, num_exponents=1, fixed_tau=true, fixed_offset=true, amp_positive=true,
                    tau_start=nothing, global_tau=true, off_start=nothing, amp_start=nothing, t0_start=nothing, all_start=nothing, bgnoise=2f0)

Fit the FLIM data `measured` with a (multi-)exponential decay model.
The function returns the fit parameters and the fit itself.
All results are in time bins as units.

# Arguments
- `to_fit::Array{Float32, 4}`: The measured data to fit.
- `use_cuda::Bool=false`: Use CUDA for the fitting.
- `verbose::Bool=true`: Print the fitting progress.
- `stat::Function=loss_poisson`: The loss function to use for the fitting.
- `iterations::Int=10`: The number of iterations to perform.
- `irf::Union{Nothing, Array{Float32, 4}}=nothing`: The instrument response function (IRF) to use for the fitting.
- `num_exponents::Int=1`: The number of exponential components to fit.
- `fixed_tau::Bool=true`: Fix the lifetime values.
- `fixed_offset::Bool=true`: Fix the offset values.
- `amp_positive::Bool=true`: Fix the amplitudes to be positive.
- `tau_start::Union{Nothing, Array{Float32, 5}}=nothing`: The starting lifetime values. Need to be oriented along dimension 5 for multiple exponentials
- `global_tau::Bool=true`: Use a global lifetime value.
- `off_start::Union{Nothing, Float32}=nothing`: The starting offset value.
- `amp_start::Union{Nothing, Array{Float32, 4}}=nothing`: The starting amplitude values.
- `t0_start::Union{Nothing, Float32}=nothing`: The starting time value.
- `all_start::Union{Nothing, NamedTuple}=nothing`: The starting values for all parameters. If this is given, other starting values (tau_start, off_start, amp_start, t0_start) are ignored.
- `bgnoise::Float32=2f0`: The background noise level.

"""
function flim_fit(to_fit; scale_factor=nothing, use_cuda=false, verbose=true, stat=loss_poisson, iterations=10, irf=nothing, num_exponents=1, fixed_tau=true, fixed_offset=true, amp_positive=true,
                    tau_start=nothing, global_tau=true, off_start=nothing, amp_start=nothing, t0_start=nothing, all_start=nothing, bgnoise=2f0)
    any(isnan, to_fit) && error("NaN in data");
    if !isnothing(all_start)
        tau_start=all_start.τs
        off_start=all_start.offset[1]
        amp_start=all_start.amps        
        t0_start= haskey(all_start, :t0) ? all_start.t0 : t0_start
    end

    sigma = 0f0
    if isa(irf, Number) # generate a Gaussian IRF
        sigma = Float32(irf)
        irf = JFLIM.normal(Float32, (1,1,1,size(to_fit,4)), sigma=sigma)
        irf = irf ./ sum(irf, dims=4) # changes datatype
    elseif isa(irf, Array)
        sigma = 2f0
        irf ./= sum(irf, dims=4) # normalize each IRF individually
    end
    # flim_curve = mean(to_fit, dims=(1, 2, 3, 5))
    # img_size = (size(to_fit)[1:3]...,1,1)
    # numtimes = size(to_fit, 4)
    if isnothing(off_start)
        off_start = 0.1 # mean(flim_curve[end-(numtimes÷10+1)]) .* ones(Float32, img_size)
    end
    to_fit_no_offset = max.(to_fit .- off_start, 0f0)    # may contain negative values
    if isnothing(tau_start)
        tau_start = max.(1, get_tau(to_fit_no_offset, irfsigma=sigma, verbose=verbose)) # 25f0 .* ones(Float32, img_size)
        if (global_tau)
            tau_start = mean(tau_start,dims=(1, 2)) # 25f0 .* ones(Float32, img_size)
        end
        if isnothing(amp_start)
            amp_start = max.(1e-8, sum(to_fit_no_offset, dims=4)./tau_start)./ size(tau_start,3) # due to the integral of the exponential decay
        end

        if num_exponents > 1 # split the estimations into different exponents
            amp_start =  ones(Float32, (1,1,1,1,num_exponents)) .* amp_start./num_exponents # due to the integral of the exponential decay
            tau_start =  ones(Float32, (1,1,1,1,num_exponents)) .* tau_start # due to the integral of the exponential decay
            # 5th dimension in tau_start contains the different exponents
            for n in 1:num_exponents
                tau_start[:,:,:,:,n] *= (n-0.5f0) # /num_exponents
            end
        end
    else
        if (size(tau_start,5) > num_exponents)            
            tau_start = tau_start[:,:,:,:,1:num_exponents]
        end
        if (size(tau_start,3) > size(to_fit,3))            
            tau_start = tau_start[:,:,1:size(to_fit,3),:,:]
        end
        if isnothing(amp_start)
            amp_start = max.(1e-8, sum(to_fit_no_offset, dims=4)./tau_start) ./ size(tau_start,3) # due to the integral of the exponential decay
        end
        if (size(amp_start,5) > num_exponents)            
            amp_start = amp_start[:,:,:,:,1:num_exponents]
        end
        if (size(amp_start,3) > size(to_fit,3))            
            amp_start = amp_start[:,:,1:size(to_fit,3),:,:]
        end
    end
    to_fit_no_offset = nothing
    # amp_start = maximum(to_fit, dims=(3,4,5)) # ones(Float32, (size(to_fit)[1:3]...,1,1))
    mytimes = axes(to_fit, 4).-1

    if isnothing(t0_start)
        t0_start = nothing # for decay-only fits
        if !isnothing(irf)
            t0_start = get_time_max(to_fit)
            t0_start = mean(t0_start .- 2*sigma) # The 2*sigma corrects for the apparant shift caused by the convolution of a single-sided peak with a Gaussian
        end
    end

    to_fit = Float32.(to_fit)
    mytimes = Float32.(reorient(mytimes[:], Val(4)))

    if !isnothing(scale_factor)
        scale_factor = Float32.(scale_factor)
        println("scale factor is: $(scale_factor)")
        if (verbose)
            all_start = get_start_vals(tau_start, off_start, amp_start, t0_start; fixed_tau=fixed_tau, fixed_offset=fixed_offset, amp_positive=amp_positive)
            println("Initial starting loss (before scale): ", get_fwd_val(to_fit, all_start, irf, mytimes; stat = stat, bgnoise=bgnoise))
        end
        to_fit = to_fit ./ scale_factor
        amp_start = amp_start ./ scale_factor
        off_start = off_start ./ scale_factor
    end

    if use_cuda
        to_fit = CuArray(to_fit)
        off_start = CuArray([off_start])
        tau_start = CuArray(tau_start)
        amp_start = CuArray(amp_start)
        t0_start = isnothing(t0_start) ? nothing : CuArray([t0_start;;;;;])
        mytimes = CuArray(mytimes)
        if !isnothing(irf)
            irf = CuArray(irf)
        end
    end
    all_start = get_start_vals(tau_start, off_start, amp_start, t0_start; fixed_tau=fixed_tau, fixed_offset=fixed_offset, amp_positive=amp_positive)

    bare, res, fit = do_fit(to_fit, all_start; mytimes=mytimes, iterations=iterations, stat=stat, verbose=verbose, irf=irf, fixed_tau=fixed_tau, bgnoise=bgnoise);

    if use_cuda
        fit = Array(fit)
        if isnothing(irf)
            res = (amps=Array(res.amps), τs=Array(res.τs), offset=Array(res.offset))
        else
            res = (amps=Array(res.amps), τs=Array(res.τs), offset=Array(res.offset), t0=Array(res.t0))
        end
    end

    any(isnan, fit) && error("NaN in fit");

    if !isnothing(scale_factor)
        res_amps = res.amps .* scale_factor
        res_offset = res.offset .* scale_factor        
        res = haskey(res, :t0) ? (amps=res_amps, offset=res_offset, t0=res.t0, τs=res.τs) : (amps=res_amps, offset=res_offset, τs=res.τs)
        fit .*= scale_factor
        to_fit = Array(to_fit .* scale_factor)
        irf = isnothing(irf) ? nothing : Array(irf)
        mytimes = Array(mytimes)
        if (verbose)
            println("Final loss (undoing scale): ", get_fwd_val(to_fit, res, irf, mytimes; stat = stat, bgnoise=bgnoise))
        end
    end

    return res, fit
end

abstract type ScalingType end
struct ScaleStdDev<:ScalingType end
struct ScaleMaximum <:ScalingType end
struct ScaleMean<:ScalingType end
struct ScaleIQR<:ScalingType end
struct ScaleNorm<:ScalingType end
struct ScaleQ10<:ScalingType end # 10 times the 99% qunatile

function flim_fit(to_fit, ::ScaleStdDev; varargs...)
    println("Scaling Data by the StdDev")
    scale_factor = stddev(to_fit)
    return flim_fit(to_fit; scale_factor=scale_factor, varargs...)
end

function flim_fit(to_fit, ::ScaleMaximum; varargs...)
    println("Scaling Data by the Maximum")
    scale_factor = maximum(to_fit)
    return flim_fit(to_fit; scale_factor=scale_factor, varargs...)
end

function flim_fit(to_fit, ::ScaleMean; varargs...)
    println("Scaling Data by the Mean")
    scale_factor = mean(to_fit)
    return flim_fit(to_fit; scale_factor=scale_factor, varargs...)
end

function flim_fit(to_fit, ::ScaleIQR; varargs...)
    println("Scaling Data by the IQR")
    scale_factor = quantile(to_fit[:], 0.75) - quantile(to_fit[:], 0.25)
    return flim_fit(to_fit; scale_factor=scale_factor, varargs...)
end

function flim_fit(to_fit, ::ScaleNorm; varargs...)
    println("Scaling Data by the Norm")
    scale_factor = sqrt(sum(abs2.(to_fit))) # norm(to_fit)
    return flim_fit(to_fit; scale_factor=scale_factor, varargs...)
end

"""
    flim_fit(to_fit, ::ScaleQ10; varargs...)

    Scale the data by the 10* 99% quantile and fit it.

# Arguments
- `to_fit::Array{Float32, 4}`: The data to fit.
- `varargs...`: Additional arguments to `flim_fit`.

"""
function flim_fit(to_fit, ::ScaleQ10; varargs...)
    println("Scaling Data by the 10* 99% Quantile")
    scale_factor = quantile(to_fit[:], 0.99) * 10 # norm(to_fit)
    return flim_fit(to_fit; scale_factor=scale_factor, varargs...)
end
