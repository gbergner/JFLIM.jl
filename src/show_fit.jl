export show_fit, show_fit_log

function show_fit(to_fit, fit, Δt=12.5/1024; positions=(size(to_fit)[1:2].÷4, size(to_fit)[1:2].÷2 .+1, 3 .*size(to_fit)[1:2].÷4))

    meandat = mean(to_fit, dims=(1, 2, 3, 5))
    meanfit = mean(fit, dims=(1, 2, 3, 5))
    mytime = Δt .* ((0:size(to_fit)[4]-1) .- get_time_max(meanfit)[:])
    h=plot(xlabel="time", ylabel="intensity / a.u.", label="mean measured")

    offsety =  maximum(meandat)
    for coord in positions
        h=plot!(mytime, offsety .+ to_fit[coord...,1,:,1], xlabel="time / ns", ylabel="intensity / a.u.", label="measured")
        h=plot!(mytime, offsety .+ fit[coord...,1,:,1], label="fit")
        offsety += maximum(to_fit[coord...,1,:,1])
        display(h)
    end
    plot!(mytime, offsety .+ meandat[:], label="mean measured")
    plot!(mytime, offsety .+ meanfit[:], label="mean fit")
    plot!(legend = :topright)
    display(h)

end


function show_fit_log(to_fit, fit, Δt=12.5/1024; positions=(size(to_fit)[1:2].÷4, size(to_fit)[1:2].÷2 .+1, 3 .*size(to_fit)[1:2].÷4))
    meandat = mean(to_fit, dims=(1, 2, 5))
    meanfit = mean(fit, dims=(1, 2, 5))
    mytime = Δt .* ((0:size(to_fit)[4]-1) .- get_time_max(meanfit[:,:,1:1,:,:])[:])

    # Create a multipanel plot layout
    n_positions = length(positions)
    layout = @layout [grid(1+ n_positions, 1)]

    # Initialize the multipanel plot
    p = plot(layout=layout)
    # Create the main plot with logarithmic intensity
    for q in axes(meandat,3)
        to_plot = slice(meandat, 3, q)
        @show size(to_plot)
        plot!(p[1], mytime, to_plot[:],  label="mean measured $(q)", yscale=:log10, yrange=(1e-1, 1e4))
        to_plot = slice(meanfit, 3, q)
        plot!(p[1], mytime, to_plot[:], label="mean fit $(q)")
        plot!(p[1], legend = :topright)
    end
    display(p)
    # Add each position to a separate subplot
    for (i, coord) in enumerate(positions)
        @show coord
        mydat = to_fit[coord...,:,:,1]
        myfit = fit[coord...,:,:,1]
        for q in axes(myfit,1)
            @show q
            to_plot = slice(mydat, 1, q)
            @show size(mydat)
            @show size(to_plot)
            plot!(p[i+1], mytime, to_plot[:], label="measured $(q)", yscale=:log10, yrange=(1e-1, 1e4))
            to_plot = slice(myfit, 1, q)
            @show size(myfit)
            @show size(to_plot)
            plot!(p[i+1], mytime, to_plot[:], label="fit $(q)")
        end
    end
    plot!(p[length(positions)+1], xlabel="time / ns", ylabel="intensity / a.u.")

    display(p)
end