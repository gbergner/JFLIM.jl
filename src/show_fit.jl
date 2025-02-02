export show_fit

function show_fit(to_fit, fit, Δt=12.5/1024)

    meandat = mean(to_fit, dims=(1, 2, 3, 5))
    meanfit = mean(fit, dims=(1, 2, 3, 5))
    mytime = Δt .* ((0:size(to_fit)[4]-1) .- get_time_max(meanfit)[:])
    h=plot(mytime, meandat[:], xlabel="time", ylabel="intensity / a.u.", label="mean measured")
    display(h)
    plot!(mytime, meanfit[:], label="mean fit")
    plot!(legend = :topright)
    display(h)

    offsety = 50
    for coord in (size(to_fit)[1:2].÷4, size(to_fit)[1:2].÷2 .+1, 3 .*size(to_fit)[1:2].÷4)
        h=plot!(mytime, offsety .+ to_fit[coord...,1,:,1], xlabel="time / ns", ylabel="intensity / a.u.", label="measured")
        h=plot!(mytime, offsety .+ fit[coord...,1,:,1], label="fit")
        offsety += 50
        display(h)
    end

end
