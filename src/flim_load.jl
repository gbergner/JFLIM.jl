export load_sdt, load_old_sdt

function load_sdt(fn)
    BioformatsLoader.init(); # Initializes JavaCall with opt and classpath

    nsets = get_num_sets(fn * ".sdt")
    dat = []
    for n in 1:nsets
        println("Loading set $n of $nsets")
        push!(dat, bf_import(fn * ".sdt", order="XYZCT", subset=n)[:,:,:,:,1])
    end
    data = cat(dat..., dims=5); # fix some strange bug in the data load algorithm
    return data
end

function load_old_sdt(fn, sz, stri=1024)
    fd = open(fn)
    dat = read(fd)
    mylen = size(dat)[1]
    spectra = (mylen ÷ (stri*2))
    start = mylen -  spectra * stri * 2
    datr = reinterpret(UInt16, dat[start+1:end])
    data = reshape(datr, (stri, sz[1], sz[2], sz[3]))
    datp = permutedims(data, (2, 3, 4, 1))
    return datp
end
