
tk = map(2) do θ
    # sleep(0.01)
    Threads.@spawn pronto(M,[θ],t0,tf,x0,u0,deepcopy(φ))
end

foreach(tk) do tk
    show(fetch(tk))
    sleep(0.01)
end

