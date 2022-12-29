
tk = map(1:10) do θ
    Threads.@spawn pronto(M,[θ],t0,tf,x0,u0,deepcopy(φ))
end

foreach(tk) do tk
    show(fetch(tk))
    sleep(0.01)
end

