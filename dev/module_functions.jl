module Foo

bar = ()->"bar"

function setbar!(x)
    global bar
    bar = x
end

function foo()
    global bar
    return bar()
end
end
