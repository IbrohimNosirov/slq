using Random
Base.Threads
const thread_count = Threads.nthreads()
Threads.foreach(f, channel::Threads.Channel; schedule=Threads.StaticSchedule(), ntasks=Threads.threadpoolsize())

function time_multithread(A::AbstractMatrix{Float64})
        dim = size(A,1)
        x = randn(dim)
        Threads.@spawn for i = 1:10000
                A*x
        end
end

function time_no_thread(A::AbstractMatrix{Float64})
        dim = size(A,1)
        x = randn(dim)
        for i = 1:10000
                A*x
        end
end

let
        # warm-start
        A = randn(10,10)
        time_multithread(A)
        time_no_thread(A)
        A = randn(1000,1000)
        println("thread count ", thread_count)
        println("multithread: ", @elapsed time_multithread(A))
        println("single thread: ", @elapsed time_no_thread(A))
end

let
        n = 20
        c = Channel{Int}(ch -> foreach(i -> put!(ch, i), 1:n), 1)
        d = Channel{Int}(n) do ch
                f = i -> put!(ch, i^2)
                Threads.foreach(f, c)
        end
        collect(d)
end
