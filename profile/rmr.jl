using BenchmarkTools
suite = BenchmarkGroup()
function rmr_bench(n)
    rmr = RepeatedMedianRegressor(n)
    y = randn(n)
    mu_rm, beta_rm, sigma_rm = rmr(y)
end
for k = 2:12
    suite["rmr"][2^k] = @benchmarkable rmr_bench(2^$k)
end
tune!(suite)
results = run(suite, verbose = true)

# the time results should be multiplied by roughly 24h/(2^k) since we have to take into account that theses calls will be repeated for a full day analysis
# The following size should be ok
# n = 200-600 (10s to 30 s at 20Hz)
# this is a good compromise between sufficiently small so that the code run fast and sufficiently large to get enough samples for the estimation
