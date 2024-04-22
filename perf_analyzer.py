import pstats
from pstats import SortKey

if __name__ == "__main__":
    p = pstats.Stats("perf_stats")
    p.sort_stats(SortKey.TIME).print_stats(30)