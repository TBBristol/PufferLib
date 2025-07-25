import pstats
from pstats import SortKey

stats = (
    pstats.Stats("profile")            # load the saved cProfile output
    .strip_dirs()                      # remove long path prefixes
    .sort_stats(SortKey.CUMULATIVE)    # or SortKey.TIME, see below
)

stats.print_stats(40)                  # show only the 20 “heaviest” functions