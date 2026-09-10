#!/bin/sh
# Which queue has the longest wall clock, and is anything waiting on it?
#
#     ssh <user>@login2.hpc.dtu.dk 'bash -lc "cd ~/PLL_Attempt && sh hpc/queues.sh"'
#
# A shell script rather than a one-liner over ssh on purpose: the nested quoting needed
# to run a for-loop with command substitution through `ssh 'bash -lc "..."'` is exactly
# what broke the log-grep on 2026-09-10, where $(...) expanded on the wrong side and
# bash tried to EXECUTE every log file.
#
# RUNLIMIT is what decides whether a cell can finish at all. Known so far: hpc, epyc and
# rome cap at 24 h; milan at 48 h. Anything longer here is news -- L4_w256 needs ~75 h at
# 8 cores and has nowhere to run today.
printf '%-16s %-12s %-9s %-9s %s\n' QUEUE RUNLIMIT PEND RUN STATUS
for q in $(bqueues 2>/dev/null | tail -n +2 | awk '{print $1}'); do
    lim=$(bqueues -l "$q" 2>/dev/null | grep -A1 'RUNLIMIT' | tail -1 | tr -d ' ')
    row=$(bqueues "$q" 2>/dev/null | tail -1)
    st=$(echo "$row"  | awk '{print $3}')
    pd=$(echo "$row"  | awk '{print $9}')
    rn=$(echo "$row"  | awk '{print $10}')
    printf '%-16s %-12s %-9s %-9s %s\n' "$q" "${lim:-?}" "${pd:-?}" "${rn:-?}" "${st:-?}"
done
echo
echo "Cores per host (a -n 24 job needs one host with 24 free slots):"
lshosts -w 2>/dev/null | awk 'NR==1 || $5 >= 16' | head -20
