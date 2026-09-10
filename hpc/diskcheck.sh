#!/bin/sh
# What is using the quota, and what is safe to delete. READ-ONLY -- deletes nothing.
#
#     ssh <user>@login2.hpc.dtu.dk 'bash -lc "cd ~/PLL_Attempt && sh hpc/diskcheck.sh"'
#
# NOTHING HERE REMOVES A DATASET. Every `data/*.npz` generated before --lhs_seed existed
# is UNREPRODUCIBLE, and deleting one silently invalidates every sweep record that names
# it and every checkpoint trained on it. If space is genuinely short the answer is to
# move a dataset to /work3/$USER and symlink it back -- the code opens datasets by
# relative path, so a symlink is transparent -- not to delete it.
echo "=== quota ==="
getquota_zhome.sh 2>/dev/null || echo "getquota_zhome.sh unavailable"
getquota_work3.sh 2>/dev/null || echo "(no work3 quota tool)"

echo
echo "=== biggest directories under ~/PLL_Attempt ==="
du -sh ~/PLL_Attempt/* 2>/dev/null | sort -rh | head -12

echo
echo "=== datasets, newest first (NEVER delete these without checking) ==="
ls -lah ~/PLL_Attempt/data/*.npz 2>/dev/null | awk '{print $5, $9}'

echo
echo "=== safe to delete: reproducible or pure clutter ==="
printf '%-34s %s\n' "logs/"            "$(du -sh ~/PLL_Attempt/logs 2>/dev/null | cut -f1)  LSF stdout/stderr, regenerated every run"
printf '%-34s %s\n' "hpc/.expanded/"   "$(du -sh ~/PLL_Attempt/hpc/.expanded 2>/dev/null | cut -f1)  rebuilt by submit.sh on every submit"
printf '%-34s %s\n' "pip cache"        "$(du -sh ~/.cache/pip 2>/dev/null | cut -f1)  reclaim with: pip cache purge"
printf '%-34s %s\n' "__pycache__"      "$(du -sh ~/PLL_Attempt/**/__pycache__ 2>/dev/null | cut -f1 | head -1)  regenerated on import"
echo
echo "logs older than 30 days: $(find ~/PLL_Attempt/logs -name '*.out' -mtime +30 2>/dev/null | wc -l) files, $(find ~/PLL_Attempt/logs -name '*.out' -mtime +30 -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)"
echo
echo "To actually reclaim, run these yourself after reading the numbers above:"
echo "  pip cache purge"
echo "  find ~/PLL_Attempt/logs -name '*.out' -mtime +30 -delete"
echo "  find ~/PLL_Attempt/logs -name '*.err' -mtime +30 -size -1k -delete"
echo "  rm -rf ~/PLL_Attempt/hpc/.expanded"
