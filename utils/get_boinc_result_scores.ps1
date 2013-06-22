$workunit=$args[0]
$count=$args[1]

for ($i=1; $i -le $count; $i++) {
Write-Host wget http://centaur.fi.muni.cz:8000/boinc/labak/fetcher.php?filename=$workunit[$i]_0_2
}