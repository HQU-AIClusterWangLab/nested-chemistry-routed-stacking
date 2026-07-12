param(
    [string]$OrcaExe = "D:\ORCA\orca.exe",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Format-Elapsed {
    param([TimeSpan]$Elapsed)
    "{0}day {1}h {2}min {3}s" -f $Elapsed.Days, $Elapsed.Hours, $Elapsed.Minutes, $Elapsed.Seconds
}

if (-not (Test-Path -LiteralPath $OrcaExe)) {
    throw "ORCA executable not found: $OrcaExe"
}

$mpiexec = Get-Command mpiexec -ErrorAction SilentlyContinue
if (-not $mpiexec) {
    Write-Warning "mpiexec was not found in PATH. ORCA inputs with %pal nprocs > 1 may fail. Fix PATH or run single-core inputs."
}

$inputs = Get-ChildItem -LiteralPath . -Recurse -File -Filter "*.inp" |
    Where-Object { $_.FullName -notmatch '\\\.orca_tmp\\' } |
    Sort-Object FullName

if ($inputs.Count -eq 0) {
    Write-Host "No .inp files found under $((Get-Location).Path)"
    exit 0
}

foreach ($inputFile in $inputs) {
    $dir = $inputFile.DirectoryName
    $inputName = $inputFile.Name
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($inputName)
    $outputPath = Join-Path $dir "$stem.out"

    Write-Host ""
    Write-Host "****** Entered $dir folder"
    Write-Host "Running $inputName with $OrcaExe ..."

    if ((-not $Force) -and (Test-Path -LiteralPath $outputPath)) {
        $existing = Get-Content -LiteralPath $outputPath -Tail 200 -ErrorAction SilentlyContinue
        if ($existing -match "ORCA TERMINATED NORMALLY") {
            Write-Host "Skip $inputName`: $stem.out already terminated normally. Use -Force to rerun."
            continue
        }
        Write-Warning "$stem.out exists but is not a normal termination record; overwriting."
    }

    Push-Location -LiteralPath $dir
    try {
        $start = Get-Date
        & $OrcaExe $inputName *> "$stem.out"
        $exitCode = $LASTEXITCODE
        $elapsed = (Get-Date) - $start

        Write-Host ("Running " + (Format-Elapsed $elapsed))

        if ($exitCode -ne 0) {
            Write-Error "$inputName failed with exit code $exitCode; see $stem.out"
        }

        $tail = Get-Content -LiteralPath "$stem.out" -Tail 300 -ErrorAction SilentlyContinue
        if ($tail -match "ORCA TERMINATED NORMALLY") {
            Write-Host "$inputName has finished normally."
        } else {
            Write-Warning "$inputName finished without ORCA normal-termination marker; inspect $stem.out."
        }
    }
    finally {
        Pop-Location
    }
}
