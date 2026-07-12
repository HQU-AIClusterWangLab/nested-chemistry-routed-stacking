$ErrorActionPreference = "Stop"

$root = "D:\lunwen\2.1sci\picture"
$dataDir = Join-Path $root "figure3_origin_data"

function Get-DoubleArray {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Rows,
        [Parameter(Mandatory = $true)]
        [string]$Column
    )

    $values = New-Object 'System.Collections.Generic.List[Double]'
    foreach ($row in $Rows) {
        $values.Add([double]$row.$Column)
    }
    return ,([double[]]$values.ToArray())
}

function Set-WorksheetColumns {
    param(
        [Parameter(Mandatory = $true)]
        $App,
        [Parameter(Mandatory = $true)]
        [string]$BookName,
        [Parameter(Mandatory = $true)]
        [string[]]$ColumnNames,
        [Parameter(Mandatory = $true)]
        [double[][]]$Columns
    )

    $App.CreatePage(2, $BookName, "origin", 2) | Out-Null
    for ($i = 0; $i -lt $Columns.Count; $i++) {
        $ok = $App.PutWorksheet($BookName, $Columns[$i], 0, $i)
        if (-not $ok) {
            throw "PutWorksheet failed for $BookName column index $i"
        }
    }

    $labtalk = New-Object 'System.Collections.Generic.List[String]'
    $labtalk.Add("win -a $BookName;")
    $labtalk.Add("wks.col1.type=4;")
    $labtalk.Add("wks.col1.lname$=""$($ColumnNames[0])"";")
    for ($i = 1; $i -lt $ColumnNames.Count; $i++) {
        $labtalk.Add("wks.col$($i + 1).type=1;")
        $labtalk.Add("wks.col$($i + 1).lname$=""$($ColumnNames[$i])"";")
    }
    $ok = $App.Execute(($labtalk -join "`n"), $null)
    if (-not $ok) {
        throw "Failed to label worksheet columns for $BookName"
    }
}

function Build-Figure3AProject {
    param(
        [Parameter(Mandatory = $true)]
        $App,
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $rows = Import-Csv (Join-Path $dataDir "figure3_budget_recall_all_systems_for_origin.csv")
    $columnNames = @("K", "AgB8", "AuB8", "LaB8", "LaSe8", "LaCu12", "LaSi9", "BSe9")
    $columns = @()
    foreach ($name in $columnNames) {
        $columns += ,(Get-DoubleArray -Rows $rows -Column $name)
    }

    $App.NewProject() | Out-Null
    Set-WorksheetColumns -App $App -BookName "Figure3AData" -ColumnNames $columnNames -Columns $columns

    $labtalk = @"
win -a Figure3AData;
plotxy iy:=(1,2:end) plot:=200 ogl:=[<new template:=Line name:=Figure3APlot>] legend:=1;
win -a Figure3APlot;
layer -u 1;
layer 78 72 14 12;
doc -e D { set %C -l 11; set %C -w 2; };
layer.x.from = 0;
layer.x.to = 150;
layer.y.from = -0.02;
layer.y.to = 1.05;
label -r title;
label -b 0 -p 2 2 -n figtitleA A Replay recall;
figtitleA.font = font("Arial");
figtitleA.fsize = 20;
legend.fsize = 8;
legend.font = font("Arial");
legend.box = 0;
legend.fillcolor = color(white);
legend.bordercolor = color(white);
legend.transparency = 100;
legend.x1 = 112;
legend.y1 = 0.99;
xb.fsize = 18;
yl.fsize = 18;
xb.box = 0;
yl.box = 0;
xb.font = font("Arial");
yl.font = font("Arial");
label -xb "Replay budget, K";
label -yl "Recall (E_ref <= 0.10 eV)";
"@
    $ok = $App.Execute($labtalk, $null)
    if (-not $ok) {
        throw "Failed to build Figure3A plot"
    }

    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    if (-not $App.Save($OutputPath)) {
        throw "Failed to save $OutputPath"
    }
}

function Build-Figure3BProject {
    param(
        [Parameter(Mandatory = $true)]
        $App,
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $rows = Import-Csv (Join-Path $dataDir "figure3_best_gap_main_systems_for_origin.csv")
    $columnNames = @("K", "LaCu12", "LaSi9", "BSe9")
    $columns = @()
    foreach ($name in $columnNames) {
        $columns += ,(Get-DoubleArray -Rows $rows -Column $name)
    }

    $App.NewProject() | Out-Null
    Set-WorksheetColumns -App $App -BookName "Figure3BData" -ColumnNames $columnNames -Columns $columns

    $labtalk = @"
win -a Figure3BData;
plotxy iy:=(1,2:end) plot:=200 ogl:=[<new template:=Line name:=Figure3BPlot>] legend:=1;
win -a Figure3BPlot;
layer -u 1;
layer 78 72 14 12;
doc -e D { set %C -l 11; set %C -w 2; };
layer.x.from = 0;
layer.x.to = 150;
layer.y.from = -0.03;
layer.y.to = 1.30;
label -r title;
label -b 0 -p 2 2 -n figtitleB B Best-of-K replay gap;
figtitleB.font = font("Arial");
figtitleB.fsize = 20;
legend.fsize = 9;
legend.font = font("Arial");
legend.box = 0;
legend.fillcolor = color(white);
legend.bordercolor = color(white);
legend.transparency = 100;
legend.x1 = 110;
legend.y1 = 1.20;
xb.fsize = 18;
yl.fsize = 18;
xb.box = 0;
yl.box = 0;
xb.font = font("Arial");
yl.font = font("Arial");
label -xb "Replay budget, K";
label -yl "Best-of-K replay gap (eV; cropped)";
"@
    $ok = $App.Execute($labtalk, $null)
    if (-not $ok) {
        throw "Failed to build Figure3B plot"
    }

    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    if (-not $App.Save($OutputPath)) {
        throw "Failed to save $OutputPath"
    }
}

function Build-Figure3CombinedProject {
    param(
        [Parameter(Mandatory = $true)]
        $App,
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $rowsA = Import-Csv (Join-Path $dataDir "figure3_budget_recall_all_systems_for_origin.csv")
    $namesA = @("K", "AgB8", "AuB8", "LaB8", "LaSe8", "LaCu12", "LaSi9", "BSe9")
    $colsA = @()
    foreach ($name in $namesA) {
        $colsA += ,(Get-DoubleArray -Rows $rowsA -Column $name)
    }

    $rowsB = Import-Csv (Join-Path $dataDir "figure3_best_gap_main_systems_for_origin.csv")
    $namesB = @("K", "LaCu12", "LaSi9", "BSe9")
    $colsB = @()
    foreach ($name in $namesB) {
        $colsB += ,(Get-DoubleArray -Rows $rowsB -Column $name)
    }

    $App.NewProject() | Out-Null
    Set-WorksheetColumns -App $App -BookName "Figure3AData" -ColumnNames $namesA -Columns $colsA
    Set-WorksheetColumns -App $App -BookName "Figure3BData" -ColumnNames $namesB -Columns $colsB

    $labtalk = @"
win -a Figure3AData;
plotxy iy:=(1,2:end) plot:=200 ogl:=[<new template:=Line name:=Figure3APlot>] legend:=1;
win -a Figure3APlot;
layer -u 1;
layer 78 72 14 12;
doc -e D { set %C -l 11; set %C -w 2; };
layer.x.from = 0;
layer.x.to = 150;
layer.y.from = -0.02;
layer.y.to = 1.05;
label -r title;
label -b 0 -p 2 2 -n figtitleA A Replay recall;
figtitleA.font = font("Arial");
figtitleA.fsize = 20;
legend.fsize = 8;
legend.font = font("Arial");
legend.box = 0;
legend.fillcolor = color(white);
legend.bordercolor = color(white);
legend.transparency = 100;
legend.x1 = 112;
legend.y1 = 0.99;
xb.fsize = 18;
yl.fsize = 18;
xb.box = 0;
yl.box = 0;
xb.font = font("Arial");
yl.font = font("Arial");
label -xb "Replay budget, K";
label -yl "Recall (E_ref <= 0.10 eV)";

win -a Figure3BData;
plotxy iy:=(1,2:end) plot:=200 ogl:=[<new template:=Line name:=Figure3BPlot>] legend:=1;
win -a Figure3BPlot;
layer -u 1;
layer 78 72 14 12;
doc -e D { set %C -l 11; set %C -w 2; };
layer.x.from = 0;
layer.x.to = 150;
layer.y.from = -0.03;
layer.y.to = 1.30;
label -r title;
label -b 0 -p 2 2 -n figtitleB B Best-of-K replay gap;
figtitleB.font = font("Arial");
figtitleB.fsize = 20;
legend.fsize = 9;
legend.font = font("Arial");
legend.box = 0;
legend.fillcolor = color(white);
legend.bordercolor = color(white);
legend.transparency = 100;
legend.x1 = 110;
legend.y1 = 1.20;
xb.fsize = 18;
yl.fsize = 18;
xb.box = 0;
yl.box = 0;
xb.font = font("Arial");
yl.font = font("Arial");
label -xb "Replay budget, K";
label -yl "Best-of-K replay gap (eV; cropped)";
"@
    $ok = $App.Execute($labtalk, $null)
    if (-not $ok) {
        throw "Failed to build combined Figure3 plot pages"
    }

    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    if (-not $App.Save($OutputPath)) {
        throw "Failed to save $OutputPath"
    }
}

$app = New-Object -ComObject Origin.Application
try {
    Build-Figure3AProject -App $app -OutputPath (Join-Path $root "Figure3A_budget_recall_7systems_origin_rebuilt_20260607.opju")
    Build-Figure3BProject -App $app -OutputPath (Join-Path $root "Figure3B_best_gap_main_systems_origin_rebuilt_20260607.opju")
    Build-Figure3CombinedProject -App $app -OutputPath (Join-Path $root "Figure3_replay_screening_origin_rebuilt_20260607.opju")
}
finally {
    try {
        $app.Exit() | Out-Null
    }
    catch {
    }
}
