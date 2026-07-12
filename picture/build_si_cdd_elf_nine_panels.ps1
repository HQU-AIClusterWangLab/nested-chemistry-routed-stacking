Add-Type -AssemblyName System.Drawing

$root = "D:\lunwen\2.1sci"
$srcDir = Join-Path $root "picture\publication_figures\fig4_2d_maps"
$outDir = Join-Path $root "picture\publication_figures"

$rows = @(
    @(
        @{ system = "BSe9"; sample = "BSe9-167"; file = "Fig4_BSe9_167_cdd_elf.png" },
        @{ system = "BSe9"; sample = "BSe9-256"; file = "Fig4_BSe9_256_cdd_elf.png" },
        @{ system = "BSe9"; sample = "BSe9-84";  file = "Fig4_BSe9_84_cdd_elf.png" }
    ),
    @(
        @{ system = "LaSi9"; sample = "LaSi9-1472"; file = "Fig4_LaSi9_1472_cdd_elf.png" },
        @{ system = "LaSi9"; sample = "LaSi9-2546"; file = "Fig4_LaSi9_2546_cdd_elf.png" },
        @{ system = "LaSi9"; sample = "LaSi9-1792"; file = "Fig4_LaSi9_1792_cdd_elf.png" }
    ),
    @(
        @{ system = "LaCu12"; sample = "LaCu12-15"; file = "Fig4_LaCu12_15_cdd_elf.png" },
        @{ system = "LaCu12"; sample = "LaCu12-33"; file = "Fig4_LaCu12_33_cdd_elf.png" },
        @{ system = "LaCu12"; sample = "LaCu12-95"; file = "Fig4_LaCu12_95_cdd_elf.png" }
    )
)

function New-Canvas {
    param(
        [string]$Title,
        [string]$Mode,
        [string]$OutStem
    )

    $images = @()
    foreach ($row in $rows) {
        foreach ($item in $row) {
            $imgPath = Join-Path $srcDir $item.file
            $img = [System.Drawing.Bitmap]::FromFile($imgPath)
            $halfWidth = [int]($img.Width / 2)
            if ($Mode -eq "CDD") {
                $rect = New-Object System.Drawing.Rectangle 0, 0, $halfWidth, $img.Height
            } else {
                $rect = New-Object System.Drawing.Rectangle $halfWidth, 0, ($img.Width - $halfWidth), $img.Height
            }
            $crop = $img.Clone($rect, $img.PixelFormat)
            $img.Dispose()
            $images += [pscustomobject]@{
                System = $item.system
                Sample = $item.sample
                Image  = $crop
            }
        }
    }

    $panelWidth = ($images[0].Image).Width
    $panelHeight = ($images[0].Image).Height
    $leftPad = 90
    $rightPad = 40
    $topPad = 95
    $bottomPad = 40
    $gapX = 32
    $gapY = 40
    $canvasWidth = $leftPad + 3 * $panelWidth + 2 * $gapX + $rightPad
    $canvasHeight = $topPad + 3 * $panelHeight + 2 * $gapY + $bottomPad

    $bmp = New-Object System.Drawing.Bitmap $canvasWidth, $canvasHeight
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $gfx.Clear([System.Drawing.Color]::White)
    $gfx.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $gfx.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $gfx.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

    $titleFont = New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)
    $rowFont = New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Bold)
    $panelFont = New-Object System.Drawing.Font("Arial", 12, [System.Drawing.FontStyle]::Regular)
    $brush = [System.Drawing.Brushes]::Black
    $darkGray = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(70, 70, 70))

    $titleSize = $gfx.MeasureString($Title, $titleFont)
    $gfx.DrawString($Title, $titleFont, $brush, [float](($canvasWidth - $titleSize.Width) / 2), 20)

    for ($r = 0; $r -lt 3; $r++) {
        $rowY = $topPad + $r * ($panelHeight + $gapY)
        $rowLabel = $rows[$r][0].system
        $gfx.DrawString($rowLabel, $rowFont, $brush, 18, [float]($rowY + ($panelHeight / 2) - 12))
        for ($c = 0; $c -lt 3; $c++) {
            $idx = $r * 3 + $c
            $item = $images[$idx]
            $x = $leftPad + $c * ($panelWidth + $gapX)
            $sampleSize = $gfx.MeasureString($item.Sample, $panelFont)
            $gfx.DrawString($item.Sample, $panelFont, $darkGray, [float]($x + ($panelWidth - $sampleSize.Width) / 2), [float]($rowY - 26))
            $gfx.DrawImage($item.Image, $x, $rowY, $panelWidth, $panelHeight)
            $item.Image.Dispose()
        }
    }

    $pngPath = Join-Path $outDir ($OutStem + ".png")
    $tiffPath = Join-Path $outDir ($OutStem + ".tiff")
    $bmp.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Save($tiffPath, [System.Drawing.Imaging.ImageFormat]::Tiff)

    $gfx.Dispose()
    $bmp.Dispose()
}

New-Canvas -Title "Figure S4 | 2D CDD maps for all nine representative structures" -Mode "CDD" -OutStem "FigureS4_cdd_nine_panel"
New-Canvas -Title "Figure S5 | ELF maps for all nine representative structures" -Mode "ELF" -OutStem "FigureS5_elf_nine_panel"
